"""TrainAudit driver for O-040 (OLMo-core commit db2b6eff, PR #614).

Bug:   In `olmo_core/train/callbacks/speed_monitor.py`, the fallback branch
       "for other GPU types, assume A100" sets
           device_peak_flops_per_second = int(312e12 * dense_correction)
       but the A100 BF16 dense peak is 624 TFLOPS (the 312 listing already
       includes the dense factor; multiplying by `dense_correction=0.5`
       therefore undercounts by 2x — reported MFU comes out 2x too high).
Fix:   Replace `312e12` with `624e12`.

Detection:
  Import `SpeedMonitorCallback`. Construct a mock trainer satisfying its
  `pre_train()` precondition (autocast_precision == bfloat16). Patch
  `torch.cuda.get_device_name` to return a non-{H100,B200} string so the
  fallback branch fires. Call `pre_train()`. Inspect the resulting
  `device_peak_flops_per_second`.

  - value close to int(312e12 * 0.5) = 156e12  -> BUG DETECTED
  - value close to int(624e12 * 0.5) = 312e12  -> CLEAN

Buggy commit: f49e7559 (parent before fix)
Fixed commit: db2b6eff2b8099e30ae43a934aa33725814dfb9a
"""
from __future__ import annotations
import os
import sys
import traceback
from types import SimpleNamespace


def _emit(verdict: str, message: str = "") -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    line = f"[O-040] {verdict}: {message}" if message else f"[O-040] {verdict}"
    print(line, flush=True)


def main() -> None:
    OLMOCORE_DIR = os.environ.get("OLMOCORE_DIR", "")
    if OLMOCORE_DIR:
        # OLMo-core layout is olmo-core/src/olmo_core
        for p in (OLMOCORE_DIR, os.path.join(OLMOCORE_DIR, "src")):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

    try:
        import torch
        from olmo_core.train.callbacks.speed_monitor import SpeedMonitorCallback
        from olmo_core.train.train_module import TransformerTrainModule
    except Exception as e:
        _emit("FAIL", f"olmocore_import: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    # Build a callback instance.
    try:
        cb = SpeedMonitorCallback()
    except Exception as e:
        _emit("FAIL", f"construct_callback: {type(e).__name__}: {e}")
        return

    # Mock train_module must satisfy `isinstance(..., TransformerTrainModule)`.
    # Parent defines several attributes as properties — we override them as
    # plain class attributes so we can set values without touching __init__.
    class _MockTrainModule(TransformerTrainModule):  # type: ignore[misc]
        autocast_precision = torch.bfloat16
        dp_config = None
        model = SimpleNamespace(num_non_embedding_params=1)

        def __init__(self):  # skip parent __init__
            pass

    mock_train_module = _MockTrainModule()
    mock_trainer = SimpleNamespace(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        train_module=mock_train_module,
        dp_process_group=None,
    )
    # SpeedMonitor needs trainer.device.type == "cuda" to enter the flops branch.
    # If no real CUDA available, fake the device attribute.
    if mock_trainer.device.type != "cuda":
        mock_trainer.device = SimpleNamespace(type="cuda")
    cb.trainer = mock_trainer  # type: ignore[assignment]

    # Force the fallback "assume A100" branch by spoofing the GPU name.
    orig_get_device_name = torch.cuda.get_device_name

    def _spoofed(*args, **kwargs):
        return "Tesla V100-SXM2"  # not H100, not B200 -> A100 fallback

    torch.cuda.get_device_name = _spoofed  # type: ignore[assignment]

    try:
        cb.pre_train()
    except Exception as e:
        _emit("FAIL", f"pre_train: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return
    finally:
        torch.cuda.get_device_name = orig_get_device_name  # type: ignore[assignment]

    observed = getattr(cb, "device_peak_flops_per_second", None)
    if observed is None:
        _emit("FAIL", "device_peak_flops_per_second not set after pre_train")
        return

    target_buggy = int(312e12 * 0.5)  # 156 TFLOPS
    target_fixed = int(624e12 * 0.5)  # 312 TFLOPS
    dist_buggy = abs(observed - target_buggy)
    dist_fixed = abs(observed - target_fixed)

    msg_common = (
        f"device_peak_flops_per_second={observed:.3e}, "
        f"target_buggy={target_buggy:.3e}, target_fixed={target_fixed:.3e}"
    )

    if dist_buggy < dist_fixed:
        _emit(
            "BUG DETECTED",
            "olmo_core_a100_peak_flops_2x_invariant: " + msg_common
            + " — A100 fallback uses 312e12 (sparsity-inclusive) instead of 624e12 (dense), MFU reported 2x too high",
        )
    else:
        _emit("CLEAN", "A100 fallback uses 624e12 dense spec: " + msg_common)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        _emit("FAIL", f"toplevel: {type(e).__name__}: {e}")
        raise
