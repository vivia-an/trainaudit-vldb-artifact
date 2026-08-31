"""TrainAudit driver for O-NEW-5 (OLMo commit ce5f30d5).

Bug:   In `olmo/initialization.py`, the `full_megatron` init branch for
       `ModuleType.emb` sets `std = config.init_std`, missing the
       `* sqrt(config.d_model)` factor. Embeddings end up sqrt(d_model)x
       smaller than intended.
Fix:   `std = config.init_std * math.sqrt(config.d_model)`.

Detection:
  Import `init_weights` + `ModuleType` from `olmo.initialization`. Build a
  minimal `ModelConfig` with `init_fn=full_megatron`, d_model=256, init_std=0.02.
  Run `init_weights(config, nn.Embedding(...), type_of_module=ModuleType.emb)`.
  Compare the realised tensor std against the two candidate target stds.
  - actual_std closer to `init_std`                       -> BUG DETECTED
  - actual_std closer to `init_std * sqrt(d_model)`       -> CLEAN

Buggy commit: ce5f30d5^
Fixed commit: ce5f30d5
"""
from __future__ import annotations
import math
import os
import sys
import traceback


def _emit(verdict: str, message: str = "") -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    line = f"[O-NEW-5] {verdict}: {message}" if message else f"[O-NEW-5] {verdict}"
    print(line, flush=True)


def main() -> None:
    OLMO_DIR = os.environ.get("OLMO_DIR", "")
    if OLMO_DIR:
        sys.path.insert(0, OLMO_DIR)

    try:
        import torch
        import torch.nn as nn
        from olmo.initialization import init_weights, ModuleType
        from olmo.config import ModelConfig, InitFnType
    except Exception as e:
        _emit("FAIL", f"olmo_import: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    torch.manual_seed(42)
    config = ModelConfig(
        d_model=256,
        n_heads=4,
        n_layers=2,
        vocab_size=50304,
        embedding_size=50304,
        init_std=0.02,
        init_fn=InitFnType.full_megatron,
        init_cutoff_factor=3,
    )
    module = nn.Embedding(config.vocab_size, config.d_model)

    try:
        init_weights(config, module, type_of_module=ModuleType.emb)
    except Exception as e:
        _emit("FAIL", f"init_weights: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    actual_std = float(module.weight.std().item())
    target_buggy = config.init_std
    target_fixed = config.init_std * math.sqrt(config.d_model)
    dist_buggy = abs(actual_std - target_buggy)
    dist_fixed = abs(actual_std - target_fixed)

    msg_common = (
        f"d_model={config.d_model}, init_std={config.init_std}, "
        f"actual_emb_std={actual_std:.4f}, "
        f"target_buggy={target_buggy:.4f}, "
        f"target_fixed={target_fixed:.4f}"
    )

    if dist_buggy < dist_fixed:
        _emit(
            "BUG DETECTED",
            "olmo_emb_init_std_sqrt_dmodel_invariant: " + msg_common
            + f" — emb std {math.sqrt(config.d_model):.0f}x smaller than spec",
        )
    else:
        _emit(
            "CLEAN",
            "init_weights for ModuleType.emb scaled by sqrt(d_model): " + msg_common,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        _emit("FAIL", f"toplevel: {type(e).__name__}: {e}")
        raise
