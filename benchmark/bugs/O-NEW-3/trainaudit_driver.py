"""TrainAudit driver for O-NEW-3 (OLMo commit a57f3803).

Bug:   In `olmo/train.cross_entropy_loss`, when reduction='mean' and
       compute_z_loss=True, `z_squared` is computed as
       `z_squared / (labels != ignore_index).mean()`. This divides a
       per-token tensor by a scalar, producing a per-token tensor — not
       the scalar that the mean reduction promised.
Fix:   `z_squared = (z_squared * (labels != ignore_index)).mean()`, which
       returns a 0-dim scalar tensor.

Detection:
  Call `cross_entropy_loss(logits, labels, reduction='mean',
  compute_z_loss=True)` and inspect `z_loss.ndim`.
  - z_loss.ndim != 0  -> BUG DETECTED
  - z_loss.ndim == 0  -> CLEAN

Buggy commit: a57f3803^
Fixed commit: a57f380332e7021755d0a36cf79406b4423cf361
"""
from __future__ import annotations
import os
import sys
import traceback


def _emit(verdict: str, message: str = "") -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    line = f"[O-NEW-3] {verdict}: {message}" if message else f"[O-NEW-3] {verdict}"
    print(line, flush=True)


def main() -> None:
    OLMO_DIR = os.environ.get("OLMO_DIR", "")
    if OLMO_DIR:
        sys.path.insert(0, OLMO_DIR)

    try:
        import torch
        from olmo.train import cross_entropy_loss
    except Exception as e:
        _emit("FAIL", f"olmo_import: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    torch.manual_seed(0)
    V = 32  # tiny vocab
    B, T = 2, 5
    logits = torch.randn(B * T, V)
    labels = torch.randint(0, V, (B * T,))

    try:
        loss, z_loss = cross_entropy_loss(
            logits, labels,
            ignore_index=-100,
            reduction="mean",
            compute_z_loss=True,
        )
    except RuntimeError as e:
        # On modern PyTorch the buggy formula `z_squared / (labels != ignore_index).mean()`
        # invokes `.mean()` on a Bool tensor, which raises
        #   RuntimeError: mean(): could not infer output dtype.
        # Older PyTorch versions silently coerced this to a float and produced a
        # wrong-shape tensor (the original silent bug). Either way the invariant
        # — "z_loss is a scalar under reduction='mean'" — is violated.
        msg = str(e)
        if "could not infer output dtype" in msg or "Bool" in msg:
            _emit(
                "BUG DETECTED",
                "olmo_zloss_mean_reduction_invariant: cross_entropy_loss raised "
                f"RuntimeError on Bool mask mean ({msg.splitlines()[0]}); buggy "
                "formula divides z_squared by Bool.mean() which is the wrong reduction",
            )
            return
        _emit("FAIL", f"cross_entropy_loss: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return
    except Exception as e:
        _emit("FAIL", f"cross_entropy_loss: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    if z_loss is None:
        _emit("FAIL", "z_loss is None (compute_z_loss flag ignored)")
        return

    msg_common = (
        f"reduction='mean', compute_z_loss=True; "
        f"loss.ndim={loss.ndim}, z_loss.ndim={z_loss.ndim}, "
        f"z_loss.shape={tuple(z_loss.shape)}"
    )

    if z_loss.ndim != 0:
        _emit(
            "BUG DETECTED",
            "olmo_zloss_mean_reduction_invariant: " + msg_common
            + " (expected scalar, got tensor — buggy formula divides z_squared by a scalar mean instead of masking)",
        )
    else:
        _emit("CLEAN", "z_loss is a 0-dim scalar tensor: " + msg_common)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        _emit("FAIL", f"toplevel: {type(e).__name__}: {e}")
        raise
