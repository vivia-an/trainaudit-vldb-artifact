"""TrainAudit driver for OC-NEW-22 (OLMo-core commit df72f4ee).

Bug:   `RotaryEmbedding.forward` did not accept `cu_doc_lens`, so when
       packing multiple documents the attention block could not request
       per-document RoPE positions. Positions were assigned globally
       (token-index = absolute position), violating the invariant that
       identical documents must produce identical hidden states regardless
       of position within a packed batch.
Fix:   Add `cu_doc_lens` parameter to `RotaryEmbedding.forward` and use
       `searchsorted` to compute per-document positions.

Detection:
  Construct a `RotaryEmbedding` instance, then verify two things about
  `RotaryEmbedding.forward`:
    1. signature must include the `cu_doc_lens` parameter
    2. invoking forward with `cu_doc_lens=...` must not raise TypeError

  - signature lacks cu_doc_lens or call raises TypeError -> BUG DETECTED
  - signature has cu_doc_lens AND call succeeds          -> CLEAN

Buggy commit: df72f4ee~1
Fixed commit: df72f4eeb79066aef19ccc4989e94c8ed1897052
"""
from __future__ import annotations
import inspect
import os
import sys
import traceback


def _emit(verdict: str, message: str = "") -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    line = f"[OC-NEW-22] {verdict}: {message}" if message else f"[OC-NEW-22] {verdict}"
    print(line, flush=True)


def main() -> None:
    OLMOCORE_DIR = os.environ.get("OLMOCORE_DIR", "")
    if OLMOCORE_DIR:
        for p in (OLMOCORE_DIR, os.path.join(OLMOCORE_DIR, "src")):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

    try:
        import torch
        from olmo_core.nn.rope import RotaryEmbedding
    except Exception as e:
        _emit("FAIL", f"olmocore_import: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    HEAD = 64
    try:
        rope = RotaryEmbedding(head_size=HEAD)
    except Exception as e:
        _emit("FAIL", f"construct: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    # Step 1: signature check.
    try:
        sig = inspect.signature(rope.forward)
    except Exception as e:
        _emit("FAIL", f"signature: {type(e).__name__}: {e}")
        return
    has_cu_doc_lens_param = "cu_doc_lens" in sig.parameters

    # Step 2: behavioural check — does forward actually accept cu_doc_lens?
    B, n_heads, T = 1, 2, 10
    q = torch.randn(B, n_heads, T, HEAD)
    k = torch.randn(B, n_heads, T, HEAD)
    cu = torch.tensor([0, 5, 10], dtype=torch.long)

    behavioural_accepts = False
    behavioural_err = ""
    try:
        rope(q, k, cu_doc_lens=cu)
        behavioural_accepts = True
    except TypeError as e:
        behavioural_err = str(e).splitlines()[0]
    except Exception as e:
        behavioural_err = f"non-TypeError {type(e).__name__}: {e}"

    msg_common = (
        f"signature_has_cu_doc_lens={has_cu_doc_lens_param}, "
        f"forward_accepts_cu_doc_lens={behavioural_accepts}"
    )

    if has_cu_doc_lens_param and behavioural_accepts:
        _emit("CLEAN", "RotaryEmbedding.forward supports cu_doc_lens for packed-doc position reset: " + msg_common)
    elif not has_cu_doc_lens_param:
        _emit(
            "BUG DETECTED",
            "rope_packed_doc_position_reset_invariant: " + msg_common
            + " — RotaryEmbedding.forward signature lacks cu_doc_lens; packed documents get global positions",
        )
    else:
        # has_cu_doc_lens_param=True but behavioural_accepts=False: partial fix?
        _emit(
            "BUG DETECTED",
            "rope_packed_doc_position_reset_invariant_partial: " + msg_common
            + f" — signature added cu_doc_lens but runtime rejected the kwarg: {behavioural_err}",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        _emit("FAIL", f"toplevel: {type(e).__name__}: {e}")
        raise
