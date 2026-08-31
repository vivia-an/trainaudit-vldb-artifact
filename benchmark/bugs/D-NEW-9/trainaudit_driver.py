"""TrainAudit driver for D-NEW-9 (DeepSpeed PR #5328).

Bug:   `BF16_Optimizer.accumulate_hp_grads_and_remove_lp` calls
       `self._update_hp_grad(..., clear_lp_grads=False)`. After accumulating
       into the hp grad copy, the lp grad should be cleared (it isn't),
       so the next iteration's forward sees stale lp_grad memory.
Fix:   Flip the kwarg to `clear_lp_grads=True`.

Detection:
  Import `BF16_Optimizer.accumulate_hp_grads_and_remove_lp`. Hook
  `_update_hp_grad` on a mock self. Invoke the method with dummy args.
  Inspect the captured `clear_lp_grads` value.
  - clear_lp_grads == False -> BUG DETECTED
  - clear_lp_grads == True  -> CLEAN

Buggy commit: 40009eb1c7a4ef27d00f36fe4f97aeae2e315c0e^
Fixed commit: 40009eb1c7a4ef27d00f36fe4f97aeae2e315c0e
"""
from __future__ import annotations
import os
import sys
import socket
import logging
import traceback


def _emit(verdict: str, message: str = "") -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    line = f"[D-NEW-9] {verdict}: {message}" if message else f"[D-NEW-9] {verdict}"
    print(line, flush=True)


def main() -> None:
    DS_DIR = os.environ.get("DS_DIR", "")
    if DS_DIR:
        sys.path.insert(0, DS_DIR)

    # Compat shims used by older DeepSpeed.
    import torch.distributed.elastic.agent.server.api as _t_api
    if not hasattr(_t_api, "log"):
        _t_api.log = logging.getLogger("torch.distributed.elastic.agent.server.api")
    if not hasattr(_t_api, "_get_socket_with_port"):
        def _get_socket_with_port():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))
            s.listen(1)
            return s
        _t_api._get_socket_with_port = _get_socket_with_port

    try:
        from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
    except Exception as e:
        _emit("FAIL", f"deepspeed_import: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    method = getattr(BF16_Optimizer, "accumulate_hp_grads_and_remove_lp", None)
    if method is None:
        _emit("FAIL", "method_missing: BF16_Optimizer.accumulate_hp_grads_and_remove_lp not present")
        return

    captured: list = []

    class _MockSelf:
        immediate_grad_update = True

        def _update_hp_grad(self, lp_param, group_idx, param_idx, clear_lp_grads):
            captured.append(clear_lp_grads)

    try:
        method(_MockSelf(), lp_param=None, group_idx=0, param_idx=0)
    except TypeError:
        # Older signature might be positional-only
        try:
            method(_MockSelf(), None, 0, 0)
        except Exception as e:
            _emit("FAIL", f"method_call: {type(e).__name__}: {e}")
            return
    except Exception as e:
        _emit("FAIL", f"method_call: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    if not captured:
        _emit("FAIL", "no_update_hp_grad_observed (method shape changed)")
        return

    val = captured[0]
    if val is False:
        _emit(
            "BUG DETECTED",
            "bf16_lp_grad_clear_invariant: "
            "accumulate_hp_grads_and_remove_lp invoked _update_hp_grad with "
            f"clear_lp_grads={val} (should be True per the hook's name 'remove_lp')",
        )
    elif val is True:
        _emit("CLEAN", "accumulate_hp_grads_and_remove_lp passes clear_lp_grads=True")
    else:
        _emit("FAIL", f"unexpected_clear_lp_grads_value: {val!r}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        _emit("FAIL", f"toplevel: {type(e).__name__}: {e}")
        raise
