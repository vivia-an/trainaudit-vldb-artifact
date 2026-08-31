"""TrainAudit driver for D-029 (DeepSpeed PR #5220).

Bug:   `DeepSpeedZeroOptimizer._average_expert_grad_norms` calls
       `dist.all_reduce(scaled_norm_tensor, ...)` without first moving the
       tensor to the accelerator when `self.device == 'cpu'` (CPU offload).
       In the buggy version the tensor stays on CPU during the all_reduce;
       in the fixed version it is moved to GPU first, all-reduced, then moved
       back to CPU.
Fixed commit: e6e8c1378de035df59034d09373b44af3319b6d7
Buggy commit: 3e06a154b4d6a2f93c2c5504f52a932184def112 (parent of the fix)

Detection:
  Hook `torch.distributed.all_reduce`. Call
  `DeepSpeedZeroOptimizer._average_expert_grad_norms` with a single MoE
  param-group whose norm tensor is on CPU. Inspect the tensor's device at
  the moment of the all_reduce call.
  - device.type == 'cpu' before all_reduce  -> BUG DETECTED
  - device.type == 'cuda'                   -> CLEAN
Contract:
  Rank-0 stdout prints exactly one of:
    [D-029] BUG DETECTED: <rule_id>: <message>
    [D-029] CLEAN: <message>
    [D-029] FAIL: <stage>: <error>
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
    line = f"[D-029] {verdict}: {message}" if message else f"[D-029] {verdict}"
    print(line, flush=True)


def main() -> None:
    DS_DIR = os.environ.get("DS_DIR", "")
    if DS_DIR:
        sys.path.insert(0, DS_DIR)

    # Compat shim for old elastic agent API used by some buggy commits.
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
        import torch
        import torch.distributed as torch_dist
    except Exception as e:
        _emit("FAIL", f"torch_import: {type(e).__name__}: {e}")
        return

    # Init a 1-rank torch process group first; deepspeed.init_distributed will
    # piggy-back on it.
    if not torch_dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29800 + (os.getpid() % 100)))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        try:
            torch_dist.init_process_group(backend="gloo", rank=0, world_size=1)
        except Exception as e:
            _emit("FAIL", f"dist_init: {type(e).__name__}: {e}")
            return

    # Import the buggy/fixed method via DeepSpeed; also initialise deepspeed.comm.
    try:
        import deepspeed
        deepspeed.init_distributed(dist_backend="gloo", auto_mpi_discovery=False)
        from deepspeed import comm as dist
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    except Exception as e:
        _emit("FAIL", f"deepspeed_import: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    method = getattr(DeepSpeedZeroOptimizer, "_average_expert_grad_norms", None)
    if method is None:
        _emit("FAIL", "method_missing: DeepSpeedZeroOptimizer._average_expert_grad_norms not found in this checkout")
        return

    # Hook deepspeed.comm.{all_reduce, get_world_size} so we can observe the
    # tensor device at call time without firing real collectives or relying on
    # multi-rank topology.
    device_observed: list[dict] = []
    orig_all_reduce = dist.all_reduce
    orig_get_world_size = dist.get_world_size

    def _hooked_all_reduce(tensor, *args, **kwargs):
        try:
            device_observed.append({"device_type": tensor.device.type})
        except Exception:
            device_observed.append({"device_type": "unknown"})
        return None

    def _hooked_get_world_size(group=None, *args, **kwargs):
        return 2  # pretend DP > 1 so the buggy code path is exercised

    dist.all_reduce = _hooked_all_reduce  # type: ignore[assignment]
    dist.get_world_size = _hooked_get_world_size  # type: ignore[assignment]

    # Construct a minimal "self" with only the fields _average_expert_grad_norms
    # reads: device, is_moe_param_group, real_dp_process_group.
    class _MockZero:
        pass
    mock = _MockZero()
    mock.device = "cpu"  # CPU offload
    mock.is_moe_param_group = [True]
    mock.real_dp_process_group = [None]  # hook ignores the group argument

    norm_groups = [torch.tensor(1.5)]  # CPU tensor (grad-norm scalar)

    try:
        method(mock, norm_groups)
    except Exception as e:
        _emit("FAIL", f"method_call: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return
    finally:
        dist.all_reduce = orig_all_reduce  # type: ignore[assignment]
        dist.get_world_size = orig_get_world_size  # type: ignore[assignment]

    if not device_observed:
        _emit("FAIL", "no_all_reduce_observed (method shape changed)")
        return

    first = device_observed[0]
    if first.get("device_type") == "cpu":
        _emit(
            "BUG DETECTED",
            "moe_cpu_offload_grad_norm_device_invariant: "
            f"_average_expert_grad_norms passed device='{first['device_type']}' tensor to dist.all_reduce "
            "(should be on accelerator under CPU offload + MoE)",
        )
    elif first.get("device_type") == "cuda":
        _emit(
            "CLEAN",
            "_average_expert_grad_norms moved scaled_norm_tensor to accelerator before all_reduce",
        )
    else:
        _emit("FAIL", f"unexpected_device_type: {first.get('device_type')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        _emit("FAIL", f"toplevel: {type(e).__name__}: {e}")
        raise
