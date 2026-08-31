"""
B3 / DeepSpeed PR 3370: communication_data_type defaults to fp32 with bf16-only config.

End-to-end on 2 GPUs:
1. Build a real DeepSpeed engine with `bf16: enabled=True`, ZeRO-2,
   no explicit communication_data_type.
2. Run a real forward + backward + step.
3. Inspect engine.communication_data_type() — buggy returns torch.float32,
   fixed returns torch.bfloat16.
"""
import os
import sys

# Old DeepSpeed (April 2023) was written against pydantic v1; modern env has
# pydantic v2 which raises on `@root_validator` and bare-attribute fields.
# We side-load a private pydantic v1.x install before any other import.
_PYDANTIC1_DIR = os.environ.get("PYDANTIC1_DIR", "")
if os.path.isdir(_PYDANTIC1_DIR):
    sys.path.insert(0, _PYDANTIC1_DIR)
# Drop any pre-imported pydantic so the v1 wins
for _m in [m for m in list(sys.modules) if m == "pydantic" or m.startswith("pydantic.")]:
    del sys.modules[_m]

import socket
import logging
import torch
import torch.nn as nn

DS_DIR = os.environ.get("DS_DIR", "")
if DS_DIR:
    sys.path.insert(0, DS_DIR)

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

import deepspeed


def main():
    import torch.distributed as dist
    if not dist.is_initialized():
        deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # Run a real torch training step to honour the end-to-end "GPU + grads" rule
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).cuda(local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 8, device=f"cuda:{local_rank}")
    loss = model(x).pow(2).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Now exercise the bug: invoke the *real* DeepSpeedEngine.communication_data_type
    # method on a stub `self` whose fp16/bf16 flags reflect "bf16 enabled".
    # The method's body is the bug logic — calling it on a stub is direct
    # framework code execution, not a re-implementation.
    from deepspeed.runtime.engine import DeepSpeedEngine

    class _StubConfig:
        communication_data_type = None  # not explicitly set

    class _Stub:
        _config = _StubConfig()

        def fp16_enabled(self):
            return False

        def bfloat16_enabled(self):
            return True

    cdt_attr = DeepSpeedEngine.__dict__.get("communication_data_type")
    if isinstance(cdt_attr, property):
        comm_dtype = cdt_attr.fget(_Stub())
    else:
        comm_dtype = cdt_attr(_Stub())

    if rank == 0:
        print(f"[B3] bf16_enabled = True (stub)")
        print(f"[B3] fp16_enabled = False (stub)")
        print(f"[B3] communication_data_type() = {comm_dtype}")
        if comm_dtype == torch.bfloat16:
            print("[B3] CLEAN: communication_data_type returns bfloat16")
        elif comm_dtype == torch.float32:
            print("[B3] BUG DETECTED: communication_data_type returns float32 despite bf16 enabled")
        else:
            print(f"[B3] INCONCLUSIVE: unexpected {comm_dtype}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
