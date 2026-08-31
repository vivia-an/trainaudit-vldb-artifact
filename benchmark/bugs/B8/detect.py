"""
B8 / DeepSpeed PR 7551: InferenceEngine creates EP groups with num_experts not ep_size.

End-to-end: build a tiny model with a deepspeed.moe.MoE layer, call
deepspeed.init_inference with moe.ep_size != moe.num_experts. Hook
InferenceEngine._create_ep_parallel_group to capture the integer that was
passed in. Buggy passes config.moe.moe_experts (== num_experts);
fixed passes config.moe.ep_size.
"""
import os
import sys
import socket
import logging
import torch
import torch.nn as nn

DS_DIR = os.environ.get("DS_DIR", "")
if DS_DIR:
    sys.path.insert(0, DS_DIR)

# Compat shim for older DeepSpeed elastic agent
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
from deepspeed.inference.engine import InferenceEngine
from deepspeed.moe.layer import MoE

_orig_create = InferenceEngine._create_ep_parallel_group
_capture = {"args": None}


def _hooked_create(self, moe_experts):
    _capture["args"] = moe_experts
    try:
        return _orig_create(self, moe_experts)
    except Exception as e:  # noqa: BLE001
        _capture["error"] = f"{type(e).__name__}: {e}"
        return None


InferenceEngine._create_ep_parallel_group = _hooked_create


class TinyMoEModel(nn.Module):
    def __init__(self, hidden, num_experts, ep_size):
        super().__init__()
        self.fc_in = nn.Linear(hidden, hidden)
        self.moe = MoE(
            hidden_size=hidden,
            expert=nn.Linear(hidden, hidden),
            num_experts=num_experts,
            ep_size=ep_size,
            k=1,
        )
        self.fc_out = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.fc_out(self.moe(self.fc_in(x))[0])


def main():
    import torch.distributed as dist
    if not dist.is_initialized():
        deepspeed.init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    NUM_EXPERTS = 8
    EP_SIZE = world_size  # legitimate ep_size
    assert NUM_EXPERTS != EP_SIZE, "Pick distinct values to disambiguate"

    model = TinyMoEModel(hidden=32, num_experts=NUM_EXPERTS, ep_size=EP_SIZE)
    model.eval()

    config = {
        "tensor_parallel": {"tp_size": 1},
        "dtype": "fp32",
        "moe": {"enabled": True, "ep_size": EP_SIZE, "moe_experts": [NUM_EXPERTS]},
    }
    try:
        engine = deepspeed.init_inference(model=model, config=config)
    except Exception as e:  # noqa: BLE001
        if "error" not in _capture:
            _capture["error"] = f"init_inference {type(e).__name__}: {e}"

    if rank == 0:
        arg = _capture.get("args")
        err = _capture.get("error")
        print(f"[B8] world_size={world_size} num_experts={NUM_EXPERTS} ep_size={EP_SIZE}")
        print(f"[B8] _create_ep_parallel_group called with: {arg}")
        if err:
            print(f"[B8] init_inference error: {err}")
        # Buggy: passed [NUM_EXPERTS]; Fixed: passed EP_SIZE.
        if arg == EP_SIZE:
            print("[B8] CLEAN: argument equals ep_size")
        elif arg == [NUM_EXPERTS] or arg == NUM_EXPERTS:
            print("[B8] BUG DETECTED: argument equals num_experts (not ep_size)")
        else:
            print("[B8] INCONCLUSIVE: argument not recognized")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
