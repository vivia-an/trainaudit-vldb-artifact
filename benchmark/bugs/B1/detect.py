"""
B1 / Megatron issue 599 (commit 3c637fc0d): SwitchMLP router init not wrapped
in get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()).

End-to-end on 2 GPUs (TP=2, num_experts=2):
- Hook CudaRNGStatesTracker.fork to record every (name, time) entry.
- Hook SwitchMLP.__init__ to mark router-init begin/end timestamps.
- After training step: rank 0 reports whether any fork() entry happened
  during router init, and what name was used.
- Cross-rank hash of router.weight as additional evidence.
"""
import os
import sys
import hashlib
import time
import torch
from contextlib import contextmanager

import megatron.core.tensor_parallel.random as tp_random

_orig_fork = tp_random.CudaRNGStatesTracker.fork
_fork_log = []  # list of (timestamp, name)


@contextmanager
def _hooked_fork(self, name=tp_random._MODEL_PARALLEL_RNG_TRACKER_NAME):
    _fork_log.append((time.time(), name))
    with _orig_fork(self, name):
        yield


tp_random.CudaRNGStatesTracker.fork = _hooked_fork


_router_window = {"begin": None, "end": None}

def _wrap_class_init(cls):
    if cls is None or not hasattr(cls, "__init__"):
        return
    _orig = cls.__init__

    def _hooked(self, *args, **kwargs):
        if _router_window["begin"] is None:
            _router_window["begin"] = time.time()
        _orig(self, *args, **kwargs)
        _router_window["end"] = time.time()

    cls.__init__ = _hooked


for _path in (
    "megatron.core.transformer.switch_mlp",
    "megatron.model.transformer",
):
    try:
        _mod = __import__(_path, fromlist=["SwitchMLP"])
        _wrap_class_init(getattr(_mod, "SwitchMLP", None))
    except Exception:
        pass


def _tensor_hash(t):
    data = t.detach().contiguous().cpu().float().numpy().tobytes()
    return int(hashlib.sha256(data).hexdigest()[:15], 16)


_ts_mod = None
for _name in ("megatron.training.training", "megatron.training"):
    try:
        _ts_mod = __import__(_name, fromlist=["train_step"])
        if hasattr(_ts_mod, "train_step"):
            break
        _ts_mod = None
    except ImportError:
        _ts_mod = None

_orig_ts = _ts_mod.train_step
_reported = [False]


def _patched_ts(*args, **kwargs):
    result = _orig_ts(*args, **kwargs)
    if _reported[0]:
        return result
    _reported[0] = True

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    try:
        from megatron.core import parallel_state as ps
        tp_group = ps.get_tensor_model_parallel_group()
        tp_size = ps.get_tensor_model_parallel_world_size()
    except Exception:
        tp_group, tp_size = None, 1

    # 1. Were any forks performed during the SwitchMLP __init__ window?
    forks_during_router = []
    if _router_window["begin"] is not None:
        for ts, name in _fork_log:
            if _router_window["begin"] <= ts <= _router_window["end"]:
                forks_during_router.append(name)

    # 2. Cross-rank hash of router weight (additional evidence)
    model = args[2] if len(args) > 2 else kwargs.get("model")
    m = model[0] if isinstance(model, list) else model
    while hasattr(m, "module"):
        m = m.module
    router_hashes = []
    for name, p in m.named_parameters():
        if "router" in name and "weight" in name:
            h = _tensor_hash(p.data)
            if tp_size > 1 and tp_group is not None:
                ht = torch.tensor([h], dtype=torch.long, device="cuda")
                gathered = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(tp_size)]
                torch.distributed.all_gather(gathered, ht, group=tp_group)
                router_hashes.append((name, [g.item() for g in gathered]))

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[B1] forks during SwitchMLP.__init__: {forks_during_router}")
        for nm, hs in router_hashes:
            print(f"[B1] {nm}: hashes per TP rank = {hs}")
        # Buggy: no fork happened during router init
        # Fixed: a fork with 'data-parallel-rng' name happened
        if any("data-parallel" in n for n in forks_during_router):
            print("[B1] CLEAN: router init wrapped in data-parallel RNG tracker fork")
        else:
            print("[B1] BUG DETECTED: router init never entered get_cuda_rng_tracker().fork(<data-parallel>)")
        print(f"{'='*60}\n")
    return result


_ts_mod.train_step = _patched_ts
print("[detect] B1 hooks installed (RNG tracker fork + SwitchMLP init)")

exec(open("pretrain_gpt.py").read())
