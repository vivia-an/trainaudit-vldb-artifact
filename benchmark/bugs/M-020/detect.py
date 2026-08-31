"""
M-020: Pipeline parallel silently trains with fewer layers than configured.

Bug: When num_layers is not divisible by pp_size, Megatron uses integer division
     (num_layers // pp_size), silently dropping layers without any error.
     E.g., num_layers=5, pp_size=2 -> 2 layers/rank -> 4 total (1 layer lost).
Fix: Added assertion: num_layers % transformer_pipeline_model_parallel_size == 0.

Invariant: actual total layers across all PP ranks == configured num_layers.
Detection: After first train_step, count transformer layers per rank via
           named_parameters, all_gather totals, compare with config.

Issue: https://github.com/NVIDIA/Megatron-LM/issues/468
"""
import os
import sys
import torch

# Patch: this commit's pretrain_gpt.py calls get_blend_and_blend_per_split
# even with --mock-data, but mock-data doesn't need real data paths.
try:
    from megatron.training import utils as _mutils
    _orig_get_blend = _mutils.get_blend_and_blend_per_split

    def _mock_get_blend(args):
        if getattr(args, "mock_data", False):
            return None, None
        return _orig_get_blend(args)

    _mutils.get_blend_and_blend_per_split = _mock_get_blend
except Exception:
    pass

_ts_mod = None
for _name in ["megatron.training.training", "megatron.training"]:
    try:
        _ts_mod = __import__(_name, fromlist=["train_step"])
        if hasattr(_ts_mod, "train_step"):
            break
        _ts_mod = None
    except ImportError:
        _ts_mod = None

if _ts_mod is None:
    print("[detect] ERROR: cannot find megatron train_step")
    sys.exit(1)

_orig_train_step = _ts_mod.train_step
_checked = [False]


def _patched_train_step(*args, **kwargs):
    result = _orig_train_step(*args, **kwargs)

    if _checked[0]:
        return result
    _checked[0] = True

    model = args[2] if len(args) > 2 else kwargs.get("model")
    if model is None:
        return result

    models = model if isinstance(model, list) else [model]
    total_local_layers = 0
    for m in models:
        while hasattr(m, "module"):
            m = m.module
        layer_indices = set()
        for name, _ in m.named_parameters():
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
        total_local_layers += len(layer_indices)

    if torch.distributed.is_initialized():
        count_t = torch.tensor([total_local_layers], dtype=torch.long, device="cuda")
        gathered = [torch.zeros(1, dtype=torch.long, device="cuda")
                    for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, count_t)
        total_layers = sum(c.item() for c in gathered)
        rank = torch.distributed.get_rank()
    else:
        total_layers = total_local_layers
        rank = 0

    try:
        from megatron.training import get_args
    except ImportError:
        from megatron import get_args
    configured = get_args().num_layers

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[M-020] configured num_layers = {configured}")
        print(f"[M-020] actual total layers    = {total_layers}")
        if total_layers != configured:
            print(f"[M-020] BUG DETECTED: model has {total_layers} layers, "
                  f"missing {configured - total_layers} layer(s)")
        else:
            print(f"[M-020] CLEAN: layer count matches")
        print(f"{'='*60}\n")

    return result


_ts_mod.train_step = _patched_train_step
print("[detect] patched train_step for M-020 layer count check")

exec(open("pretrain_gpt.py").read())
