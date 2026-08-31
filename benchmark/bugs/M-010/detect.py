"""
M-010: MoE aux_loss accumulated twice with activation checkpointing.

Bug: When --moe-layer-recompute is enabled, the recomputed forward pass calls
     save_to_aux_losses_tracker() again, doubling the aux_loss contribution.
Fix: Guard aux_loss computation with torch.is_grad_enabled() to skip during
     the no_grad recompute forward.

Invariant: save_to_aux_losses_tracker() must be called exactly once per
           (step, layer, loss_name). More than once = bug.
Detection: Hook save_to_aux_losses_tracker, count calls per step/layer.

Issue: https://github.com/NVIDIA/Megatron-LM/issues/784
"""
import atexit
import json
import os
import sys
from collections import Counter

import megatron.core.transformer.moe.moe_utils as moe_utils

_orig_save = moe_utils.save_to_aux_losses_tracker
_call_log = []
_step = [0]


def _hooked_save(name, loss, layer_number, num_layers, **kwargs):
    _call_log.append({"name": name, "layer": layer_number, "step": _step[0]})
    return _orig_save(name, loss, layer_number, num_layers, **kwargs)


moe_utils.save_to_aux_losses_tracker = _hooked_save

try:
    import megatron.core.transformer.moe.router as router_mod
    router_mod.save_to_aux_losses_tracker = _hooked_save
except Exception:
    pass

import megatron.training.training as mtt

_orig_ts = mtt.train_step


def _patched_ts(*args, **kwargs):
    result = _orig_ts(*args, **kwargs)
    _step[0] += 1
    return result


mtt.train_step = _patched_ts
print("[detect] patched train_step and save_to_aux_losses_tracker for M-010")


def _report():
    rank = 0
    try:
        import torch
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
    except Exception:
        pass

    counts = Counter()
    for entry in _call_log:
        counts[(entry["step"], entry["layer"], entry["name"])] += 1

    violations = [
        {"step": s, "layer": l, "name": n, "count": c}
        for (s, l, n), c in sorted(counts.items()) if c > 1
    ]

    if rank == 0:
        print(f"\n{'='*60}")
        if violations:
            print(f"[M-010] BUG DETECTED: {len(violations)} violations "
                  f"(aux_loss called {violations[0]['count']}x per step)")
            for v in violations[:5]:
                print(f"  step={v['step']} layer={v['layer']} "
                      f"name={v['name']} count={v['count']}")
        else:
            print(f"[M-010] CLEAN: aux_loss called 1x per step/layer")
        print(f"{'='*60}\n")

    output_dir = os.environ.get("DETECT_OUTPUT_DIR", "/tmp")
    out_path = os.path.join(output_dir, f"m010_detect_rank{rank}.json")
    with open(out_path, "w") as f:
        json.dump({
            "bug_id": "M-010",
            "rank": rank,
            "total_calls": len(_call_log),
            "violations": violations,
            "detected": len(violations) > 0,
        }, f, indent=2)


atexit.register(_report)

exec(open("pretrain_gpt.py").read())
