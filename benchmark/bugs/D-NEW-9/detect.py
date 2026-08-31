"""
D-NEW-9: BF16 optimizer lp_grads not cleared after accumulation.

Bug: accumulate_hp_grads_and_remove_lp calls _update_hp_grad with clear_lp_grads=False.
     Low-precision gradients survive into the next forward pass, wasting memory and
     potentially accumulating stale gradients across iterations.
Fix: Set clear_lp_grads=True.

Detection: Hook _update_hp_grad, check the clear_lp_grads argument value.
"""
import os, sys, torch

DS_DIR = os.environ.get("DS_DIR", "")
if DS_DIR:
    sys.path.insert(0, DS_DIR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from deepspeed_train_wrapper import SimpleGPT
import deepspeed

from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

_orig_update = BF16_Optimizer._update_hp_grad
_clear_lp_grads_values = []
_checked = [False]

def _hooked_update(self, lp_param, group_idx, param_idx, clear_lp_grads=False):
    if not _checked[0]:
        _clear_lp_grads_values.append(clear_lp_grads)
    return _orig_update(self, lp_param, group_idx, param_idx, clear_lp_grads=clear_lp_grads)

BF16_Optimizer._update_hp_grad = _hooked_update


def main():
    import torch.distributed as dist
    if not dist.is_initialized():
        deepspeed.init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = SimpleGPT(vocab_size=1024, d_model=64, n_heads=4, n_layers=1).to(torch.bfloat16)

    ds_config = {
        "train_batch_size": 2 * world_size,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
        "zero_optimization": {"stage": 1},
        "bf16": {"enabled": True},
        "data_types": {"grad_accum_dtype": "fp32"},
    }

    engine, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config)

    for step in range(3):
        input_ids = torch.randint(0, 1024, (2, 32), device=engine.device)
        loss, _ = engine(input_ids, labels=input_ids.clone())
        engine.backward(loss)
        engine.step()
        if rank == 0:
            print(f"  step {step}: loss={loss.item():.4f}")

    _checked[0] = True

    if rank == 0:
        false_count = sum(1 for v in _clear_lp_grads_values if not v)
        true_count = sum(1 for v in _clear_lp_grads_values if v)
        total = len(_clear_lp_grads_values)
        print(f"\n{'='*60}")
        print(f"[D-NEW-9] _update_hp_grad clear_lp_grads values ({total} calls):")
        print(f"  clear_lp_grads=True: {true_count}, False: {false_count}")
        if false_count > 0 and true_count == 0:
            print(f"[D-NEW-9] BUG DETECTED: lp_grads never cleared ({false_count} calls with False)")
            print(f"  Stale gradients accumulate across iterations")
        elif true_count > 0:
            print(f"[D-NEW-9] CLEAN: lp_grads properly cleared")
        else:
            print(f"[D-NEW-9] NOTE: _update_hp_grad not called (BF16_Optimizer not used)")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
