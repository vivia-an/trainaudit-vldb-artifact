"""
B11 / DeepSpeed PR 5150: clip_grad_norm_ uses torch.max instead of torch.min.

Bug: deepspeed/runtime/utils.py:clip_grad_norm_ has
        clip_coef = torch.max(tmp_tensor, clip_coef)   # buggy: never < 1
     Fix changes max -> min so clip_coef = min(1, max_norm/total_norm).

Trigger: FP32 + ZeRO-0 + gradient_clipping > 0 + total_norm > max_norm.
         Engine._take_model_step calls self.clip_fp32_gradients() which calls
         deepspeed.runtime.utils.clip_grad_norm_ -> the buggy function.

Detection: hook the function, capture (max_norm, total_norm, post_clip_norm).
           buggy: post_clip_norm == pre_clip_norm (>> max_norm)
           fixed: post_clip_norm ~= max_norm
"""
import os, sys, math, socket, logging, torch, torch.nn as nn

DS_DIR = os.environ.get("DS_DIR", "")
if DS_DIR:
    sys.path.insert(0, DS_DIR)

# Compat shim: 005afe12-era DeepSpeed imports `log` and `_get_socket_with_port`
# from torch.distributed.elastic.agent.server.api which were removed in newer torch.
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
import deepspeed.runtime.utils as ds_utils
from deepspeed.runtime.engine import DeepSpeedEngine

_orig_clip = ds_utils.clip_grad_norm_
_records = []


def _hooked_clip(parameters, max_norm, norm_type=2, mpu=None):
    params = [p for p in parameters if p.grad is not None]
    pre_norm = math.sqrt(sum(p.grad.data.float().norm(2).item() ** 2 for p in params))
    total_norm = _orig_clip(params, max_norm, norm_type=norm_type, mpu=mpu)
    post_norm = math.sqrt(sum(p.grad.data.float().norm(2).item() ** 2 for p in params))
    _records.append({
        "max_norm": float(max_norm),
        "pre_norm": pre_norm,
        "post_norm": post_norm,
        "total_norm_returned": float(total_norm) if torch.is_tensor(total_norm) else total_norm,
    })
    return total_norm


ds_utils.clip_grad_norm_ = _hooked_clip
# engine.py imported the symbol by name -> patch the imported reference too
import deepspeed.runtime.engine as ds_engine
ds_engine.clip_grad_norm_ = _hooked_clip


def main():
    import torch.distributed as dist
    if not dist.is_initialized():
        deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(0)
    # Tiny FP32 model. We need real grads with norm > max_norm.
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    max_norm = 0.1
    ds_config = {
        "train_batch_size": 2 * world_size,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
        "zero_optimization": {"stage": 0},
        "gradient_clipping": max_norm,
        # FP32 (no bf16/fp16) so engine takes the clip_fp32_gradients() branch
    }

    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    device = engine.device

    for step in range(2):
        x = torch.randn(2, 8, device=device, dtype=torch.float32)
        # Scale up output so loss-grad easily exceeds max_norm
        out = engine(x) * 1000.0
        loss = out.pow(2).sum()
        engine.backward(loss)
        engine.step()

    if rank == 0:
        print(f"[DETECT] max_norm = {max_norm}")
        for i, r in enumerate(_records):
            print(f"[DETECT] step {i}: pre={r['pre_norm']:.4f} post={r['post_norm']:.4f} returned={r['total_norm_returned']:.4f}")

        # Buggy: post == pre (no clip applied) when pre > max_norm.
        # Fixed: post ~= max_norm.
        bug_hits = 0
        for r in _records:
            if r["pre_norm"] > r["max_norm"] * 2 and r["post_norm"] > r["max_norm"] * 2:
                bug_hits += 1
        if bug_hits > 0:
            print(f"[BUG] {bug_hits}/{len(_records)} clip calls left grad norm >> max_norm (no clipping)")
            print("[RESULT] BUG DETECTED: clip_grad_norm_ used torch.max -> coef floor at 1.0")
        else:
            print("[RESULT] CLEAN: post-clip grad norm <= max_norm")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
