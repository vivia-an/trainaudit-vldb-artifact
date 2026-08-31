"""B11 via TrainAudit Phase 1 (T0).

Same trigger as detect.py (FP32 + ZeRO-0 + gradient_clipping=0.1, grads >> max_norm),
but instead of hand-rolled hooks we just enable TrainAudit T0 and let
T0-clip-grad-bounded fire automatically.
"""
import os
import socket
import logging
import sys
import torch
import torch.nn as nn

# 1) point to local trainaudit package
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "trainaudit_pkg"))

DS_DIR = os.environ.get("DS_DIR", "")
if DS_DIR:
    sys.path.insert(0, DS_DIR)

# 2) compat shim for old DeepSpeed elastic agent (same as detect.py)
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

import trainaudit
import deepspeed


def main():
    import torch.distributed as dist
    if not dist.is_initialized():
        deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 3) enable trainaudit T0 BEFORE building optimizer/engine
    trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path=":memory:")

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    max_norm = 0.1
    ds_config = {
        "train_batch_size": 2 * world_size,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
        "zero_optimization": {"stage": 0},
        "gradient_clipping": max_norm,
        # FP32 path -> engine calls clip_fp32_gradients() -> ds_utils.clip_grad_norm_
    }

    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    trainaudit.snapshot_build(engine.module, engine.optimizer or None)
    device = engine.device

    # also patch deepspeed.runtime.utils.clip_grad_norm_ AND engine import to forward
    # to torch.nn.utils.clip_grad_norm_ (which trainaudit hooks). DeepSpeed's clip
    # impl is the buggy/fixed one we want to test.
    # NOTE: we rely on trainaudit's wrap of torch.nn.utils.clip_grad_norm_;
    # DeepSpeed's own ds_utils.clip_grad_norm_ is a separate function — to catch
    # the bug at T0 we need to also wrap that path.
    import deepspeed.runtime.utils as ds_utils
    import deepspeed.runtime.engine as ds_engine
    import math

    _orig_ds_clip = ds_utils.clip_grad_norm_
    _store = trainaudit.get_store()
    from trainaudit.schema import HP_CLIP_GRAD_PRE, HP_CLIP_GRAD_POST

    def _wrapped_ds_clip(parameters, max_norm, norm_type=2, mpu=None):
        params = [p for p in parameters if p.grad is not None]
        with torch.no_grad():
            pre = math.sqrt(sum(p.grad.data.float().norm(2).item() ** 2 for p in params))
        _store.emit(HP_CLIP_GRAD_PRE, {
            "kind": "clip_grad", "fn": "deepspeed.runtime.utils.clip_grad_norm_",
            "max_norm": float(max_norm), "pre_norm": pre, "n_params": len(params),
        })
        result = _orig_ds_clip(params, max_norm, norm_type=norm_type, mpu=mpu)
        with torch.no_grad():
            post = math.sqrt(sum(p.grad.data.float().norm(2).item() ** 2 for p in params))
        _store.emit(HP_CLIP_GRAD_POST, {
            "kind": "clip_grad", "fn": "deepspeed.runtime.utils.clip_grad_norm_",
            "max_norm": float(max_norm), "pre_norm": pre, "post_norm": post,
            "ratio": post / pre if pre > 0 else 1.0,
        })
        return result

    ds_utils.clip_grad_norm_ = _wrapped_ds_clip
    ds_engine.clip_grad_norm_ = _wrapped_ds_clip

    for step in range(2):
        trainaudit.set_step(step)
        x = torch.randn(2, 8, device=device, dtype=torch.float32)
        out = engine(x) * 1000.0
        loss = out.pow(2).sum()
        engine.backward(loss)
        engine.step()

    if rank == 0:
        results = trainaudit.run_rules()
        print(trainaudit.summarize(results))
        violated = [r for r in results if r.violated]
        if violated:
            print(f"\n[B11/trainaudit] BUG DETECTED via {len(violated)} rule(s):")
            for v in violated:
                print(f"   - {v.rule_id}: {v.message}")
        else:
            print("\n[B11/trainaudit] CLEAN: no rule violations")

    dist.barrier()
    dist.destroy_process_group()
    trainaudit.disable()


if __name__ == "__main__":
    main()
