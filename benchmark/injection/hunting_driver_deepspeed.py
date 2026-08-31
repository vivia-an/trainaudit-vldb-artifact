"""DeepSpeed hunting driver — clean training, no bug-specific trigger.

Runs a generic Sequential model under deepspeed.initialize for a few steps,
varies (precision, ZeRO stage, gradient_clipping) via env vars, lets every
trainaudit rule observe the run. If any rule fires, prints the contract
block

    [HUNT/trainaudit] BUG DETECTED via N rule(s):
       - <rule_id>: <message>

which `commit_regression_check.py::parse_violations` already understands.

Env knobs (all optional):
  HUNT_DTYPE         fp32 | bf16              (default fp32)
  HUNT_ZERO_STAGE    0 | 1 | 2                (default 0)
  HUNT_CLIP          float, e.g. 0.1 / 1.0    (default 1.0)
  HUNT_STEPS         int, default 5
  HUNT_TIER          T0_PYTORCH | T1_FW_METADATA  (default T1_FW_METADATA)
  DS_DIR             DeepSpeed checkout root  (required when not pre-installed)
"""
import logging
import math
import os
import socket
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "trainaudit_pkg"))

DS_DIR = os.environ.get("DS_DIR", "")
if DS_DIR:
    sys.path.insert(0, DS_DIR)

import torch.distributed.elastic.agent.server.api as _t_api  # noqa: E402

if not hasattr(_t_api, "log"):
    _t_api.log = logging.getLogger("torch.distributed.elastic.agent.server.api")
if not hasattr(_t_api, "_get_socket_with_port"):
    def _get_socket_with_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.listen(1)
        return s
    _t_api._get_socket_with_port = _get_socket_with_port

import trainaudit  # noqa: E402
import deepspeed  # noqa: E402


def _tier(name: str):
    name = (name or "T1_FW_METADATA").upper()
    if hasattr(trainaudit.Tier, name):
        return getattr(trainaudit.Tier, name)
    return trainaudit.Tier.T1_FW_METADATA


def main():
    import torch.distributed as dist
    if not dist.is_initialized():
        deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype = os.environ.get("HUNT_DTYPE", "fp32").lower()
    zero_stage = int(os.environ.get("HUNT_ZERO_STAGE", "0"))
    clip = float(os.environ.get("HUNT_CLIP", "1.0"))
    steps = int(os.environ.get("HUNT_STEPS", "5"))
    tier = _tier(os.environ.get("HUNT_TIER", "T1_FW_METADATA"))

    trainaudit.enable(tier=tier, db_path=":memory:")

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, 32))

    ds_config = {
        "train_batch_size": 4 * world_size,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
        "zero_optimization": {"stage": zero_stage},
        "gradient_clipping": clip,
    }
    if dtype == "bf16":
        ds_config["bf16"] = {"enabled": True}
    elif dtype == "fp16":
        ds_config["fp16"] = {"enabled": True, "initial_scale_power": 8}

    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    trainaudit.snapshot_build(engine.module, engine.optimizer or None)
    device = engine.device

    # Wrap DeepSpeed's own clip_grad_norm_ so trainaudit sees clip events
    # (DeepSpeed bypasses torch.nn.utils.clip_grad_norm_ in many paths).
    import deepspeed.runtime.utils as ds_utils
    import deepspeed.runtime.engine as ds_engine
    _orig_ds_clip = ds_utils.clip_grad_norm_
    _store = trainaudit.get_store()
    from trainaudit.schema import HP_CLIP_GRAD_PRE, HP_CLIP_GRAD_POST

    def _wrapped_ds_clip(parameters, max_norm, norm_type=2, mpu=None):
        params = [p for p in parameters if p.grad is not None]
        with torch.no_grad():
            pre = math.sqrt(sum(p.grad.data.float().norm(2).item() ** 2
                                 for p in params))
        _store.emit(HP_CLIP_GRAD_PRE, {
            "kind": "clip_grad",
            "fn": "deepspeed.runtime.utils.clip_grad_norm_",
            "max_norm": float(max_norm), "pre_norm": pre,
            "n_params": len(params),
        })
        result = _orig_ds_clip(params, max_norm, norm_type=norm_type, mpu=mpu)
        with torch.no_grad():
            post = math.sqrt(sum(p.grad.data.float().norm(2).item() ** 2
                                  for p in params))
        _store.emit(HP_CLIP_GRAD_POST, {
            "kind": "clip_grad",
            "fn": "deepspeed.runtime.utils.clip_grad_norm_",
            "max_norm": float(max_norm), "pre_norm": pre, "post_norm": post,
            "ratio": post / pre if pre > 0 else 1.0,
        })
        return result

    ds_utils.clip_grad_norm_ = _wrapped_ds_clip
    ds_engine.clip_grad_norm_ = _wrapped_ds_clip

    torch_dtype = (torch.bfloat16 if dtype == "bf16"
                   else torch.float16 if dtype == "fp16"
                   else torch.float32)

    for step in range(steps):
        trainaudit.set_step(step)
        x = torch.randn(4, 64, device=device, dtype=torch_dtype)
        out = engine(x)
        loss = out.pow(2).sum()
        engine.backward(loss)
        engine.step()

    if rank == 0:
        results = trainaudit.run_rules()
        print(trainaudit.summarize(results))
        violated = [r for r in results if r.violated]
        if violated:
            print(f"\n[HUNT/trainaudit] BUG DETECTED via {len(violated)} "
                  f"rule(s):")
            for v in violated:
                print(f"   - {v.rule_id}: {v.message}")
        else:
            print(f"\n[HUNT/trainaudit] CLEAN: no rule violations "
                  f"(dtype={dtype} zero={zero_stage} clip={clip})")

    dist.barrier()
    dist.destroy_process_group()
    trainaudit.disable()


if __name__ == "__main__":
    main()
