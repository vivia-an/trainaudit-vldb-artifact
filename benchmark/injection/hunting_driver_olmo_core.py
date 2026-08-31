"""OLMo-core hunting driver — clean train step under trainaudit.

Builds a small Linear stack, picks a scheduler/optimizer combination,
runs forward + backward + optim.step, lets every trainaudit rule observe.

Env knobs:
  HUNT_OPTIM       adamw | skip_step_adamw   (default skip_step_adamw)
  HUNT_DECAY       cosine | sqrt | none      (default sqrt)
  HUNT_STEPS       int                       (default 6)
  HUNT_TIER        T0_PYTORCH | T1_FW_METADATA  (default T1_FW_METADATA)
  OLMO_CORE_DIR    OLMo-core checkout root, prepended to sys.path/src
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "trainaudit_pkg"))

OLMO_CORE_DIR = os.environ.get("OLMO_CORE_DIR", "")
if OLMO_CORE_DIR:
    sys.path.insert(0, os.path.join(OLMO_CORE_DIR, "src"))

import trainaudit  # noqa: E402


def _tier(name: str):
    name = (name or "T1_FW_METADATA").upper()
    if hasattr(trainaudit.Tier, name):
        return getattr(trainaudit.Tier, name)
    return trainaudit.Tier.T1_FW_METADATA


def _build_optimizer(model, name):
    name = (name or "skip_step_adamw").lower()
    if name == "skip_step_adamw":
        try:
            from olmo_core.optim.adamw import SkipStepAdamWConfig
            return SkipStepAdamWConfig(lr=1e-3, betas=(0.9, 0.999)).build(model)
        except Exception as e:
            print(f"[HUNT] SkipStepAdamW import failed ({e}); fall back to AdamW")
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def _build_scheduler(opt, name, total):
    name = (name or "sqrt").lower()
    if name == "sqrt":
        try:
            from olmo_core.optim.scheduler import SqrtScheduler
            return SqrtScheduler(warmup=2)
        except Exception:
            pass
    if name == "cosine":
        try:
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
        except Exception:
            pass
    return None


def main():
    optim_name = os.environ.get("HUNT_OPTIM", "skip_step_adamw")
    decay_name = os.environ.get("HUNT_DECAY", "sqrt")
    n_steps = int(os.environ.get("HUNT_STEPS", "6"))
    tier = _tier(os.environ.get("HUNT_TIER", "T1_FW_METADATA"))

    trainaudit.enable(tier=tier, db_path=":memory:")
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, 64))
    opt = _build_optimizer(model, optim_name)
    sch = _build_scheduler(opt, decay_name, n_steps)

    trainaudit.snapshot_build(model, opt)

    for step in range(n_steps):
        trainaudit.set_step(step)
        x = torch.randn(8, 64)
        loss = model(x).pow(2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()
        if sch is not None and hasattr(sch, "step"):
            try:
                sch.step()
            except Exception:
                pass

    results = trainaudit.run_rules()
    print(trainaudit.summarize(results))
    violated = [r for r in results if r.violated]
    if violated:
        print(f"\n[HUNT/trainaudit] BUG DETECTED via {len(violated)} rule(s):")
        for v in violated:
            print(f"   - {v.rule_id}: {v.message}")
            if v.evidence:
                print(f"     evidence: {v.evidence}")
    else:
        print(f"\n[HUNT/trainaudit] CLEAN: no rule violations "
              f"(optim={optim_name} decay={decay_name})")
    trainaudit.disable()


if __name__ == "__main__":
    main()
