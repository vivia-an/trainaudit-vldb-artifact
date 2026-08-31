"""B12 via TrainAudit Phase 1 (T0).

Trigger same as detect.py: build a real model + AdamWConfig optimizer on GPU,
then construct an LRScheduler with last_epoch=0 (resume mode). The scheduler
will fail with KeyError on buggy commit; trainaudit's T0-initial-lr-present
rule will catch it without any framework-specific hook.
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "trainaudit_pkg"))
import trainaudit


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 1) enable trainaudit BEFORE building optimizer/scheduler
    trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path=":memory:")

    from olmo_core.optim import AdamWConfig

    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(1024, 128), nn.Linear(128, 1024)).to(device)
    config = AdamWConfig(lr=3e-4)
    optim = config.build(model)

    # 2) snapshot model + optimizer at build time
    trainaudit.snapshot_build(model, optim)

    # 3) real backward + step (end-to-end requirement)
    input_ids = torch.randint(0, 1024, (2, 16), device=device)
    for step in range(3):
        trainaudit.set_step(step)
        optim.zero_grad()
        loss = model(input_ids).float().sum()
        loss.backward()
        optim.step()

    # 4) trigger the resume-mode scheduler construction (this is what fails on buggy)
    sched_err = None
    try:
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda e: 1.0, last_epoch=0,
        )
        _ = sched.get_last_lr()
    except Exception as e:  # noqa: BLE001
        sched_err = f"{type(e).__name__}: {e}"

    if dist.get_rank() == 0:
        print(f"[B12] scheduler construction error = {sched_err}")
        results = trainaudit.run_rules()
        print(trainaudit.summarize(results))
        violated = [r for r in results if r.violated]
        if violated:
            print(f"\n[B12/trainaudit] BUG DETECTED via {len(violated)} rule(s):")
            for v in violated:
                print(f"   - {v.rule_id}: {v.message}")
        else:
            print("\n[B12/trainaudit] CLEAN: no rule violations")

    dist.barrier()
    dist.destroy_process_group()
    trainaudit.disable()


if __name__ == "__main__":
    main()
