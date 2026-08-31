"""OC-NEW-2 via TrainAudit T0: optimizer state['step'] must increment.

Bug: SkipStepAdamW had `step.add_(step_factor)` commented out, so step counter
never increments. T0-optim-step-counter-monotonic rule fires when
state[p]['step'] doesn't increase between optim.step.post events.
"""
import os
import sys
import torch
import torch.nn as nn

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "trainaudit_pkg"))
OLMO_CORE_DIR = os.environ.get("OLMO_CORE_DIR", "")
if OLMO_CORE_DIR:
    sys.path.insert(0, os.path.join(OLMO_CORE_DIR, "src"))

import trainaudit
trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path=":memory:")

from olmo_core.optim.adamw import SkipStepAdamWConfig

torch.manual_seed(42)
model = nn.Linear(64, 64)
optimizer_config = SkipStepAdamWConfig(lr=1e-3, betas=(0.9, 0.999))
optimizer = optimizer_config.build(model)

trainaudit.snapshot_build(model, optimizer)

for step in range(5):
    trainaudit.set_step(step)
    x = torch.randn(4, 64)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

results = trainaudit.run_rules()
print(trainaudit.summarize(results))
violated = [r for r in results if r.violated]
if violated:
    print(f"\n[OC-NEW-2/trainaudit] BUG DETECTED via {len(violated)} rule(s):")
    for v in violated:
        print(f"   - {v.rule_id}: {v.message}")
        if v.evidence:
            print(f"     evidence: {v.evidence}")
else:
    print("\n[OC-NEW-2/trainaudit] CLEAN: no rule violations")
trainaudit.disable()
