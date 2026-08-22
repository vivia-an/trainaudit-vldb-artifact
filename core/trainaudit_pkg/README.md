# TrainAudit

> Tiered trace + invariant checking for distributed training silent-error detection.

## Status: Phase 1 (T0 PyTorch core)

This is the minimum-viable trace mechanism — pure PyTorch hooks, zero framework-specific code. See `docs/v2_semantic_guided/21_落地roadmap.md` for the 5-week plan.

## What works in Phase 1

- 5 PyTorch hookpoints: `torch.distributed.*`, `nn.Module` fwd/bwd, `Optimizer.step`, `clip_grad_norm_`
- Build snapshot of model + optimizer
- DuckDB trace store with versioned schema (OTel-style attribute namespace)
- 7 T0 invariant rules covering ~13 of the 48 reproduced bugs
- Smoke test: enables on toy training, verifies trace + rules pass clean

## Quick start

```python
import trainaudit
import torch.nn as nn, torch.optim as optim

trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path="./trace.duckdb")

model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 4))
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

trainaudit.snapshot_build(model, optimizer)

# ... your training loop ...

violations = trainaudit.run_rules()
for v in violations:
    if v.violated:
        print(f"VIOLATION {v.rule_id}: {v.message}")
trainaudit.disable()
```

## Layout

```
trainaudit/
├── pyproject.toml
├── README.md
├── trainaudit/
│   ├── __init__.py            # public API: enable / snapshot_build / run_rules / disable
│   ├── tiers.py               # Tier enum (T0..T4)
│   ├── schema.py              # OTel-style attribute key constants
│   ├── store.py               # DuckDB trace store
│   ├── verifier.py            # rule runner
│   ├── core_trace/
│   │   ├── _utils.py          # tensor summarization
│   │   ├── dist_hook.py       # torch.distributed wrappers
│   │   ├── module_hook.py     # nn.Module global hooks
│   │   ├── optim_hook.py      # Optimizer.step + clip_grad_norm_ wrappers
│   │   └── build_snapshot.py  # one-shot model/optimizer scan
│   └── rules/
│       ├── base.py            # @rule decorator + RuleResult
│       └── T0_*.py            # 7 T0 rules
└── tests/
    └── test_smoke.py
```
