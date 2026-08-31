# TrainAudit core

The core release has three explicit components:

| Component | Path | Role |
|---|---|---|
| Online runtime | `trainaudit_pkg/` | trace collection, versioned schema, topology-aware rules, and the public Python API |
| Offline miner | `agents/`, `run_miner.py` | catalog-guided candidate construction and verification before deployment |
| Recorded-run replay | `sdccheck/` | compatibility utility for executing constraint specifications against archived DuckDB traces |

The online runtime does not call an LLM. Mining is an offline step; accepted or
recorded constraints are evaluated deterministically by the runtime/replay path.

## Install the runtime package

```bash
python -m pip install -e core/trainaudit_pkg
```

## Offline smoke test

From the repository root:

```bash
python core/run_smoke.py
```

The smoke test creates clean and faulty toy traces, checks catalog integrity,
exercises the mining state machine, verifies the acceptance gate, and runs the
topology pruning example. It requires no model, GPU, or API key.

## Use the runtime API

```python
import trainaudit

trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path="trace.duckdb")
trainaudit.snapshot_build(model, optimizer)

# training steps

violations = trainaudit.run_rules()
trainaudit.disable()
```

Framework adapters may add higher-tier attributes while preserving the same
trace-store and rule interface.

## Run the offline miner

The miner requires the optional mining dependencies and a configured provider:

```bash
python -m pip install -r core/requirements-mining.txt
export SDC_PAPER_ALIGN=1
export DEEPSEEK_API_KEY=...
python core/run_miner.py
```

`SDC_PAPER_ALIGN=1` enables the frozen Pattern Catalog and verified acceptance
path. `SDC_HEALTHY_DBS` may point to colon-separated healthy DuckDB traces for
candidate validation.

## Frozen records

- `config/frozen_template_catalog.json` — catalog snapshot;
- `config/frozen_template_catalog.sha256` — integrity digest;
- `config/generated_sql.json` — SQL recovered from recorded runs;
- `data/funnel_counts.csv` — released funnel aggregation inputs.

The repository-level release checks are run by `scripts/check_release.sh`.
