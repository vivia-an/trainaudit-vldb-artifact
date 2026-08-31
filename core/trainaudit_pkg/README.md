# TrainAudit runtime

`trainaudit` is the online trace and rule-execution package used by the public
artifact. It provides:

- versioned DuckDB trace storage;
- PyTorch lifecycle and distributed hookpoints;
- model/optimizer build snapshots;
- topology-aware rule registration and execution;
- framework adapters built on the same event schema.

## Install

```bash
python -m pip install -e core/trainaudit_pkg
```

## Minimal example

```python
import trainaudit

trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path="trace.duckdb")
trainaudit.snapshot_build(model, optimizer)

# run the training step(s) to be audited

for violation in trainaudit.run_rules():
    if violation.violated:
        print(violation.rule_id, violation.message)

trainaudit.disable()
```

## Package layout

```text
trainaudit/
  core_trace/       PyTorch and distributed hookpoints
  adapters/         framework-specific attribute adapters
  rules/            executable rule implementations
  dsl/              declarative rule registry
  store.py          DuckDB event store
  schema.py         versioned attribute namespace
  verifier.py       deterministic rule runner
  tiers.py          integration tiers
```

The package test suite is included under `tests/` and is executed by the
repository-level `scripts/check_release.sh`.
