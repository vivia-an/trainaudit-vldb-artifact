# Paper §4.1 — TrainAudit vs TrainCheck baseline

## Harness status

- TrainCheck import: ✓
- TrainCheck location: `/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025/exp/traincheck/TrainCheck/traincheck/__init__.py`
- Submodules probed: `instrumentor, checker, infer_engine, collect_trace, invariant`

## Per-bug verdict comparison

| bug_id | TrainAudit | TrainCheck | note |
|---|---|---|---|
| B11 | DETECTED | PENDING_INTEGRATION | TrainCheck imports OK; full instrumentation integration is the follow-up — its A |
| B12 | DETECTED | PENDING_INTEGRATION | TrainCheck imports OK; full instrumentation integration is the follow-up — its A |
| O-NEW-1 | DETECTED | PENDING_INTEGRATION | TrainCheck imports OK; full instrumentation integration is the follow-up — its A |
| OC-NEW-2 | DETECTED | PENDING_INTEGRATION | TrainCheck imports OK; full instrumentation integration is the follow-up — its A |
| M-014 | DETECTED | PENDING_INTEGRATION | TrainCheck imports OK; full instrumentation integration is the follow-up — its A |

## Status

TrainCheck is installed (polars + orjson + astor + dill) and its public API (`Instrumentor`, `traincheck.{instrumentor,checker,infer_engine,collect_trace,invariant}` modules) imports cleanly. Same-machine programmatic comparison via `synthetic_runners` is the follow-up task: route the surrogate's training loop through `Instrumentor.start()` and write to a file-backed trace, then run `traincheck.checker` and parse its verdicts into the same DETECTED/CLEAN/FAIL contract. Until that ships, this table reports the import status and TrainAudit's verdicts for the diff baseline.

Per CLAUDE.md, if same-machine integration cannot stabilise we fall back to a related-work qualitative comparison referencing the published TrainCheck paper numbers — but DO NOT splice their paper numbers into a same-machine table.
