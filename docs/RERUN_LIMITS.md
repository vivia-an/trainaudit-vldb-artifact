# Reproduction scope

TrainAudit separates lightweight result verification from resource-intensive
experiment reruns.

## Offline, no model or API key

The following paths run from a normal clone:

```bash
bash scripts/check_release.sh
python core/run_smoke.py
python benchmark/eval/paper_v2/verify_catalog_direct_ablation.py
python benchmark/eval/verify_catalog_generalization.py --check
python benchmark/injection/parse_overhead_logs.py --check
```

These commands validate released records, recompute reported aggregates, and
exercise the verifier on a toy trace.

## Requires downloaded trace bundles

Executing compiled SQL against recorded DuckDB traces and repeating trace-level
aggregations requires the release assets fetched by `scripts/fetch_trace_dbs.sh`.

## Requires an LLM provider

The mining pipeline and generation of SQL for a previously unseen rule require a
configured provider. The SQL recovered from recorded experiments is stored in
`core/config/generated_sql.json` and can be executed without an API key.

## Requires GPUs and framework checkouts

Real-SE training replays and collector-performance measurements require the
framework revisions, model setup, and distributed GPU environment associated
with each run. The manifest records the relevant upstream revisions and expected
outcomes; the repository supplies launchers, instrumentation, and comparison
records.

Exact wall-clock values can vary with hardware and software versions. Result
verification therefore checks the released records exactly, while a fresh system
rerun should reproduce the reported direction and operating range under a matched
environment.
