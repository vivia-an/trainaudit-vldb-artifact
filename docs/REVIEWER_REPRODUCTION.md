# Reviewer reproduction guide

## Minimal path

```bash
git clone https://github.com/vivia-an/trainaudit-vldb-artifact.git
cd trainaudit-vldb-artifact
bash scripts/install_release_env.sh
bash scripts/check_release.sh
```

The release check covers the runnable smoke example, Real-SE outcome records,
Catalog ablation, catalog generalization, corpus construction, annotation
statistics, collector timing logs, release manifests, citations, and PDF/source
synchronization.

## Focused checks

```bash
# Core pipeline
python core/run_smoke.py

# Real-SE records
python benchmark/eval/real_sdc/extract_detection_csv.py
python benchmark/eval/real_sdc/extract_replay_outcomes.py

# Catalog evaluation
python benchmark/eval/paper_v2/verify_catalog_direct_ablation.py
python benchmark/eval/verify_catalog_generalization.py --check

# System timing
python benchmark/injection/parse_overhead_logs.py --check
```

## Trace-backed checks

```bash
bash scripts/fetch_trace_dbs.sh --dest traces
TAG=trace-events-v2 bash scripts/fetch_trace_dbs.sh --dest event-traces
python core/validate_generated_sql.py --db <trace>
```

See [`RERUN_LIMITS.md`](RERUN_LIMITS.md) for the distinction between offline
record verification and full GPU/framework reruns.
