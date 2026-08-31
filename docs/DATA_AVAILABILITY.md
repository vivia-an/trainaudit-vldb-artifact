# Data availability

## Included in Git

The repository includes the source code, manifests, compact evaluation records,
annotation files, aggregation scripts, paper sources, and raw collector timing
logs needed by the offline release checks.

The canonical evaluation paths are:

- `benchmark/eval/real_sdc/` — Real-SE manifest and per-case outcomes;
- `benchmark/eval/template_induction/` — catalog induction protocol and records;
- `benchmark/eval/catalog_generalization/` — held-out catalog evaluation;
- `benchmark/eval/paper_v2/` — matched ablation and compact system summaries;
- `benchmark/injection/overhead_raw/` — collector timing logs;
- `experiments/guard_ablation/` — recorded guard-ablation cells.

## Trace release assets

Large DuckDB trace files are published as checksummed GitHub release assets:

```bash
bash scripts/fetch_trace_dbs.sh --dest traces
TAG=trace-events-v2 bash scripts/fetch_trace_dbs.sh --dest event-traces
```

The first bundle contains the coredump-schema traces used by the recorded guard
experiments. The second contains event-schema traces used by the current collector
and topology-aware examples. `fetch_trace_dbs.sh` verifies the downloaded files
against the committed manifests.

## External dependencies

Full framework replays require the upstream framework repositories at the buggy
and fixed commits named by `benchmark/eval/real_sdc/real_sdc_manifest.json`.
TrainCheck is referenced by upstream revision rather than vendored. Model weights
and third-party framework repositories retain their original licenses and are not
duplicated here.

## Compact external-cluster records

The CSVs in `benchmark/eval/paper_v2/` are canonical compact records for the
matched Catalog ablation, end-to-end overhead, schema coverage, and transfer
endpoint. Their validation scripts check arithmetic, pairing, and internal
consistency. They are intended for result auditing; GPU/framework reruns require
the environments described above.
