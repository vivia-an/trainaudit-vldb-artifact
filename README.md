# TrainAudit

[![release checks](https://github.com/vivia-an/trainaudit-vldb-artifact/actions/workflows/release-checks.yml/badge.svg)](https://github.com/vivia-an/trainaudit-vldb-artifact/actions/workflows/release-checks.yml)

Artifact for **“TrainAudit: Silent Error Detection for LLM Training via Verified
Guarded Relational Constraints.”**

TrainAudit represents distributed-training state as relational traces and checks
topology- and phase-guarded constraints over those traces. The repository contains
the runnable verifier, trace collector, rule-mining pipeline, Real-SE evaluation
records, and the scripts used to verify the principal reported results.

## Repository map

| Path | Contents |
|---|---|
| `core/` | constraint catalog, mining pipeline, SQL verifier, and offline smoke examples |
| `collector/vtimeline/` | training instrumentation and GPU-side fingerprint collection |
| `benchmark/eval/real_sdc/` | Real-SE manifest, per-case outcomes, and replay records |
| `benchmark/eval/template_induction/` | frozen-catalog induction and held-out evaluation |
| `benchmark/eval/catalog_generalization/` | catalog generalization records and checks |
| `benchmark/eval/paper_v2/` | canonical compact records used by the matched Catalog ablation and schema study |
| `benchmark/injection/` | fault-injection launchers and raw collector-overhead logs |
| `experiments/` | recorded guard-ablation cells and aggregation inputs |
| `baselines/` | baseline adapters, run records, and upstream pointers |
| `paper/` | synchronized manuscript sources and submitted PDFs |
| `scripts/` | release validation, data fetch, and integrity utilities |

The compact result-to-command index is in
[`docs/CLAIM_TO_ARTIFACT_MAP.md`](docs/CLAIM_TO_ARTIFACT_MAP.md).
The immutable submission snapshot is published under
[`vldb-2027-artifact-v1.0.1`](https://github.com/vivia-an/trainaudit-vldb-artifact/releases/tag/vldb-2027-artifact-v1.0.1).

## Quick start

The offline validation path needs Python 3.10+. The installer provisions the
pinned CPU-compatible dependencies used by continuous integration:

```bash
bash scripts/install_release_env.sh
bash scripts/check_release.sh
python core/run_smoke.py
```

The smoke run creates a toy trace, applies guarded checks, and requires no GPU,
model checkpoint, API credential, or downloaded trace bundle.

## Reproduce the principal results

### Real-SE outcomes

```bash
python benchmark/eval/real_sdc/extract_detection_csv.py
python benchmark/eval/real_sdc/extract_replay_outcomes.py
python benchmark/eval/verify_appendix_detection_table.py --check
```

These commands verify the per-case TrainAudit outcomes and the paired fixed-side
records used by the paper.

### Matched Catalog ablation

```bash
python benchmark/eval/paper_v2/verify_catalog_direct_ablation.py
```

This checks the matched Catalog/free-form funnel, endpoint detection, paired
discordances, and exact McNemar calculation from the released aggregate and
per-case endpoint records.

### Frozen-catalog generalization

```bash
python benchmark/eval/verify_catalog_generalization.py --check
```

This rebuilds the reported held-out coverage summaries from the released
per-record data.

### Corpus and annotation checks

```bash
python benchmark/eval/verify_corpus_construction.py --check
python benchmark/eval/verify_irr.py --check
python benchmark/eval/verify_taxonomy_table.py --check
```

### Collector microbenchmark

```bash
python benchmark/injection/parse_overhead_logs.py --check
```

This recomputes the full-snapshot timings, including the measured
191.9 s to 25.0 s collector reduction, from the raw H20 logs.

### Schema coverage

```bash
python benchmark/eval/verify_tier_coverage_axis.py --check
```

This rebuilds the cumulative coverage axis from the released per-record tier
assignments.

## Trace bundles

Large DuckDB traces are distributed as checksummed GitHub release assets rather
than ordinary Git objects:

```bash
bash scripts/fetch_trace_dbs.sh --dest traces
TAG=trace-events-v2 bash scripts/fetch_trace_dbs.sh --dest event-traces
```

The fetcher verifies each archive against the committed manifests. Details are
in [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

## Full reruns

Offline checks consume only released records. Re-running the mining stage needs
an LLM provider, and replaying framework bugs or collector measurements needs
the framework revisions and GPU environment recorded by the relevant manifests.
See [`docs/RERUN_LIMITS.md`](docs/RERUN_LIMITS.md) for the exact boundary.

## Paper build

Prebuilt PDFs are included. To rebuild the manuscript:

```bash
cd paper
pdflatex -shell-escape main.tex
bibtex main
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
```

The build requires the ACM/PVLDB LaTeX dependencies, `minted`, and scalable font
packages. The empirical checks do not require a LaTeX installation.

## Citation

Citation metadata for the software artifact is provided in
[`CITATION.cff`](CITATION.cff). The repository is released under the MIT
License; see [`LICENSE`](LICENSE).
