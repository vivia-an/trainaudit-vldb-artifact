# Data availability

## In this repository (~66 MB, ~1.1k files)

| Path | Contents |
|---|---|
| `paper/` | `main.tex`, `appendix.tex`, `numbers.tex`, `main.bib`, PVLDB style files, built `main.pdf` (14 pp.) and `appendix_supplement.pdf` |
| `core/` | verified-mining pipeline: Pattern Catalog (T01–T35, frozen + sha256), multi-agent FSM, Accept gate, healthy-run replay, SQL verifier, ablation library builders |
| `collector/vtimeline/` | the live training instrumentation actually imported by the training runs (`from vtimeline import MegatronCollector`); `collector/megatron_collector_legacy_sha256.py` is the pre-optimisation copy kept for the 192 s → 25 s comparison |
| `benchmark/eval/` | complete small evaluation tree, at the paths the paper cites (`real_sdc/`, `template_induction/`, `catalog_generalization/`, `traincheck_surrogates/`, `gpu_logs/`, `paper_v2/`, `v2_full/`, …) |
| `benchmark/injection/` | 51 fault-injection and overhead launch scripts, 14 raw H20 collector logs, and an index of the on-disk trace DBs |
| `experiments/` | leave-one-out guard ablation (126 cells) and holdout mining transcripts (gzipped) |
| `figures/` | the 15 figure files the paper includes, plus the generator scripts that exist |
| `docs/` | this audit set, plus the design/runbook notes carried over from the paper repo |

## Deliberately not in this repository

| Data | Size | Where it lives | Why excluded |
|---|---|---|---|
| Trace DuckDB databases: 42 fault-injected + clean runs, plus overhead runs | ~600 MB total, 3–33 MB each | `Megatron-LM/*_test_db`, `normal_db/`, `dp_normal_db/`, `overhead_*_db/`; indexed in `benchmark/injection/trace_db_index.csv` | too large for a git repo; **should be published as a release asset or archival record** — without them the verifier cannot be re-run end to end |
| `rebuttal_v1/` replay outputs | 9.2 GB | `sdc_llm_icml_2025/benchmark/eval/rebuttal_v1/`; indexed in `benchmark/heavy_outputs_index.csv` | prior review round, not cited by this paper |
| `hunt_log/` mining transcripts | 550 MB | `sdc_llm_icml_2025/benchmark/eval/hunt_log/`; indexed | superseded by `experiments/holdout_mining/` |
| TrainCheck baseline implementation | 452 MB | upstream project, checked out locally; see `baselines/TRAINCHECK_UPSTREAM.txt` | third-party code, cited not vendored |
| Framework checkouts (Megatron-LM, DeepSpeed, OLMo) | ~9 GB | upstream repos at the per-case buggy/fixed revisions recorded in `benchmark/eval/real_sdc/real_sdc_manifest.json` | reconstructible from the manifest |

## Reconstructing a full replay

1. Check out the framework at the buggy/fixed revisions named in
   `benchmark/eval/real_sdc/real_sdc_manifest.json`.
2. Put `collector/vtimeline/src` on `PYTHONPATH` and launch with the matching
   script from `benchmark/injection/launch_scripts/`; `VTIMELINE_FAST=1` selects
   the optimised (GPU-fingerprint) collector path.
3. Run the verifier from `core/` against the produced DuckDB trace with the
   frozen rule library in `core/config/`.
4. Compare against the recorded verdict in `benchmark/eval/results.csv`.
