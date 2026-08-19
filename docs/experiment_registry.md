# Experiment Registry

Every number that appears in the paper should be registered here before it is used in the abstract, introduction, or result tables.

Status labels:

- `CONFIRMED`: source files and command are known.
- `NEEDS CHECK`: number appears in draft but source/definition needs confirmation.
- `CONFLICT`: conflicting values exist in the draft.
- `TODO`: experiment still needs to be run.

## Headline Results

| Metric | Current Value | Status | Source / File | Owner Notes |
|---|---:|---|---|---|
| Real-SDC buggy detection | 17/19 (89.5%) | NEEDS CHECK | `main_cn.tex`, detection tables | Confirm denominator includes 17 reproduced bugs + 2 boundary cases. |
| TrainCheck detection | 5/19 (26.3%) | NEEDS CHECK | `benchmark/eval/baseline_traincheck_results.csv` or table | Confirm same harness and exact TrainCheck setup. |
| Naive monitoring detection | 0/19 (0.0%) | NEEDS CHECK | `benchmark/eval/baseline_naive_results.csv` or table | Confirm thresholds and metrics. |
| Fixed replay FP | 0/17 vs 1/17 | CONFLICT | `main_cn.tex` abstract vs table | Must resolve before any rewrite. |
| H200 real GPU buggy detection | 13/14 (92.9%) | NEEDS CHECK | `benchmark/eval/results_gpu.csv` | Decide whether this is main result or appendix. |
| H200 fixed FP | 0/12 | NEEDS CHECK | `benchmark/eval/results_gpu.csv` | Define relation to 0/17 or 1/17 result. |
| New Megatron-LM bug | issue #4641 / PR #4642 | NEEDS CHECK | case-study figure and upstream issue | Need citation or anonymous-safe wording for submission. |

## Required SIGMOD Overhead Experiments

| Experiment | Status | Required Output | Suggested Files |
|---|---|---|---|
| Baseline training step time | TODO | mean/median/p95 step time, tokens/sec | `benchmark/eval/overhead.py`, new CSV |
| S0 overhead | TODO | step overhead %, trace MB/step, query latency | new CSV |
| S1-S2 overhead | TODO | step overhead %, trace MB/step, query latency | new CSV |
| S3-S4 overhead | TODO | step overhead %, trace MB/step, query latency | new CSV |
| S5-S6 overhead | TODO | step overhead %, trace MB/step, query latency | new CSV |
| DuckDB query scalability | TODO | latency vs rows/ranks/params | new CSV |
| Trace write throughput | TODO | MB/s, CPU overhead | new CSV |

## Verification / Mining Experiments

| Experiment | Current Status | Required Output | Notes |
|---|---|---|---|
| Full funnel candidate counts | NEEDS CHECK | L1/L2/L3/L4 counts and accepted rules | Tie to `fig_funnel_ablation`. |
| Skip adversarial verification | NEEDS CHECK | FP rate and candidate counts | Must define what is skipped. |
| Skip healthy replay | NEEDS CHECK | FP rate and candidate counts | Main evidence for verification necessity. |
| Skip topology guard | NEEDS CHECK | clean-run FP/1M | Tie to three-predicate ablation. |
| Temporal holdout | TODO | detection and FP on unseen time split | Needed for overfitting concern. |
| Framework holdout | TODO | detection and FP on unseen framework | If too hard, document partial holdout. |

## Existing Candidate Source Files

Small files that appear useful for shared evaluation:

- `benchmark/eval/results.csv`
- `benchmark/eval/results_gpu.csv`
- `benchmark/eval/baseline_traincheck_results.csv`
- `benchmark/eval/baseline_naive_results.csv`
- `benchmark/eval/fp_audit.csv`
- `benchmark/eval/funnel_counts.csv`
- `benchmark/eval/funnel_skip_l3_results.csv`
- `benchmark/eval/funnel_skip_l4_results.csv`
- `benchmark/eval/overhead.csv`
- `benchmark/eval/silent_evidence_392.json`
- `benchmark/eval/manifest_v2.json`
- `benchmark/eval/manifest_summary.md`

## Rules

- Do not edit a paper table directly unless this file already contains the corresponding value.
- Record the command, input files, and output files for every new experiment.
- If a number changes, update this file and then update the paper.
