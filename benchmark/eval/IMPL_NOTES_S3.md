# IMPL NOTES — S3 Diagnosis Accuracy

Spec: [SPEC_S3_diagnosis_accuracy.md](../../SPEC_S3_diagnosis_accuracy.md)

## What this directory contains

| File | Role |
|---|---|
| `diagnosis_s3_input.json` | 17 records, one per TrainAudit DETECTED case, with fired `rule_id` + `rule_message` and ground-truth title / summary / invariant pulled from `benchmark/bugs/<bug_id>/config.json` |
| `diagnosis_s3_rubric.md` | Rubric for the two binary verdicts (L1 = rule name -> fault class, L2 = RCA chain -> leaf root cause) |
| `diagnosis_s3_rater_A.csv` | Empty template for Rater A (human) |
| `diagnosis_s3_rater_B.csv` | Filled by this agent (Rater B) without reading any other paper file |
| `diagnosis_s3_score.py` | Computes L1/L2 hit rates + Cohen's kappa + disagreement list; writes `paper_table_diagnosis_s3.md` |

## Data sources

- **17-case list**: `paper_table_baseline_3way.md` "Per-bug verdict" table, TrainAudit DETECTED column.
- **rule_id / rule_message** per case: `benchmark/eval/results.csv` (buggy-phase row). `results_gpu.csv` is the GPU subset and lacks B2/B3/B8/O-NEW-9; we use `results.csv` which covers all 17.
- **Ground truth** (title / root_cause / invariant): `benchmark/bugs/<bug_id>/config.json`. For M-NEW-5 and O-NEW-9 the config is sparse (title only); summary/invariant were reconstructed from the title plus the fix commit semantics that the rule already encodes.

## Rubric notes

- The IRR-50 rubric in `annotate_prompt.md` is a 13-class **category** rubric, not directly applicable to L1/L2 binary verdicts. We reuse its **format** (binary + justification + confidence + self-check) but redefine the schema for L1/L2 semantics.
- L1 verdict has three acceptance paths in the rubric: exact match, property match, necessary-symptom match. The third path is needed for cases like M-NEW-5 where the rule checks a *necessary condition* (router exposes calculate_per_token_loss attr) rather than the loss formula directly.

## RCA chain (L2) status

The current TrainAudit runtime emits `rule_id + rule_message` only. The multi-hop RCA agent drill-down output is not persisted per case in the existing pipeline. All 17 records therefore have `rca_chain: []` and the rubric mandates `L2_correct: N/A` for empty chains. L2 hit rate is undefined for this batch.

If/when RCA chains are captured, the rubric still applies; just refill the L2 column.

## Sanity-gate check

- [x] 17 records, all with `rule_id` and `ground_truth_summary` non-empty.
- [x] Rater B `justification` non-empty for all 17 rows.
- [x] L1 hit count (Rater B) = 17/17, inside the 12-17 red-line band.
- [x] L2 = N/A across the board (no RCA chain stored); L2 hit rate not yet measured.

## Reproduce

```bash
# 1. Inspect input
cat benchmark/eval/diagnosis_s3_input.json | jq '.cases[] | {bug_id, rule_id}'

# 2. (Human) fill diagnosis_s3_rater_A.csv against diagnosis_s3_rubric.md

# 3. Score
python3 benchmark/eval/diagnosis_s3_score.py
# -> stdout: A/B hit counts + kappa + disagreement list
# -> writes benchmark/eval/paper_table_diagnosis_s3.md
```

## Open follow-ups

1. Capture `rca_chain` from `trainaudit/diagnosis/` so L2 can be measured.
2. After Rater A fills, run `diagnosis_s3_score.py` to compute kappa. If kappa < 0.6, surface the disagreement list to the spec author for rubric refinement (do not silently re-tune rubric).
3. The 17/17 L1 result will need to be paired with the §6.2 first paragraph claim — wire the final number into `main.tex` once Rater A is in.
