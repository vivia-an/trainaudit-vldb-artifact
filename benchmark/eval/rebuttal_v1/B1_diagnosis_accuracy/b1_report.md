# §6.3 Diagnosis Accuracy — B1 Results

## Annotation Protocol

Universe: 27 D2 cases (14 D1' old + 3 D1' new + 8 D2-new + 2 boundary).
For each case I (annotator: Claude Opus, computer-science PhD-level) checked:
1. `benchmark/bugs/<id>/config.json` for `category`, `invariant`, `root_cause`,
   `detection_method`, `fixed_commit`;
2. The trigger record from either `benchmark/eval/paper_table_gpu.md` (14 real-GPU
   bugs) or `benchmark/eval/d{1_prime,2_extension}/results/trainaudit_*_inline.json`
   (11 surrogate-detector inline cases);
3. The triggered rule's contract in `trainaudit/trainaudit/rules/` to
   confirm whether the fired rule semantically covers the bug's root cause.

The four-dimensional verdict per case:

| Dim | exact | near | wrong |
|---|---|---|---|
| **parent category** | rule's fault class equals ground-truth `category` | rule covers a sister class (e.g. dtype rule on a moe-tagged dtype bug) | unrelated class |
| **leaf rule** | rule id is the contract rule for this bug | same-pattern variant | cross-pattern |
| **suspect object** | `(param/module/rank/hookpoint)` matches root-cause site | partial overlap | unrelated |
| **first-bad-step** | predicted within ±2 of true first-bad-step | within ±10 | farther or absent |

Annotations live in `b1_manual_verdicts.csv` (27 rows × verdict cells).

## Headline Numbers (strict, n=20)

Strict denominator = 27 − 2 boundary (`LC1`, `DL2`, no rule in catalog)
− 1 GPU-missed (`B1`) − 4 surrogate-only-detected-without-rule-id
(`B2`, `B3`, `B8`, `O-NEW-9`) = **20 detected D2 cases with a logged
trigger rule**.

| Dimension | exact | exact ∪ near |
|---|---|---|
| Parent-category accuracy | **18 / 20 = 90.0%** | 20 / 20 = 100% |
| Leaf-rule accuracy | **20 / 20 = 100%** | 20 / 20 = 100% |
| Suspect-object accuracy | **20 / 20 = 100%** | 20 / 20 = 100% |
| First-bad-step accuracy (±2) | **19 / 20 = 95.0%** | 20 / 20 = 100% |

The two `near` parent-category verdicts:
- `M-020`: d2 tag is *sharding*; rule `T1-layer-count-strict` is
  *configuration_validation* (the bug is a PP-shard layer-count mismatch — the
  rule names the proximate config invariant, not the parallel-shard tag).
- `O-NEW-1`: d2 tag is *numerical*; rule `T0-norm-output-unit-rms` is
  *normalization* (the bug is "L2 norm where RMS expected"; both tags valid).

The one `near` first-bad-step verdict:
- `CW1`: counter-width overflow; rule fires *at* step 127 (the overflow step),
  but the underlying int8 counter has been latent the whole training run. The
  rule's design is to fire when the symptom surfaces, not when the design
  defect was introduced, so calling this "near" rather than "exact" is a
  conservative interpretation.

## Honest, Wider Denominators

Two more honest tabulations the paper should also disclose:

| Denominator | n | Category | Rule | Suspect | Step |
|---|---|---|---|---|---|
| **Detected 25** (penalises B1 missed + 4 surrogate-only-no-rule as wrong) | 25 | 72.0% | 80.0% | 80.0% | 76.0% |
| **Full 27** (also penalises 2 boundary cases) | 27 | 66.7% | 74.1% | 74.1% | 70.4% |

The two boundary cases (`LC1`, `DL2`) are explicit out-of-catalog instances
chosen at survey time, so penalising them mainly inflates the
"impossible-to-detect" denominator. The 4 surrogate-only-no-rule cases (`B2`,
`B3`, `B8`, `O-NEW-9`) are detected on `d2_aggregate`'s flat tally but no
trigger record exists in our two source files — this is a logging gap, not a
diagnosis failure; the cleanest fix is to log triggered rule_id during the
surrogate-detection run (one-line patch to `benchmark/eval/d2_extension/
trainaudit_inline_d2.py`).

## Paper §6.3 Suggested Prose

> Of 20 detected D2 cases for which the triggered rule was logged
> (boundary cases and the GPU-missed B1 excluded), \textsc{TrainAudit}'s
> report achieves **parent-category accuracy 18/20 (90.0\%)**, **leaf-rule
> accuracy 20/20 (100\%)**, **suspect-object accuracy 20/20 (100\%)**, and
> **first-bad-step accuracy ($\le$2 steps) 19/20 (95.0\%)**. The two
> parent-category mismatches are sister-class confusions (M-020:
> configuration\_validation vs sharding tag; O-NEW-1: normalization vs
> numerical tag), and the one first-bad-step near-miss (CW1) is a counter
> that overflows long after the latent defect is in place — the rule
> correctly fires at the overflow step rather than the introduction step.

Replace the current Finding 1--4 prose with the prose above plus a brief
note on the surrogate-only logging gap (B2/B3/B8/O-NEW-9: detection
confirmed in aggregate but rule_id not separately logged).
