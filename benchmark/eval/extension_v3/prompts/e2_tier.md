# E2 Prompt: Trace Schema Tier Coverage Annotation

> Output: `trace_tier_392.json` per-bug entry containing `min_tier_required` + `fields_required`.
> Reference: `28_392_extension_brief.md` §3.2, `main_cn.tex` :535 (Tier table).

## TASK

For the given bug, determine the **minimum trace schema tier required** to observe (detect) the bug at runtime. If the bug is unobservable at any tier, output `unobservable`.

## TIER DEFINITIONS (cumulative, additive)

| Tier | Newly added fields | 128-pool cumulative coverage |
|---|---|---|
| **0** | parameter checksum, param_name, rank coords (dp/tp/pp/ep), step, stage, dtype, shape | 30% |
| **1** | + grad_norm, grad_cksum (gradient stats) | 41% |
| **2** | + loss_value | 51% |
| **3** | + learning_rate, micro_step_id (control variables) | 59% |
| **4** | + optimizer_state_cksum (Adam/momentum state) | 61% |
| **5-6** | + ep/cp_rank, zero_stage, has_nan/inf, num_tokens (full 27-field schema) | 74% |
| **unobservable** | runtime hooks see no distinguishing signal — bug requires source-code inspection or offline ground-truth comparison | 26% gap |

## DECISION RULES

1. Identify the **detection field set** — the minimum trace fields required to distinguish buggy from clean runs.
2. Look up **the highest-tier field** in your set; that's the `min_tier_required`.
3. **Take the lowest tier**: if the bug can be detected by either dtype (Tier 0) OR optimizer_state_cksum (Tier 4), record Tier 0 (the cheaper one).
4. **Unobservable** if: (a) detection_method is `source_analysis`, OR (b) trace would record bit-identical values for buggy vs clean (e.g., function call wrong but produces same intermediate values), OR (c) requires post-training model-quality regression.

### Common field-to-tier mapping

| Field needed for detection | Tier |
|---|---|
| param checksum, dtype, shape, name | 0 |
| rank-id (DP/TP/PP) | 0 |
| step counter, stage | 0 |
| grad_norm, grad checksum | 1 |
| loss_value | 2 |
| learning_rate | 3 |
| micro_step_id, accumulation_count | 3 |
| optimizer state (Adam m/v cksum, momentum) | 4 |
| ep_rank, cp_rank | 5-6 |
| zero_stage flag | 5-6 |
| has_nan / has_inf | 5-6 |
| num_tokens | 5-6 |
| function-call count / order (Type-B hook) | 5-6 (treat as "module attribute") |
| module class / named_modules snapshot | 5-6 |
| op-level intermediate value (forward activation) | unobservable (op-level not in schema) |
| dataloader internal state (shuffle index, seed) | unobservable |
| source code structure | unobservable |

### Few-shot examples

**Example 1** (Tier 0):
- bug: SwitchMLP router weight diverges across TP ranks (B1)
- detection: cross-rank checksum compare on `router.weight`
- fields_required: ["param_cksum", "param_name", "tp_rank", "step"]
- min_tier_required: 0
- rationale: All needed fields are param checksum + rank coords; pure Tier 0.

**Example 2** (Tier 1):
- bug: gradient clipping wrong threshold causes scale anomaly
- detection: grad_norm exceeds expected bound
- fields_required: ["grad_norm", "param_name", "step"]
- min_tier_required: 1

**Example 3** (Tier 2):
- bug: loss reduction mode wrong (`mean` vs `sum`)
- detection: loss value scale mismatch
- fields_required: ["loss_value", "step"]
- min_tier_required: 2

**Example 4** (Tier 3):
- bug: micro_step_id counter off-by-one in ZeRO-2
- detection: counter at micro-batch boundaries inconsistent with config
- fields_required: ["micro_step_id", "step"]
- min_tier_required: 3

**Example 5** (Tier 4):
- bug: optimizer momentum buffer corrupted on checkpoint load
- detection: Adam state cksum diverges between save and load
- fields_required: ["optimizer_state_cksum", "param_name"]
- min_tier_required: 4

**Example 6** (Tier 5-6):
- bug: MoE aux-loss tracker double-counts (M-NEW-21)
- detection: Type-B hook counts function calls per step; expected 1, observed 2
- fields_required: ["function_call_count", "step"]
- min_tier_required: 5-6
- rationale: function-call count is a Type-B hook output, beyond Tier 4.

**Example 7** (unobservable):
- bug: TopKGate applies topk to logits instead of softmax (D-NEW-1)
- detection: NONE — softmax is monotonic, topk indices may match; only gate weights differ subtly which won't show in any schema field
- fields_required: []
- min_tier_required: unobservable
- rationale: Source-level algorithmic substitution with no runtime fingerprint.

## INPUT FORMAT (per bug)

```
### Bug ID, Framework, Source pool
### Title
### Category
### Root cause
### Invariant
### Detection method
### Trigger conditions
```

## OUTPUT FORMAT (strict JSON)

```json
{
  "bug_id": "<id>",
  "min_tier_required": 0 | 1 | 2 | 3 | 4 | "5-6" | "unobservable",
  "fields_required": ["param_cksum", "param_name", "tp_rank"],
  "rationale": "<30-200 chars; explain field choice and why not a lower tier>"
}
```

### Output rules

- `min_tier_required` is one of: `0`, `1`, `2`, `3`, `4`, `"5-6"`, `"unobservable"` (string for last two, integer for 0-4).
- `fields_required` is the **minimum** field set, not all fields touched by the fix.
- If unobservable, `fields_required` should be `[]` and rationale must explain why no tier helps.
- Take the **lowest** tier when a bug can be detected by multiple field sets.

### Self-check

- Did I take the lowest possible tier?
- Did I list only fields strictly needed, not all related fields?
- Is unobservable really unobservable, or am I just lazy?
