# S5 — Pattern Catalog Ablation: paper table draft

> **RESULT: SPEC §3 red line triggered.** The frozen Pattern Catalog does **not**
> improve validated-rule yield (A/B = 0.95×, arms overlap), and the two arms
> produce a **byte-identical deployed rule set** (Jaccard = 1.000). The
> contribution-3 claim that the catalog "narrows candidate generation" is not
> supported by this experiment. See `IMPL_NOTES_S5.md` for method and
> `§ Recommended rewrite` below.

## Table: mining funnel with and without the frozen Pattern Catalog

Real LLM (`deepseek-v4-flash`, T=1.0), 5 independent replicates per arm, 112
seed-file cells per replicate across Megatron-LM / DeepSpeed / OLMo / OLMo-core.
Arms differ **only** in the L1 system prompt; L2/L3/L4 are byte-identical.
Median across replicates, [min–max] over 5 replicates.

| Stage | A: with catalog | A+: catalog, harness gap repaired | B: free-form | A/B | A+/B |
|---|---:|---:|---:|---:|---:|
| L1 hypotheses emitted by LLM | 349 [345–351] | 349 [345–351] | 367 [364–374] | 0.95 | 0.95 |
| L1 hypotheses that parse into the schema | 330 [327–343] | 349 [345–351] | 367 [364–374] | 0.90 | 0.95 |
| L2 enumerated candidates | 5,365 [4,853–5,837] | 5,887 [5,268–5,985] | 4,926 [4,738–5,362] | 1.09 | 1.20 |
| L3 healthy-pass | 3,117 [2,764–3,378] | 3,458 [2,986–3,518] | 3,047 [2,936–3,290] | 1.02 | 1.13 |
| L4 adversarial-pass | 364 [355–391] | 379 [369–401] | 360 [343–380] | 1.01 | 1.05 |
| **yield = L4/L2** | **0.0688 [0.0660–0.0732]** | **0.0659 [0.0636–0.0702]** | **0.0724 [0.0709–0.0740]** | **0.95** | **0.91** |
| L3 rejection rate | 0.421 [0.402–0.430] | 0.413 | 0.383 [0.380–0.386] | 1.10 | 1.08 |
| **distinct deployed predicates** | **11** | **11** | **11** | **1.00** | **1.00** |

Arm A+ exists because the first analysis was wrong about *why* 4.6% of A-arm
proposals were discarded (see next section). It gives the catalog every benefit
of the doubt. It does not win either — it does slightly worse, because the
rescued proposals enlarge L2 without adding survivors.

**Caption draft.** *Mining funnel with (A) and without (B) the frozen 35-entry
Pattern Catalog, over 5 replicates of a real LLM on the same 112 framework
source files. The catalog does not raise the validated-rule yield: the two arms
are statistically indistinguishable (0.0688 vs 0.0724, ranges overlap) and
converge on the identical 11-predicate deployed set (Jaccard = 1.000). Arm A+
repairs a harness gap that discards 4.6% of catalog-arm proposals and supplies
the missing operator→relation mapping; it does not help (0.0659), and deploys
the same 11 predicates.*

## The three pre-registered predictions, and what happened

| # | SPEC §3 prediction | Observed | Verdict |
|---|---|---|---|
| 1 | A yield **>** B yield | 0.0688 vs 0.0724, **A < B**, ranges overlap | 🚨 **RED LINE** — "contribution 3 is empty, stop and report" |
| 2 | B's L2 count ≥ A's (free-form diverges wider) | B 4,926 **<** A 5,365 | ⚠️ SPEC flags strawman suspicion — **investigated and ruled out**, see below |
| 3 | B's L3 rejection rate **>** A's (`main.tex:1002` "over-fit workload coincidences") | A 0.421 **>** B 0.383 — **opposite direction** | 🚨 `main.tex:1002` assertion is contradicted, not merely unsupported |
| 4 | Real-SE detection: B < A's 17 | **NOT RUN** — blocked | ⛔ see `detection_comparison.csv` |

### Prediction 2 is not a strawman — it is an artifact of `entities` count

B's L2 count is *lower* only because A emits more `entities` per hypothesis
(mean 2.13 vs 1.85), and `_resolve_hookpoints` yields one hookpoint match per
entity. The B-arm prompt is otherwise identical in task, schema, effort budget
and source chunk (full text in `IMPL_NOTES_S5.md`). B is not handicapped: it
produces *more* usable hypotheses than A (367 vs 330) and a *higher* yield.

## Correction: the 4.6% discard is a harness defect, not an LLM failure

A first pass read the 80 discarded A-arm proposals as the model "inventing
illegal relation types" — i.e. the catalog confusing its own arm. **That reading
was wrong**, and the correction matters because it is the one place where the
catalog critique could have been unfair.

The A-arm prompt renders each entry as `T12: checkpoint-save-completeness
(count_frequency_match)` (`layer1_hypothesis.py:23-27`) and then demands
`relation_type` from a *disjoint* six-value enum. Checked across all 5
replicates: **80 of 80 discarded items carry a valid `catalog_template_id`, and
in 80 of 80 the "illegal" `relation_type` is exactly that template's
`relation_operator`.** The model used the catalog correctly and reported the
operator the catalog showed it. The harness threw the result away.

The root cause is that the catalog's **20 `relation_operator` values** and L2's
**6 `RelationType` dispatch keys** are disjoint taxonomies with **no mapping
between them anywhere in the codebase** — `relation_operator` is referenced in
exactly one place, the prompt string at `layer1_hypothesis.py:25`, and is never
consumed. Arm A+ authors that missing mapping and replays the identical cached
responses. Result: yield **drops** to 0.0659, and the survivor set is unchanged.
So the defect is real and worth fixing on its own merits, but it is not what is
suppressing the catalog's effect.

## Why the catalog cannot matter here — the mechanism

The result is not noise; it is structural, and it is visible in the code.

1. **`catalog_template_id` is inert downstream.** `layer2_enumerate.py:91-93`
   stamps it onto predicates *after* enumeration, as provenance. Nothing reads
   it again — not L3, and not the L4 filter.
2. **All six hardcoded L2 enumerators ignore the Hypothesis entirely.** Verified
   by source inspection over the complete `_ENUMERATORS` table: every one of
   `_enum_cross_rank_equal`, `_enum_payload_field_compare`,
   `_enum_tensor_stat_bound`, `_enum_cross_step_monotonic`,
   `_enum_structural_presence`, `_enum_conditional_check` takes `hyp` as an
   argument and reads **none** of `hyp.entities`, `hyp.dimensions`,
   `hyp.scope_hint`, `hyp.catalog_template_id`. They branch on the **trace
   schema alone**. So the moment either arm proposes the `relation_type`
   `cross_rank_equal`, both arms receive the *same fixed predicate*. Only
   `_parametric_enum` reads the hypothesis (`entities`, `dimensions`) — and its
   fallbacks (below) discard the difference.
3. **Therefore L1's entire causal contribution is the choice of
   `relation_type`** — one of six enum values — and both arms choose from the
   same six. Catalog guidance improves the *wording* of `entities` (A: `param`,
   `grad_norm`, `loss`, `tensor`; B: `world_size`, `n_heads`, `partition_count`,
   `g_idx`), but `_resolve_hookpoints` falls back to *all* hookpoints when an
   entity fails to match, and `_parametric_enum` falls back to *all* numeric
   fields when dimensions fail to match. Both fallbacks erase the difference.

**This is why the survivors are identical.** The 11 deployed predicates —
including all 4 semantic ones (`hyp/cross_rank_equal/replica_cksum`,
`hyp/csm/optim.step.post/state_step_min`, `hyp/sp/build_has_modules`,
`hyp/tsb/no_nan_inf`) — are fully determined by the trace schema, not by L1.

## Recommended rewrite of contribution 3

The current intro claim ("the versioned relation catalog **narrows candidate
generation**") is measurably false as stated: it narrows nothing, because
nothing downstream reads it. Two honest options:

**Option 1 — reframe as provenance/governance (recommended, no new experiment).**
The catalog's real, defensible function is *auditability and versioning*: every
deployed rule carries a template id that ties it to a reviewed, frozen
taxonomy, so rule sets are diffable across releases and a reviewer can ask "which
template does this rule instantiate?". That is a genuine engineering
contribution — it is just not a *candidate-generation* contribution. Drop
"narrows candidate generation" and any efficiency implication.

**Option 2 — make the claim true, then re-measure.** If the catalog is meant to
narrow generation, wire it into L2: have `enumerate_predicates` dispatch on
`catalog_template_id` (and honour `hyp.entities`/`scope_hint`) instead of
ignoring it. That is a real code change, after which this ablation becomes
meaningful. Do not claim the effect before the mechanism exists.

**Either way `main.tex:1002` must change.** The assertion that free-form miners
"over-fit workload coincidences" predicts a higher L3 rejection rate for the
free-form arm. We measured the opposite (A 0.421 > B 0.383). Delete the clause
or restrict it to Daikon-style *value*-invariant miners, which is not what arm B
is.

## Caveats a reviewer will raise, stated up front

- **Funnel counts are per-seed-file sums, not distinct predicates.** L4 = 364
  counts 11 distinct predicates re-derived across 112 seed files (~11× inflation
  in every stage). This is exactly how `run_funnel.py` produces the published
  420/5,334/3,436/357, so the arms remain comparable to each other and to the
  paper — but the published funnel numbers should not be read as "357 distinct
  rules".
- **L4 remains the `PatternGuidedFilterLLM` simulator**, as in the published
  funnel. It is catalog-blind (it branches on hookpoint/field/value and never
  reads `catalog_template_id`), so it cannot favour an arm — but the *absolute*
  yield is a property of that heuristic, not of a real LLM.
- **Detection half of the headline is missing.** SPEC §3 pre-registered *two*
  headline metrics; only yield was obtained. Real-SE replay needs GPU hosts that
  do not resolve from this cluster, and no cached traces exist.
- **Single model.** `deepseek-v4-flash` only. A stronger model might exploit the
  catalog better at L1 — but it could not change the survivor set, because the
  bottleneck is L2's schema-driven enumeration, not L1's proposal quality.
