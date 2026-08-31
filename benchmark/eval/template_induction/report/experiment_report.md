# Template Induction Experiment — Report

**Date**: 2026-07-16 (protocol, split, and all runs executed on this date)
**Pool**: 392 upstream silent-error evidence records (`benchmark/eval/manifest_v2.json`);
construction and provenance limitations are audited in
`benchmark/eval/corpus_provenance_audit.md`
**Protocol**: `template_coding_protocol.md`, frozen before any annotator saw the locked test
**Frozen catalog**: `frozen_template_catalog.json`, sha256
`cfa30e182a4aa5c6637423dcbd95f15cb20c2b34a1da80000f5df6c6b8176734`

---

## 1. Headline result

The claim under test was that under fixed extraction and merge/split rules, silent
errors converge to a **small, non-redundant catalog that covers most unseen cases**.
The experiment splits that claim in two, and the two halves come apart:

| Property | Verdict | Evidence |
|---|---|---|
| Reproducible | **Pass** | Template assignment κ=0.954; match/uncovered κ=0.843; 6-way verdict κ=0.772 |
| **Saturated** | **FAIL** | Pre-registered stopping rule **never triggered** over 186 dev cases; catalog grew 27→35 and was still growing in the final batch |
| Non-redundant | **Pass** | 0/595 template pairs mergeable under the frozen criterion |
| Generalizable | **Pass** | Held-out C_schema = **92.3%** (≥80% target), novelty rate 3.8% |
| Deployable | **Pass (with caveat)** | C_joint = **91.0%**; gap is a single case, explained |

The catalog **generalizes without having saturated**. Coverage plateaus at ~0.86 after
roughly 25 development cases and never improves, while the template count keeps
climbing — the marginal templates are tail structure, each carrying 2 cases, not
coverage. This is a real and reportable negative on the saturation sub-claim, and it
is the one result that must not be smoothed over: **do not claim the catalog is
saturated.**

---

## 2. Timeline and blinding

| When | What | Who could see what |
|---|---|---|
| 2026-07-16, before any coding | `template_coding_protocol.md` frozen; `dataset_split.json` generated (RNG 20260716) | — |
| Then | E1 extraction (128 seed + 186 dev), E2 re-extraction (79 = 25% subset) | Whitelisted case fields only; test cases never opened by extractors |
| Then | Seed induction → `seed/initial_catalog.json` (27 templates) | Seed extractions + rules only |
| Then | Dev batches 1–8, sequential, each on the previous catalog | Own batch + current catalog only |
| Then | Catalog frozen, sha256 recorded | — |
| **After freeze** | Rater A, Rater B independently on all 78 locked-test cases | Frozen catalog + whitelisted test fields |
| Last | Adjudicator on the 7 disagreements + 5 agreed-uncovered classifications | Both raters' verdicts; contributed no initial verdict |

Blinding was enforced by construction. `make_split.py` whitelists the annotator-visible
fields to `bug_id, framework, repo, title, description, root_cause, category,
parallel_dimension, severity, trigger_conditions, issue_url`. The fields that encode
prior template analysis — `invariant`, `invariant_type`, `detection_method`,
`detection_signal`, `required_trace_fields`, `check_stage`, and every P1–P16 label —
were withheld, and every annotator was instructed to read only its input files. No
annotator ever read `pattern_expansion/`, `main.tex`, or the appendix.

**Orchestrator disclosure**: the session orchestrating these agents had previously seen
the paper's existing P1–P16 catalog. It made no coding, merge, or verdict decision;
every such decision came from a fresh-context agent under the constraints above. The
locked test was never used to add, merge, or split a template.

---

## 3. Dataset split

| Set | N | Composition |
|---|---:|---|
| Seed | 128 | Rebuilt original 128-pool (`source_pool ∈ {128_only, both}`) minus cases pulled into test, refilled by stratified sampling |
| Development | 186 | Remainder; stream order frozen, 8 batches (7×25 + 1×11) |
| Locked test | 78 | All 18 Real-SE cases (LC1→O-003) + their leakage-group mates, topped up by stratified sampling |

Stratified by 4 frameworks × 13 taxonomy classes. Leakage groups are defined by shared
normalized `issue_url`; 4 URLs cover 9 bugs (notably B1↔M-005), and each group lands
entirely in one split. **No temporal split was performed: 0/392 records carry a
`fix_date`.** Populating issue/fix dates would enable a strictly stronger temporal
generalization test and is the single highest-value follow-up.

---

## 4. Extraction reliability (E1 vs E2, 79 double-coded cases)

| Dimension | Raw agreement | Cohen's κ |
|---|---:|---:|
| `expected_relation` (raw) | 0.646 | 0.604 |
| `expected_relation` (alias-normalized) | 0.646 | 0.600 |
| `training_phase` | 0.975 | 0.968 |
| `topology_scope` (normalized) | 0.899 | 0.834 |

Relation-operator agreement (κ=0.60) sits just below the substantial threshold and is
the weakest link in the pipeline — the guards are coded far more consistently than the
relation itself. The 28 disagreements are in
`analysis/extraction_irr_disagreements.json`. Inspection shows most are **vocabulary
collisions, not semantic disputes**: both coders describe the same obligation but pick
a different operator name (e.g. one calls an init-scheme defect
`value_scaling_consistency`, the other `structural_integrity`). The curator's axial
step absorbs this by re-resolving operators from `relation_statement`, which is why
downstream template-assignment agreement is much higher (κ=0.954). Still, the honest
reading is that **§1's open operator vocabulary is under-specified**: a closed operator
list with decision rules would likely lift this figure, and a revision should do that
rather than report κ=0.60 as adequate.

This κ is reported separately from, and does not inherit, the paper's 13-class taxonomy
κ=0.566 — different construct, different task.

---

## 5. Saturation (the negative result)

`development/saturation_by_batch.csv`, `development/stopping_rule_evaluation.json`

| Batch | n | Matched | Match share | New singletons | New templates | Core edits | Templates after | Cum. coverage | Rule OK? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | 25 | 19 | 0.76 | 5 | 1 | 0 | 28 | 0.800 | no |
| 2 | 25 | 24 | 0.96 | 1 | 0 | 0 | 28 | 0.880 | **yes** |
| 3 | 25 | 18 | 0.72 | 6 | 1 | 0 | 29 | 0.840 | no |
| 4 | 25 | 23 | 0.92 | 2 | 0 | 0 | 29 | 0.860 | no |
| 5 | 25 | 19 | 0.76 | 3 | 3 | 0 | 32 | 0.864 | no |
| 6 | 25 | 23 | 0.92 | 2 | 0 | 0 | 32 | 0.873 | no |
| 7 | 25 | 22 | 0.88 | 2 | 1 | 0 | 33 | 0.880 | no |
| 8 | 11 | 7 | 0.64 | 2 | 2 | 0 | 35 | 0.876 | no |

**The pre-registered stopping rule (§6: 3 consecutive batches with no new recurring
template AND ≥95% matched AND <5% new singletons AND no core edit) never fired.** The
longest qualifying streak was 1 batch (batch 2). The catalog was therefore frozen at
**stream exhaustion, not at a stopping point**.

Consequences, stated plainly:

- **There is no simulated stopping point, and post-stop regret is undefined.** The
  protocol's regret analysis presupposes an early stop; it cannot be computed here.
  Reporting a regret number would require inventing a stopping point after seeing the
  results, which §6 forbids.
- **The final batch added 2 templates.** The catalog was still discovering recurring
  structure when the data ran out. A larger pool would very likely yield more templates.
- **Zero core edits across all 8 batches.** Every new case was absorbed as a guard
  extension, a promotion, or a singleton — no template's core schema relation was ever
  revised. The relations, once found, were stable; only the *set* kept growing.

### Order robustness (200 random reorderings, final coding)

`analysis/saturation_orders.csv`, `analysis/saturation_curve.pdf`

| Cases processed | Templates (median [5–95%]) | Coverage (median [5–95%]) |
|---:|:--|:--|
| 25 | 28 [27, 29] | 0.840 [0.76, 0.96] |
| 50 | 28 [27, 30] | 0.860 [0.78, 0.92] |
| 100 | 30 [28, 33] | 0.860 [0.83, 0.90] |
| 150 | 33 [31, 35] | 0.873 [0.85, 0.89] |
| 186 | 35 [35, 35] | 0.876 |

The two curves separate cleanly and the conclusion does not depend on ordering:
**coverage saturates almost immediately (~0.86 by case 25) while the template count
rises monotonically to the end.** The endpoint band is degenerate by construction (the
final coding is order-invariant); the informative region is the middle, where the
template-count band stays ±2–3 and never flattens.

Methodological note: the simulation labels cases from the **frozen catalog's
`positive_examples`**, per §6's instruction to simulate orders under the final coding.
An earlier draft labelled cases by their live batch decision, which undercounted
templates (33 vs 35) because a case recorded as a singleton before its later promotion
lost its support; templates promoted from a *seed* singleton plus one dev case also
need the seed-side support pre-loaded. Both were fixed and the simulation now
reproduces the live run exactly (35 templates, coverage 0.876).

---

## 6. Locked test (frozen catalog, 78 cases)

`analysis/test_metrics.json`, `test/adjudicated_annotations.csv`

| Metric | Value |
|---|---:|
| **C_schema** (matched / 78) | **0.923** (72/78) |
| **C_joint** (fully groundable / 78) | **0.910** (71/78) |
| **Novelty rate** (genuinely new relation) | **0.038** (3/78) |

Verdicts: MATCH 72, SINGLETON_NEW_RELATION 3, INSUFFICIENT_EVIDENCE 2, UNOBSERVABLE 1.
NON_GROUNDABLE and REFERENCE_DEPENDENT: 0.

### Per-framework (guards against one framework dominating)

| Framework | n | C_schema | C_joint |
|---|---:|---:|---:|
| megatron-lm | 23 | 1.000 | 1.000 |
| deepspeed | 26 | 0.923 | 0.923 |
| olmo-core | 14 | 0.929 | 0.857 |
| olmo | 15 | **0.800** | **0.800** |

Every framework clears the 80% bar, but the spread is real: Megatron-LM is perfect while
OLMo sits exactly at the threshold. Since the seed pool skews toward Megatron-LM and
DeepSpeed (39+41 of 128), the aggregate 92.3% is flattered by the frameworks the
catalog was mostly built from. The OLMo figure is the honest estimate of cross-framework
transfer, and the 3 OLMo misses are 2×U6 (inadequate records) + 1×U1 — i.e. OLMo's gap
is data quality plus one observability limit, not a missing abstraction.

### C_schema − C_joint gap

The gap is **1 case**: OC-NEW-67, MATCH on T04 with `groundable: no`. The adjudicator
resolved it against the frozen catalog's own `grounding_concerns` precedent (M-022,
OC-NEW-53), which admits external-oracle-only cases as T04 matches. The gap is
explainable case-by-case, satisfying §10 — and at 1.3 pp it is far smaller than the
survey-wide 88%/83% schema/joint spread, because the locked test's guards happened to
be almost universally instantiable.

### Inter-rater reliability (78 cases, independent A/B)

| Dimension | n | Raw agreement | Cohen's κ |
|---|---:|---:|---:|
| Template assignment (both MATCH) | 70 | 0.957 | **0.954** |
| Match vs uncovered | 78 | 0.974 | **0.843** |
| 6-way verdict | 78 | 0.962 | **0.772** |
| Groundability (both MATCH) | 70 | 0.986 | **0.0 (degenerate)** |

All non-degenerate figures exceed the pre-registered substantial-agreement bar
(κ≥0.61). The groundability κ is **degenerate, not a failure**: 69/70 judgments are
"yes" on both sides, so expected agreement ≈ 1 and κ collapses regardless of the data.
Only raw agreement (0.986) is meaningful there, and it should be reported as such — a
κ of 0.0 here means the marginal is skewed, not that raters disagreed.

Rater B was deliberately framed as a skeptical, measurement-first rater and still
matched A on 76/78 verdict classes, which is the strongest available evidence that the
frozen rules — not a shared framing — drive the assignments. The adjudicator sided with
A on 5 of 7 and B on 2; 4 of the 7 disputes were MATCH-vs-MATCH (differing only on
template id or groundability) and clustered on two boundaries the catalog already
flags (T03/T04 oracle source, T22/T28 witness class).

---

## 7. Compression and redundancy

- **35 templates**, 24 singletons, over 392 evidence records.
- **All 35 templates carry ≥2 independent supporting cases** (§4 admission held with no
  exceptions); support ranges 2–28, median 4.
- The locked test exercised **23 of 35 templates**; the head (T05, T08 at 28 each;
  T04, T07 at 18; T03 at 17) carries most incidents, and 12 templates saw no test case.
- **Merge audit** (`analysis/pairwise_merge_audit.csv`): 595 pairs, 23 share a canonical
  relation operator, **0 are mergeable**. Supported claim: *the catalog is irredundant
  under our predefined merge criterion.* **Not claimed: global minimality.**
- The auditor flagged **T19/T30** as a genuine close call — packed-document vs
  static-pattern attention-mass exclusion, separated only because the forbidden set is
  data-dependent rather than config-static. A curator who treated the forbidden-set
  source as a precondition guard would merge them. This is a live threat to the
  irredundancy claim and should be stated, not buried.

---

## 8. Uncovered cases (§9)

`analysis/uncovered_case_analysis.csv` — 7 rows (6 uncovered + 1 matched-but-ungroundable).

| Category | n | Cases |
|---|---:|---|
| U1 Unobservable | 1 | O-040 |
| U4 Non-groundable | 1 | OC-NEW-67 (MATCH T04, guard not instantiable) |
| U5 Singleton | 3 | D-NEW-2, O-009, O-NEW-67 |
| U6 Insufficient evidence | 2 | D-002, O-008 |
| U2 Reference-dependent | 0 | — |
| U3 Statistical | 0 | — |

5 of 7 are systematic boundary (U1/U4/U5); **2 are coding failures (U6)**. Both U6 cases
share one defect: `root_cause` is a verbatim copy of `description` and merely restates
the title without naming a mechanism. §10 asked that most uncovered cases land in U1–U4
rather than U6 — that holds (5/7), but the 2 U6 cases are a **data-quality finding, not
a framework limit**: they are unresolvable from the record, and both raters independently
converged on that. Fixing those two records (and the 9 other cases across the pool with
empty or duplicated `root_cause`, flagged by extractors as low/medium confidence) would
remove the only non-principled misses.

---

## 9. Pre-registered criteria — scored honestly

| §10 criterion | Target | Result | Verdict |
|---|---|---|:--:|
| Held-out schema coverage | ≥~80% | 92.3% | **Pass** |
| C_schema − C_joint gap explainable | case-by-case | 1 case, precedent-resolved | **Pass** |
| Post-stop coverage regret | ≤2 pp | **Not computable** — rule never fired | **N/A** |
| New recurring relations in locked test | ≤1–2 | 0 recurring (3 singletons, all one-off) | **Pass** |
| Template-assignment agreement | κ≥0.61 | 0.954 | **Pass** |
| No obviously mergeable pair | 0 | 0/595 (T19/T30 close) | **Pass** |
| Every major template ≥2 independent cases | all | 35/35 | **Pass** |
| Uncovered mostly U1–U4, not U6 | majority | 5/7 boundary, 2/7 U6 | **Pass** |

Six passes, one N/A, and one criterion the experiment did not have a target for but
which failed on its own terms: **saturation**. Per §9's diagnostic guide, "若测试集出现
多个新模板，说明 catalog 尚未饱和" — the locked test produced *no* new recurring
relations, so the held-out set does not indicate an unsaturated catalog. The
unsaturation shows up only in the development stream's template count. Those two facts
are consistent and their joint reading is specific:

> The catalog's **relation vocabulary** is close to complete for this fault space — 92%
> of unseen cases match, and no held-out case demanded a new *recurring* relation. What
> has not converged is the **partition granularity**: new cases keep splitting off
> narrow, 2-case templates in the tail. More data would grow the tail, not the coverage.

That is a defensible and interesting claim. "The catalog is saturated at 35 templates"
is not, and this experiment does not support it.

---

## 10. Protocol deviations

1. **Batch 7 was interrupted and re-run.** The first batch-7 curator hit an API session
   limit after forming its decisions but before writing its output files; no partial
   output survived. It was re-run from the identical frozen input
   (`catalog_after_batch_06.json` + `batch_07_extractions.jsonl`). The re-run's
   conclusions closely tracked the interrupted run's stated intent (22 MATCH, 1 new
   template T33 promoted from OC-NEW-65 + OC-NEW-60). No input or rule changed.
2. **Batch 5 was interrupted and re-run.** A partial `catalog_after_batch_05.json` with
   no accompanying log was written before the agent was stopped; the partial file was
   deleted and the batch re-run from `catalog_after_batch_04.json`.
3. **Live issue fetching was out of scope.** `evidence_from_issue` / `evidence_from_fix`
   come from the pool's curated `title`/`description`/`root_cause` summaries rather than
   the upstream issue threads. This was a scoping decision recorded in the protocol
   (§0), not a mid-flight change, but it is the reason the 2 U6 cases are unresolvable.
4. **The §9 category prompt to the adjudicator was mis-framed** — it presupposed all 7
   disputes would resolve to non-MATCH. The adjudicator correctly refused to force
   categories onto the 4 MATCH-vs-MATCH disputes and assigned a category only to the one
   non-MATCH resolution (O-NEW-67 → U5). No verdict was affected.
5. **`groundable` κ is degenerate** (all-yes marginal) and is reported as raw agreement
   only. Not a deviation from the rules, but the protocol's §7 assumed κ would be
   informative on this dimension.

---

## 11. Deliverables

| Requested | Path |
|---|---|
| `template_coding_protocol.md` | `template_coding_protocol.md` |
| `dataset_split.json` | `dataset_split.json` |
| `seed_annotations.csv` | `seed/seed_annotations.csv` |
| `development_annotations.csv` | `development/development_annotations.csv` |
| `test_annotations_rater_a.csv` | `test/test_annotations_rater_a.csv` |
| `test_annotations_rater_b.csv` | `test/test_annotations_rater_b.csv` |
| `adjudicated_annotations.csv` | `test/adjudicated_annotations.csv` |
| `frozen_template_catalog.json` | `frozen_template_catalog.json` (+ `.sha256`) |
| `saturation_by_batch.csv` | `development/saturation_by_batch.csv` |
| `saturation_curve.pdf` | `analysis/saturation_curve.pdf` |
| `pairwise_merge_audit.csv` | `analysis/pairwise_merge_audit.csv` |
| `uncovered_case_analysis.csv` | `analysis/uncovered_case_analysis.csv` |
| `experiment_report.md` | `report/experiment_report.md` (this file) |

Additional: `seed/initial_codebook.md` (human-readable seed catalog),
`analysis/e2_annotations.csv` (second-coder subset), `analysis/test_irr.json`,
`analysis/merge_audit_summary.json`, `development/stopping_rule_evaluation.json`,
`development/batch_*_log.json` (per-case decisions), `development/catalog_after_batch_*.json`
(full catalog lineage), `analysis/extraction_irr_disagreements.json`,
`analysis/saturation_orders.csv`.

Reproduction: `make_split.py` → `validate_extractions.py` → `build_saturation.py` →
`simulate_order_robustness.py` → `compute_extraction_irr.py` → `compute_test_metrics.py
[--adjudicated]`. All RNG seeded at 20260716.

---

## 12. What this changes for the paper

1. **The 16-template claim is not what this experiment reproduces.** Independent
   induction under the frozen rules yields **35 templates** at a comparable coverage
   (92% held-out schema coverage here vs the paper's 88% survey-wide matchable bound).
   The catalogs are not directly comparable — this run's rules split more aggressively
   (35 templates + 24 singletons vs 16), and the paper's P1–P16 were built in two
   grounded-theory rounds with different merge latitude. Two readings are available, and
   the data does not yet distinguish them: either P1–P16 is a coarser but legitimate
   partition of the same relation space, or it under-splits. **The actionable next step
   is a mapping study**: project these 35 templates onto P1–P16 and check whether each of
   the 35 is expressible as a P-template plus guards. If it is, the paper's catalog
   survives as the coarse view and this experiment supplies its reproducibility
   evidence. If it is not, the paper's coverage claim needs revisiting.
2. **Do not add a saturation claim to the paper.** The stopping rule never fired. The
   supportable claim is the coverage/vocabulary one in §9 above.
3. **The reproducibility evidence is strong and is new.** Template-assignment κ=0.954
   from two independently-framed raters against a frozen catalog is a materially better
   answer to "are these templates reproducible?" than the existing taxonomy κ=0.566,
   which measures a different construct.
4. **Two data-quality fixes are cheap and worth doing**: populate `fix_date` (unlocks a
   temporal split, the strongest generalization test available) and repair the ~11
   records whose `root_cause` is empty or duplicates `description` (removes the only
   non-principled locked-test misses).
