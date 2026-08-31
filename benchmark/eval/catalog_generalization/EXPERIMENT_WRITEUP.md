# Pattern-Catalog Evaluation — consolidated setup, results, figure plan

Covers the two experiments run for §\ref{subsec:catalog-evaluation}
("Pattern-Catalog Reproducibility and Generalization"):

- **S5 — catalog ablation** (`benchmark/eval/catalog_ablation/`): does the
  catalog change end-to-end mining *yield*? Answer: no.
- **S5b — catalog generalization** (`benchmark/eval/catalog_generalization/`):
  does a frozen catalog *cover held-out bugs better than no catalog*? Answer:
  yes, decisively.

---

## 0. Reconciliation with what the subsection already says (read first)

The current draft already carries a catalog-evaluation story with **different
numbers and a different method**:

| | Current draft (main.tex:704–716) | S5b (this work) |
|---|---|---|
| Reproducibility | human inter-annotator κ (0.954 / 0.843 / 0.772) | not measured (orthogonal) |
| Held-out coverage | **92.3%** on a 78-record locked test | **83.1%** (95.1% observable) on 249 `-NEW` |
| Judge | two human raters | LLM judge (deepseek-v4-flash, T=0), spot-checked |
| Split | 128 seed / 186 dev / 78 locked (case-level) | 128 freeze / 249 `-NEW` (temporal) |
| **No-catalog baseline** | **absent** | **present (B1 free-form, B2 re-mine)** |

The 92.3% figure is load-bearing (also cited at main.tex:188 and :1014), so S5b
does **not** replace it. S5b supplies the one thing the current text lacks: a
**contrast against not using the catalog**. "92.3% coverage" alone cannot answer
a reviewer's "would any invariant vocabulary do that?"; S5b answers it (free-form
reaches 47.8–55.4% under the same judge).

**Decision needed:** do we (a) add S5b as the baseline-contrast half of this
subsection alongside the existing human-judged 92.3%/κ story, or (b) migrate the
whole subsection onto the S5b LLM-judged protocol? Recommendation: **(a)** — keep
92.3%/κ as the human-validated coverage+reproducibility spine, add S5b as
"catalog vs no-catalog" with the coverage-vs-size figure. The two use different
holdouts and judges, so present them as two lenses, not one number.

---

## 1. Experiment setup

### 1.1 Corpus and split (shared)

392 annotated bug records (`benchmark/eval/v2_full/annotations_392_v2.json`),
each with framework, category, `invariant_type`, `required_trace_fields`,
runtime-observability `tier_field`, and a free-text root-cause `rationale`.

- **S5b temporal split:** freeze pool = 128 original bugs (ids without `NEW`,
  excluding the 15 `B*` smoke cases); held-out = **249 `-NEW`** bugs added in a
  later expansion round. The 35-template catalog predates the `-NEW` bugs →
  frozen-before-seeing by construction. Of the 249, **163 are runtime-observable**
  (`tier_field` not `exceeds_tier6`/`none`); the other 86 are unobservable by any
  runtime method and form the fairest denominator's complement.

### 1.2 S5 ablation setup (yield)

- L1 hypothesis proposal run with a **real LLM** (deepseek-v4-flash, T=1.0,
  max_tokens 32768), the only change between arms being the L1 system prompt.
  L2/L3/L4 are byte-identical repo code; L4 uses the catalog-blind
  `PatternGuidedFilterLLM`.
- Arms: **A** (catalog prompt: 35-template table), **B** (free-form prompt:
  same task/schema/effort, no template table), **A+** (A with a repaired harness
  that maps the catalog's `relation_operator` back to a legal `RelationType`).
- 5 replicates/arm × 112 framework source files (Megatron/DeepSpeed/OLMo/
  OLMo-core), 1,120 L1 calls, 0 errors.
- Metric: `yield = L4_pass / L2_enumerated`, plus the distinct deployed-predicate
  set per arm.

### 1.3 S5b generalization setup (coverage)

Three arms scored by **one shared LLM coverage judge** (deepseek-v4-flash,
**T=0**). The judge decides, per held-out bug, whether checking any invariant in
a given taxonomy on a runtime trace would be *violated by that bug*. Strict rule:
source-only bugs (no trace field distinguishes buggy/clean) are NONE for every
arm, so the runtime-unobservable gap is handled identically.

- **A — catalog:** 35 frozen templates. 0 new LLM calls to sustain on new bugs.
- **B1 — free-form frozen:** taxonomy induced (open-coded, no catalog) from the
  S5 free-form L1 proposals; 1,841 proposals collapsed to **15** canonical types.
- **B2 — free-form re-mine:** per held-out bug a free-form LLM proposes 5
  invariants from framework+category+`required_trace_fields` (not the root
  cause); the same judge scores its own proposals. Given a per-bug custom
  proposal and the annotated trace fields → an *upper bound* generous to
  free-form.
- **Judge fairness:** spot-checked. Catalog-only wins are tight, specific
  template matches (T12/T09/T11/T05); in the reverse set the judge grants B2
  passes on unobservable bugs, so it is not biased toward the catalog.

---

## 2. Results

### 2.1 S5 — yield is unchanged by the catalog (Table)

| Arm | L2 enum | L4 pass | **yield** | distinct deployed |
|---|---:|---:|---:|---:|
| A catalog | 5,365 | 364 | 0.0688 | 11 |
| A+ catalog (repaired) | 5,887 | 379 | 0.0659 | 11 |
| B free-form | 4,926 | 360 | 0.0724 | 11 |

Medians over 5 reps. yield_A/yield_B = 0.95; A+ = 0.91. All three arms deploy the
**identical 11 predicates** (Jaccard = 1.000). Mechanism: the L2 enumerators do
not read the hypothesis' catalog id/entities, so the catalog cannot change the
funnel. → the catalog's value is not yield.

### 2.2 S5b — the catalog generalizes and beats no-catalog (Table)

| Arm | size | all held-out (249) | runtime-observable (163) | marginal cost/new bug |
|---|---:|---:|---:|---:|
| **A catalog (frozen)** | 35 | **207 = 83.1%** | **155 = 95.1%** | **0 calls** |
| B2 free-form re-mine (UB) | per-bug | 138 = 55.4% | 99 = 60.7% | 1 call/bug |
| B1 free-form frozen | 15 | 119 = 47.8% | 90 = 55.2% | 0 (non-reproducible) |

**Coverage-vs-size (catalog, from matched-template frequencies, no extra calls):**

| top-K templates | 5 | 10 | 15 | 20 | 25 | 35 |
|---|---:|---:|---:|---:|---:|---:|
| held-out coverage | 38.2 | 58.2 | 67.5 | 74.3 | 78.3 | 83.1 |

Key contrasts:
- **Equal size (15):** catalog 67.5% vs free-form 47.8% (+19.7 pts).
- Catalog **top-10 (58.2%)** already beats the re-mine upper bound (55.4%).
- All 35 templates are used (no dead entries).
- Honest miss: of A's 42 misses, 34 are runtime-unobservable; one class
  (`numerical_consistency`) appears only post-freeze.

---

## 3. Figure plan

House style: SIGMOD half-column, 1.64in wide, 7pt serif, palette OK `#2166ac`
(catalog), ACCENT `#b2182b` (free-form), GRAY `#9a9a9a` (reference), matplotlib
→ PDF (`pdf.fonttype=42`). Matches `figures/gen_ablation_v2.py`.

### Primary figure — "Catalog generalizes at equal size and fixed cost"

Single panel, the coverage-vs-size curve with both baselines overlaid. It proves
three things at once: catalog dominates free-form at equal size, catalog beats
re-mine, and coverage saturates.

- X: taxonomy size (invariant types), 5–35.
- Y: held-out coverage %, 0–100.
- **Catalog**: solid blue line+markers through the 6 points to 83.1%.
- **Free-form frozen B1**: one red marker at (15, 47.8) — vertically below the
  catalog's (15, 67.5), the money comparison.
- **Re-mine B2**: gray dashed horizontal at 55.4, labeled "re-mine upper bound";
  the catalog line crosses it near size 10.
- **Observable ceiling**: faint dotted horizontal at 95.1% with a small label.

```
 cov%
 100 |············································· 95.1% observable ceiling
  83 |                                    ●───  catalog@35
  80 |                         ●
  74 |                   ●
  68 |             ●  ← catalog@15 = 67.5   (equal-size win: +19.7)
  58 |       ●
  55 |----△--------------------------------- re-mine B2 upper bound = 55.4
  48 |     ▲  free-form frozen B1 @15 = 47.8
  38 |  ●
     +--+----+----+----+----+----+----+----
        5   10   15   20   25   30   35   taxonomy size (#invariant types)
```

Caption skeleton: *"A catalog frozen before the held-out bugs covers 83.1% of
249 later-discovered bugs (95.1% of runtime-observable ones). At equal taxonomy
size (15) it covers 67.5% vs a free-form taxonomy's 47.8%; even a free-form LLM
re-mining every bug reaches only 55.4%, which the catalog's ten most-used
templates already exceed. The catalog sustains this at zero marginal cost per new
bug; re-mining pays one LLM call per bug and yields a different set each run."*

### Optional companion panel (only if a 2-panel is wanted, stacked like the
funnel fig)

(b) free-form does not self-curate: bar of induced type sizes — 1,841 proposals
→ 15 types, the top bucket (`cross-rank-consistency`) alone = 280, plus a
`miscellaneous` catch-all = 50. Visual argument for *why* free-form transfers
worse: it collapses to a few coarse buckets while the catalog keeps 35
fine-grained obligations. This doubles as the "reproducibility" point (free-form
induction is unstable/coarse).

### Relationship to existing figures

`figures/fig_generalization.pdf` already exists — check whether it renders the
92.3% human-judged story; if so, the S5b curve is a **new** figure (propose
`fig_catalog_generalization.pdf`), not a replacement. The S5 ablation belongs in
`fig_funnel_ablation_v2.pdf`'s neighborhood or an appendix honesty note; it is a
table, not a headline figure.

### Data sources for the figure

- Curve + baseline points: `generalization_summary.csv` (this dir).
- Per-bug verdicts (for any error bars / robustness): `cov_A.jsonl`,
  `cov_B1.jsonl`, `cov_B2.jsonl`.
