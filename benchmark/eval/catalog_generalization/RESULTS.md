# S5b — Pattern-Catalog Generalization (RQ for §"Pattern-Catalog Reproducibility and Generalization")

> **Result: the frozen catalog generalizes, and the no-catalog baselines do not
> match it.** A 35-template catalog frozen *before* the held-out bugs existed
> covers **83.1%** of 249 later-discovered bugs (**95.1%** of the runtime-
> observable ones). A size-matched free-form taxonomy reaches 47.8%, and even a
> free-form LLM re-mining every bug on the fly — given generous hints — reaches
> only 55.4%. The catalog wins at equal taxonomy size and at zero marginal cost.

This is the complement to the S5 ablation. S5 showed the catalog does **not**
change end-to-end validated-rule *yield*, because the current L2 enumerator
ignores it. This experiment measures what the catalog is actually *for* — the
coverage and transfer of the taxonomy itself — and there its value is real and
large. Contribution 3 should be argued on **this** axis, not on yield.

## Design

- **Temporal split (frozen-before-seeing by construction).** Freeze pool = 128
  original bugs; held-out = **249 `-NEW` bugs** discovered in a later expansion
  round. The 35-template catalog predates the `-NEW` bugs, so covering them is a
  genuine generalization test, not a fit to labels.
- **Three arms, one shared LLM judge (temperature 0), identical protocol.** The
  judge decides, per held-out bug, whether checking any invariant in a given
  taxonomy on a runtime trace would be *violated by that bug*. Strict rule:
  source-only bugs (no runtime trace field distinguishes buggy from clean) are
  NONE for every arm, so the paper's runtime-unobservable gap is handled
  identically and cannot bias the comparison.
  - **A — catalog:** the 35 frozen templates.
  - **B1 — free-form frozen:** a taxonomy induced (open-coded, no catalog) from
    the free-form L1 proposals of the S5 ablation, then size-limited. Induction
    collapsed 1,841 proposals into only **15** canonical types — dominated by a
    single `cross-rank-consistency` class (280) plus a `miscellaneous` bucket —
    so free-form is intrinsically coarser than the curated 35.
  - **B2 — free-form re-mine:** for each held-out bug a free-form LLM proposes 5
    invariants from framework+category+trace-fields (but not the root cause);
    the same judge scores its own fresh proposals. This gives free-form a per-bug
    custom proposal *and* the annotated trace fields — deliberately generous, so
    a catalog win is conservative.

## Results

| Arm | taxonomy size | all held-out (249) | runtime-observable (163) | marginal cost / new bug |
|---|---:|---:|---:|---:|
| **A — catalog (frozen)** | 35 | **207 = 83.1%** | **155 = 95.1%** | **0 LLM calls** |
| B2 — free-form re-mine (upper bound) | per-bug | 138 = 55.4% | 99 = 60.7% | 1 propose call/bug |
| B1 — free-form frozen | 15 | 119 = 47.8% | 90 = 55.2% | 0 (non-reproducible induction) |

**Equal-size control (from `cov_A.jsonl` matched-template frequencies, no extra
calls).** Catalog coverage-vs-size: top-5 38.2%, top-10 58.2%, top-15 67.5%,
top-20 74.3%, top-25 78.3%, top-35 83.1%. All 35 templates are used (35 distinct
matched; no dead entries).

- At **equal size 15**: catalog 67.5% vs free-form B1 47.8% (+19.7 pts).
- Catalog's **top-10 templates alone (58.2%)** already beat the re-mine upper
  bound B2 (55.4%).

## Why this is a real result and not judge bias

- **Spot-check of catalog-only wins.** Of the 87 bugs A covers but B2 misses, the
  sampled matches are tight and specific, not loose: frozen params omitted from a
  ZeRO3 checkpoint → T12 checkpoint-save-completeness; wrong grad-norm source
  under precision-aware opt → T09 gradient-norm-computation-fidelity; wrong
  divisor in expert-checkpoint metadata → T11 checkpoint-restore-state-equality;
  bf16 logits into an fp32 sigmoid → T05 designated-dtype-fidelity. The catalog
  genuinely contains a template expressing each; free-form failed to propose it.
- **No pro-catalog bias.** In the reverse set (18 bugs B2 covers but A misses),
  several are `exceeds_tier6` runtime-unobservable cases where the judge grants
  B2's fresh proposal a pass. The judge lets the baselines win where it can, so
  it is not systematically favouring the catalog.
- **The honest miss.** Of A's 42 total misses, 34 are runtime-unobservable
  (`exceeds_tier6`) — the paper's fundamental gap, uncoverable by any runtime
  method. Only 8 observable held-out bugs escape the frozen catalog, and one
  invariant class (`P2` / `numerical_consistency`) appears only post-freeze. The
  catalog generalizes strongly but not perfectly, which is the credible outcome.

## What to claim in the paper

The catalog's contribution is **generalization and reproducibility at fixed
cost**, not higher mining yield:

1. A small frozen, versioned taxonomy (35 entries) covers **95% of runtime-
   observable bugs discovered after it was frozen** — it transfers to new bugs
   without re-mining.
2. It does so at **equal or better coverage-per-entry** than a free-form
   taxonomy (67.5% vs 47.8% at size 15) and **beats per-bug re-mining** (83.1%
   vs 55.4%) at **zero marginal LLM cost**, whereas re-mining pays one LLM call
   per new bug and yields a different, non-reproducible set each run.
3. Free-form proposals do not curate themselves: 1,841 of them open-code into
   just 15 coarse types, half of them one `cross-rank-consistency` bucket. The
   catalog's value is precisely the curation the baseline lacks.

**Do not** reuse the S5 "narrows candidate generation / higher yield" wording —
that remains false until L2 reads the catalog (S5 Option 2). Argue reproducibility
+ generalization + fixed cost, which is what these numbers support.

## Caveats a reviewer will raise

- **Coverage is LLM-judged.** One model (`deepseek-v4-flash`, temp 0), one judge
  prompt, shared across arms. Spot-checked above; not human-audited at scale. A
  human-rated calibration subset would harden it.
- **B1 induction is itself an LLM step** and is not reproducible run-to-run; its
  15-type output is one draw. The equal-size *catalog* curve does not depend on
  it.
- **Split is temporal within one corpus.** `-NEW` bugs share frameworks with the
  freeze pool; this tests transfer to new bugs, not new frameworks. A
  leave-one-framework-out cut is the stronger follow-up (the label-level preview
  showed 100% pattern transfer across frameworks, but that must be re-run through
  this judge protocol, not the pre-baked `pattern_id`).

## Files

| File | Contents |
|---|---|
| `build_data.py` | temporal split + catalog taxonomy A |
| `judge.py` | shared coverage judge (temp 0, strict runtime-violated criterion) |
| `run_coverage.py` | judge a taxonomy over all 249 held-out (arms A, B1) |
| `run_b2_remine.py` | arm B2 propose+judge per bug |
| `induce_freeform_taxonomy.py` | build B1's free-form taxonomy by open-coding |
| `cov_A.jsonl`, `cov_B1.jsonl`, `cov_B2.jsonl` | per-bug judge verdicts (auditable) |
| `taxonomy_catalog.json`, `taxonomy_freeform.json` | the two frozen taxonomies |
| `heldout_bugs.json`, `freeze_bugs.json` | the split |

Reproduce: `python build_data.py && python induce_freeform_taxonomy.py`, then
`run_coverage.py --taxonomy taxonomy_catalog.json --out cov_A.jsonl`,
`... --taxonomy taxonomy_freeform.json --out cov_B1.jsonl`, and
`run_b2_remine.py`. (`DEEPSEEK_API_KEY` in env; all runners resume from their
`.jsonl`.)
