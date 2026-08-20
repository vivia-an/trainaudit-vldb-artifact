# Paper-side actions (report only — `main.tex` has not been modified)

Items in `overleaf/sdc_llm_icml_2025/main.tex` that need an author decision before
submission. Each gives the exact location, the evidence, and a suggested edit. Item 3 was
raised in an earlier pass and is **withdrawn** — it is recorded here rather than deleted so
the reasoning is auditable.

---

## 1. Content exceeds the 12-page limit — 25 lines over

PVLDB Vol. 20: "up to 12 pages excluding references", and "All content, including any
appendices and acknowledgements but excluding the references, must fit on the given
number of pages."

Measured on a clean build (0 errors, 0 undefined references or citations): content ends
on page 13 at y≈358 pt, `REFERENCES` starts at y=364.9 pt on the same page. The overflow
is **25 lines / 252 pt / 0.43 of a column** — the tail of §6 Related Work plus all of
§7 Conclusion.

§6 is 323 words (~31 lines) and §7 is 92 words (~9 lines), so the room cannot come from
those two alone. Full breakdown and the mechanical levers are in `PAGE_BUDGET.md`; the
short version is that trimming the four full-column plots by 10% recovers roughly 90 pt
without touching a word, and the remaining ~160 pt has to come from prose or from
dropping a float.

Related: `main.tex:1159` still reads `%% Acknowledgements removed for double-blind
review`. Single-blind permits them again, but they count toward the same 12 pages —
decide the two together.

---

## 2. The production discovery can now be cited

`main.tex:220-223` (abstract), `main.tex:277` (intro), `main.tex:1097` (§5.6) describe
the flagship out-of-pool result as "reported the fault upstream, where maintainers
confirmed and fixed this previously unrecorded bug" — with no identifier, because that
is what double-blind required. VLDB is single-blind, so the identifier can be given,
which turns the paper's strongest claim from unverifiable into checkable.

Verified upstream state (2026-08-19, details in `UPSTREAM_BUG_EVIDENCE.md`):

- `NVIDIA/Megatron-LM` **issue #4641** — "[BUG] Muon + PP + MTP: tied word_embeddings on
  MTP-only last PP stage is routed to Muon instead of Adam" — **closed**, labeled a bug,
  assigned to a maintainer, opened 2026-05-06 by `yezhengmao1` (co-author Zhengmao Ye).
- `NVIDIA/Megatron-LM` **PR #4642** — the authors' fix — **closed, not merged**; a
  duplicate (PR #5034) landed first.

Suggested edit: cite issue #4641 at `main.tex:1097`. Keep the attribution as it stands —
"maintainers confirmed and fixed" is accurate and should **not** be strengthened into a
claim that the authors' patch was merged, because it was not.

---

## 3. ~~The §5.2 footnote names a case the data does not support~~ — withdrawn, the footnote is right

An earlier pass flagged the §5.2 footnote ("Sixteen fixed sides complete cleanly; M-020's
upstream fix instead rejects the faulty configuration as intended") as unsupported,
because `benchmark/eval/results.csv` records M-020's fixed side as `CLEAN`. That file
scores a superseded case set and the wrong M-020 harness.

The current run says otherwise. `benchmark/eval/real_sdc/logs/smoke/M-020_smoke.log`:

```
>>> Buggy: BUG DETECTED (expected)
>>> Fixed: ASSERT fired (expected — bug prevented)
```

Extracting all 21 smoke logs and scoping to the 17 current cases
(`real_sdc/extract_replay_outcomes.py` → `real_se_replay_outcomes.csv`) gives **17 buggy
sides detected, 16 fixed sides clean, 1 assertion-fired (M-020), 0 false positives** —
the footnote, `\NumRealSEDet`, `\NumFixedReplay` and `\FixedFPFrac` all exactly as
published. No edit needed here.

---

## 4. What the appendix is carrying, now that appendices count toward the 12 pages

PVLDB counts appendices inside the limit, and reviewers are not obliged to read
supplemental material. The main text makes **18 `Appendix~\ref` calls across 9 targets**,
and they are not all detail — roughly two thirds carry the evidence for a main-text claim:

| Target | What the main text leans on it for | Load-bearing? |
|---|---|---|
| `app:method-reproducibility` | Real-SE's six admission criteria and construction protocol — the basis for the headline benchmark being a benchmark and not a sample | **yes** |
| `app:extended_data` (6 calls) | the per-case detection table behind 17/18 and 5/17, the per-case baseline mappings, and `fig:tier-coverage` behind the "~8% → ~1.5%" claim in §4.4 | **yes** |
| `app:methodology` | provenance strength of the corpus (289 fixed-commit / 96 issue-PR-only / 7 summary-only) | **yes** |
| `app:diagnosis` | the quantitative diagnosis study behind "53% of cases localized" and "~6× compression" | **yes** |
| `app:pattern-catalog` | the agreement statistic quoted inline in §5.4 | **yes** |
| `app:algorithm` | funnel detail behind the 342 → 429/551/598 ablation, plus the FSM state diagram | partly — the numbers themselves are in the main text |
| `app:production-cases` | Case 1, referenced from a main-text figure caption | partly |
| `app:sql`, `app:io_examples` | full SQL specification, persisted Accept-gate record | navigational |

So the central evaluation table — the per-case detection results a reviewer would most
want to check — lives only in the supplement, as does the admission protocol that makes
Real-SE defensible. The supplement itself is 9 sections, ~3,200 words and 15 floats.

This interacts badly with item 1: the main paper is already 25 lines over, so folding any
of it in makes the overflow worse. The three ways out, in rough order of how much they
cost:

1. **Leave it as supplemental material and make the main text self-sufficient** — every
   claim above needs enough in the main text to stand alone, with the appendix as
   corroboration rather than as the evidence. Cheapest, and it is what the PVLDB
   supplemental-material rule actually contemplates.
2. **Promote the two critical pieces** (the per-case detection table, the admission
   criteria) into the 12 pages and cut elsewhere. Costs more than the current 25-line
   deficit.
3. **Leave it as is** and accept that a reviewer who does not open the supplement sees
   the headline detection rate with no per-case breakdown.

Rename matters too: calling it "Appendix" invites the reading that it is part of the
paper and therefore inside the page limit. "Supplementary material" describes what it is.

---

## Not paper edits, but worth knowing before submission

Four numbers in the paper have no backing data in the artifact
(`GAP_AUDIT.md` O3, O16, O17, and the `\NumFixedReplay` accounting):

- `fig:tier-coverage` (coverage 28→78%, overhead 1.5→7.5%) and the "~8% → ~1.5%"
  sentence in §4.4 — the data file is header-only.
- `fig:portability_matrix` cell values — the CSV that appears to back it was transcribed
  from `main_cn.tex`.
- `tab:db-baselines` 3/13 and 5/13 — stated in the paper as an analytical bound, with no
  executable harness.
- The clean-trace FP/1M figures (25.8, 83.3, 4.8×10⁵, 1.3×10⁵ over 504 K evaluations) —
  the six-case run is not in the artifact; the CSV pointing at it cites a file that does
  not exist.

Two presentation points, neither of which breaks a claim:

- **The amortization crossings are computed two different ways.** `fig:amortization` and
  §5.5 give K≈2630 for the naive collector — the exact crossing from the measured 192 s
  dump (recomputed: 2622) — but K≈380 and K≈760 for the optimised one, which are 2630
  divided by the rounded 7×. Recomputed from the measured 25 s dump they are **342 and
  683**. Nothing is wrong: at K=380 the real overhead is 9.0%, comfortably under the
  stated 10%. But "the optimized collector meets any budget at a ~7× shorter period"
  reads as an independent finding when it is in fact the input that produced 380.
- **`fig:catalog_generalization`'s "+19.7 pts at equal size"** is the catalog's coverage
  at the free-form arm's taxonomy size (67.5% at 15 templates) minus that arm's 47.8% —
  *not* the headline gap, which is 35.3 pts overall and 39.9 pts on observable records.
  Verified correct; worth making the basis explicit in the caption so a reviewer comparing
  it against Table numbers does not read a contradiction.

Everything else checks out: `python3 scripts/verify_paper_numbers.py` recomputes 29
published numbers and `python3 scripts/verify_figures.py` checks 12 numbers rendered
inside the figures against the data. All agree.
