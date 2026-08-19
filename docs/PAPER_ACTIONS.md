# Paper-side actions (report only — `main.tex` has not been modified)

Three items in `overleaf/sdc_llm_icml_2025/main.tex` need an author decision before
submission. Each gives the exact location, the evidence, and a suggested edit.

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

## 3. The §5.2 footnote names a case the data does not support

`main.tex:840-843`:

> Sixteen fixed sides complete cleanly; M-020's upstream fix instead rejects the faulty
> configuration as intended. We count this as a fixed-side verdict, not as a completed
> CLEAN replay.

In the shipped data **M-020's fixed side is `CLEAN`** (`benchmark/eval/results.csv`:
`M-020,fixed,CLEAN,[fixed] T1-layer-count-strict did not fire`). The case that has no
clean fixed row there is `OC-NEW-3` — and `OC-NEW-3` is not in the current Real-SE set at
all, so that file cannot settle it either.

The footnote may still be correct for the current set; the point is that nothing in the
artifact confirms which case it refers to. Suggested fix: name the case as it appears in
`benchmark/eval/real_sdc/real_se_detection.csv`, and record its fixed-side outcome there
so the footnote becomes checkable. The `0/\NumFixedReplay` claim itself is not in
question — see `../benchmark/eval/DETECTION_FILES_NOTE.md`.

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
