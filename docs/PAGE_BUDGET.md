# Page budget — measured, not estimated

PVLDB Vol. 20 allows **12 pages excluding references**, and "All content, including any
appendices and acknowledgements but excluding the references, must fit on the given
number of pages."

## Measurement

Built from `paper/main.tex` with the bundled TeX Live (0 errors, 0 undefined
references or citations). Positions read from `pdftotext -bbox`:

| Quantity | Value |
|---|---|
| Total pages | 14 |
| Content ends | page 13, column 1, y ≈ 358 pt |
| `REFERENCES` heading | page 13, column 1, y = 364.9 pt |
| Content lines on page 13 | **25** (tail of §6 Related Work + all of §7 Conclusion) |
| Vertical extent of that overflow | 252 pt, at an 11.25 pt line pitch |
| References occupy | page 13 col. 1 (from y 365) through page 14 |

**Overflow: 25 lines ≈ 0.43 of one column.** Content must end at the bottom of page 12.

**And the availability URL costs another 19.** Applying
`docs/patches/01-availability-url.patch` — which PVLDB effectively requires — renders a
block on page 1 whose displacement cascades through the floats, taking the spill from 25
lines to **44**. Measured by building both in an identical environment; `02` costs nothing.
So plan the trim for ~44 lines, and apply `01` as part of that work rather than before it.

## Why the obvious cuts do not work

§6 Related Work is 323 words (~31 lines) and §7 Conclusion is 92 words (~9 lines).
Deleting 25 lines from those two would remove most of both. The room has to come from
the long sections or from float geometry:

| Section | Words | ~lines |
|---|---:|---:|
| §2 Background and Motivation | 1019 | 97 |
| §3 Empirical Study and Problem Formulation | 1837 | 175 |
| §4 TrainAudit: Mining and Checking | 2311 | 220 |
| §5 Evaluation | 2409 | 229 |
| §6 Related Work | 323 | 31 |
| §7 Conclusion | 92 | 9 |

## Mechanical levers (no prose changes)

`main.tex` has 14 floats. Figure widths today:

| Location | Figure | Width |
|---|---|---|
| `main.tex:248` | `figure1_vector_v5` | 0.78 `\columnwidth` |
| `main.tex:303` | `fig_silent_motivation` | `\columnwidth` |
| `main.tex:568` | `fig_overview_v32` | 0.70 `\linewidth` (wide float) |
| `main.tex:641` | `fig_three_predicate_sql` | 0.72 `\textwidth` (wide float) |
| `main.tex:851` | `fig_predicate_ablation_v2` | `\linewidth` |
| `main.tex:856` | `fig_funnel_ablation_v2` | `\linewidth` |
| `main.tex:950` | `fig_catalog_generalization` | `\columnwidth` |
| `main.tex:1023` | `fig_amortization` | `\columnwidth` |
| `main.tex:1064` | `fig_portability_matrix` | 0.78 `\columnwidth` |

A 10% width cut on a `\columnwidth` figure returns roughly 10% of its height — for a
square-ish plot that is 20–25 pt each. Trimming the four full-width single-column
plots by 10% recovers on the order of 90 pt, about a third of the deficit, with no text
touched. `\textfloatsep`/`\dbltextfloatsep` are already tightened to 13 pt in the
preamble, so there is little left there.

The remaining ~160 pt has to come from prose or from dropping a float. That is an
authors' call, not a mechanical one — this file only states the size of the problem.

## Related decision

`main.tex:1159` still reads `%% Acknowledgements removed for double-blind review`.
VLDB is single-blind, so acknowledgements are permitted again — but they would count
toward the same 12 pages. Restoring them makes the deficit larger, so decide the two
together.
