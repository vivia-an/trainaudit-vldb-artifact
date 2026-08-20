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

## Which levers actually work — measured, not estimated

An earlier version of this file claimed that "trimming the four full-width single-column
plots by 10% recovers on the order of 90 pt, about a third of the deficit". **That was
arithmetic on figure heights, and it is wrong.** Measured, figure shrinking recovers
nothing.

Every row below was built in one environment, with `patch 01` applied, and the overflow
located by finding `REFERENCES` on page 13 with `pdftotext -bbox`:

| Change | Content lines spilling past p.12 | Total pages |
|---|---:|---:|
| baseline, no patches | 25 | 14 |
| `patch 01` (availability URL) | 44 | 14 |
| `patch 01` + six figures scaled to 90% of their own width | **44 — no change** | 14 |
| `patch 01` + ~139 words of prose removed (~13 lines) | 17 | 13 |
| `patch 01` + ~269 words of prose removed (~26 lines) | **0 — content ends on p.12** | 13 |

Two things to take from this.

**Figure width is not a lever.** Tested twice: once with a regex that mistakenly enlarged
the two plots living inside a 0.44-column `minipage`, then correctly, scaling six figures to
90% of whatever unit each already used. Both gave 44. The likely reason is `\flushbottom`,
which acmart sets: columns are stretched to full height regardless, so space freed inside a
float is absorbed by inter-paragraph glue rather than pulling text back — and float
placement is discrete, so a slightly shorter figure still occupies the same top-of-page
slot.

**Prose has roughly 2:1 leverage.** Removing 13 rendered lines cut the overflow by 27, and
26 lines cleared it entirely. Shorter text lets a float move up a page, and that cascades.

So the target for O1 is **about 26 lines — some 270 words, two paragraphs — of prose**, with
`patch 01` included. Not 44 lines, and not geometry.

The paragraphs used to calibrate this were the four longest in §5.1 Experimental Setup,
chosen because they are descriptive rather than result-bearing. **That was a measurement, not
a recommendation** — which text to cut is an authors' judgement.

## Related decision## Related decision

`main.tex:1159` still reads `%% Acknowledgements removed for double-blind review`.
VLDB is single-blind, so acknowledgements are permitted again — but they would count
toward the same 12 pages. Restoring them makes the deficit larger, so decide the two
together.
