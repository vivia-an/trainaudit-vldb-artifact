# Ready-to-apply patches

`main.tex` is deliberately left unmodified. These are the edits the audit recommends,
as unified diffs against `overleaf/sdc_llm_icml_2025/main.tex`, so applying one is a
single command and reverting it is `git checkout`.

```bash
cd /volume/posttrain/users/lsk/sdc/lsk/overleaf/sdc_llm_icml_2025
git apply --check ../../trainaudit-vldb-artifact/docs/patches/01-availability-url.patch  # dry run
git apply         ../../trainaudit-vldb-artifact/docs/patches/01-availability-url.patch
```

## Measured page cost — read before applying

Built each patch in an identical environment and measured the overflow past page 12
(`REFERENCES` position on page 13, `pdftotext -bbox`):

| Applied | content lines spilling onto p.13 | vs baseline |
|---|---:|---:|
| nothing (baseline) | 25 | — |
| `02` only | 25 | **no cost** |
| `01` only | **44** | **+19 lines** |
| `01`+`02`+`03` | 44 | +19 lines, and the supplement grows 10 → 11 pages |

**Patch 01 costs 19 lines of page budget.** Rendering `\vldbavailabilityurl` puts a block on
page 1, and once page 1 loses room the displacement cascades through the floats. The paper is
already 25 lines over the 12-page limit (O1), so applying 01 as-is takes the deficit to 44
lines — roughly three quarters of a column.

That does not argue against applying it: PVLDB wants the availability URL, and O1 has to be
solved regardless. It does mean **01 should be applied as part of the page-trim work, not
before it**, and the trim needs to recover ~44 lines rather than ~25. `02` is free. `03`
affects only the supplement.

| Patch | What it does | Why |
|---|---|---|
| `01-availability-url.patch` | fills `\vldbavailabilityurl` | PVLDB requires the supplemental URL; R5/R6 of the checklist |
| `02-cite-upstream-issue.patch` | names `NVIDIA/Megatron-LM#4641` | single-blind permits it, and it turns the flagship production claim from unverifiable into checkable |
| `03-appendix-vldb-format.patch` | reformats `appendix.tex` for PVLDB | the supplement was never reformatted for the venue: it renders "Anonymous Author(s)", declares `SIGMOD/PODS '27` as its conference, and cites the "SIGMOD/PACMMOD appendix policy". Five front-matter changes; GAP_AUDIT O29 |

Deliberately **not** provided as patches, because they are judgement calls rather than
mechanical edits:

- **The 25-line page overflow.** Which content to cut is the authors' call.
  `PAGE_BUDGET.md` gives the measured deficit and the figure-width levers that recover
  about 90 pt of it without touching a word.
- **The appendix's status** now that PVLDB counts appendices inside the 12 pages.
  `PAPER_ACTIONS.md` §4 lists which of the 18 `Appendix~\ref` calls carry evidence for a
  main-text claim, and the three ways out with their costs.
- **Patch 03 grows the supplement from 10 to 11 pages**, because the author block takes
  space. PVLDB counts appendices inside the 12-page limit (O2), so this interacts with the
  overflow in item 1 — decide them together. Verified by building the patched file: seven
  authors on page 1 and zero occurrences of "anonymous", "SIGMOD/PODS" or "PACMMOD"
  anywhere in the PDF.
- **`%% Acknowledgements removed for double-blind review`** at `main.tex:1159`.
  Single-blind permits acknowledgements, but they count toward the same 12 pages, so this
  interacts with the overflow.
- **How to present the trace-data caveats** (`TRACE_DATA_CAVEATS.md`): whether to re-capture
  the four affected clean runs and re-run the clean ablation arm, or to state the current
  numbers as a bound.
