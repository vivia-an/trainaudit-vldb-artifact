# PVLDB Vol. 20 (VLDB 2027) submission checklist — TrainAudit

Authoritative source: <https://vldb.org/pvldb/volumes/20/submission> (fetched 2026-08-19).
Verbatim requirements are quoted; everything else is our status.

Paper under audit: `overleaf/sdc_llm_icml_2025/main.tex` @ `main` = commit
`6f58c25 "Format paper for VLDB 2027 submission"`.
Built here as `paper/main.pdf` (clean build, 0 errors, 0 undefined refs/citations).

| # | Requirement | Verbatim / source | Status | Action |
|---|---|---|---|---|
| R1 | Page limit — regular research paper | "up to **12 pages** excluding references" | **OVER by 25 lines** (252 pt ≈ 0.43 column) — measured, see `PAGE_BUDGET.md` | trim 25 lines |
| R2 | Appendices count toward the limit | "All content, **including any appendices and acknowledgements** but excluding the references, must fit on the given number of pages." | **RISK** — `appendix.tex` is compiled as a separate supplementary PDF and the main text makes 20+ `Appendix~\ref{}` forward references to it | decide: fold essentials into 12 pages, or keep supplement but make the main paper self-contained |
| R3 | Single-blind, names on page 1 | "VLDB is a single-blind conference. Therefore, authors MUST include their names and affiliations on the first page of the manuscript." | **OK** — 7 authors + affiliations at `main.tex:126–183` | — |
| R4 | Supplemental material is mandatory | "Authors **must submit supplemental material**, such as code, data, and other implementation artifacts used to produce the results reported in the paper." | **THIS REPO** | publish, then record URL |
| R5 | Public archival repository + URL at submission | "Authors should place the supplemental material in a **publicly accessible archival repository** and provide a URL during the submission process." | **PENDING** — `github.com/vivia-an/sdccheck-opensource` today holds only the 1.1 MB mining skeleton (52 files) | replace with this artifact (66 MB, 1.1k files) |
| R6 | `\vldbavailabilityurl` | PVLDB template macro; renders the availability footnote | **EMPTY** — `main.tex:16` `\renewcommand\vldbavailabilityurl{}` | one-line edit: `\renewcommand\vldbavailabilityurl{https://github.com/vivia-an/trainaudit-vldb-artifact}` (left to the authors — `main.tex` is unmodified by request) |
| R7 | PVLDB template block intact | `pvldb.sty` + `\vldbdoi`/`\vldbpages` placeholders | **OK** — `main.tex:5–14`, placeholders `XX.XX/XXX.XX` / `XXX-XXX` are correct for review | — |
| R8 | CMT registration, COI declarations, reviewer nomination, concurrent-submission disclosure | per submission page | **AUTHOR TASK** — outside this artifact | authors |
| R9 | No leftover placeholders in the PDF | — | **OK** — no `XXX`/`URL_TO_YOUR_ARTIFACTS`/`example.com` outside the PVLDB DOI/page placeholders | — |
| R10 | Figure accessibility (`\Description`) | acmart requirement | **OK** — every `includegraphics`/table float in `main.tex` carries `\Description` | — |

## Single-blind consequence for the artifact

The current public repo is built for **double-blind** review: `ANONYMOUS_RELEASE.md`
withholds author identity and the canonical URL, and `README.md` opens with
"Anonymous artifact (double-blind review)". VLDB is single-blind and the paper
carries author names, so anonymisation buys nothing and actively hurts — a
reviewer who cannot tell whether the repo belongs to the authors cannot credit it.

Action: de-anonymise. Replace `ANONYMOUS_RELEASE.md` with a normal
`AUTHORS`/availability statement, restore author identity in `CITATION.cff`, and
name the canonical repository in the paper via `\vldbavailabilityurl`.
