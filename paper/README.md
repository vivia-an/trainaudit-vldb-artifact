# Paper sources

| File | Notes |
|---|---|
| `main.tex` | the VLDB submission source, `overleaf/sdc_llm_icml_2025` @ `6f58c25` |
| `main.pdf` | built here with TeX Live 2026: 0 errors, 0 undefined references or citations, **14 pages** (content to p.13, references p.13–14) |
| `appendix.tex` | the supplementary-material source that accompanies `main.tex` |
| `appendix_supplement.pdf` | built here from `appendix.tex`: **10 pages**, 0 undefined references |
| `numbers.tex` | headline counts, the single source of truth shared by both documents |
| `main.bib`, `pvldb.sty`, `acmart.cls`, `ACM-Reference-Format.bst` | bibliography and style |

## Two things to know about these builds

**The appendix still targets the wrong venue.** `appendix.tex:1–3` reads

```
%%%%%%%% SIGMOD / PACMMOD DOUBLE-BLIND SUBMISSION — SUPPLEMENTARY MATERIAL %
\documentclass[sigconf,anonymous]{acmart}
```

so it renders "Anonymous Author(s)" and its abstract cites the "SIGMOD/PACMMOD appendix
policy", while `main.tex` was reformatted for single-blind PVLDB and names all seven
authors. See `../docs/GAP_AUDIT.md` O29.

**Code listings are unhighlighted in `appendix_supplement.pdf`.** `minted` v3 wants its
`latexminted` helper reachable from pdflatex's shell-escape subprocess, which it was not
in this container. The document is otherwise complete — 0 undefined references — but
rebuild it in your own environment for the final version:

```bash
pdflatex -shell-escape appendix.tex && bibtex appendix \
  && pdflatex -shell-escape appendix.tex && pdflatex -shell-escape appendix.tex
```

The previous `appendix_supplement.pdf` shipped here was worse than unhighlighted: it was a
2026-07-17 build of a *different* checkout's `appendix.tex` (27 pages, different content
hash), so it did not correspond to the source beside it.
