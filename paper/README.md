# Paper sources

This directory contains the public six-author manuscript snapshot from
`merlintang/sdc_llm_icml_2025`. The main-paper snapshot originates from commit
`085b8bd`; the supplementary source, figures, and PDF are synchronized with
the same canonical paper snapshot.

| File | Notes |
|---|---|
| `main.tex` | PVLDB Regular Research Paper source; six named authors |
| `main.pdf` | current build: 13 pages total; manuscript content ends on page 12, zero undefined references/citations |
| `appendix.tex` | matching six-author supplementary-material source |
| `appendix_supplement.pdf` | current build: 11 pages, zero undefined references/citations |
| `numbers.tex` | headline-count macros shared by both documents |
| `main.bib`, `pvldb.sty`, `acmart.cls`, `ACM-Reference-Format.bst` | bibliography and venue style |

The main and supplement author blocks are intentionally identical: Qingsong Cai, Shikai
Li, Zhengmao Ye, Mingjie Tang, Ran Tao, and Bryan Dai.  No anonymous or seventh-author
variant is part of the canonical snapshot.

## Build

Use the checked build entry point:

```bash
./build.sh
./build.sh appendix
```

The ACM template needs scalable fonts.  On Debian/Ubuntu, a working installation is:

```bash
apt-get install texlive-fonts-extra texlive-latex-extra cm-super
```

or with `tlmgr`:

```bash
tlmgr install libertine newtx cm-super
```

`kpsewhich libertine.sty` should return a path. Without these fonts, acmart's
microtype expansion can fail. The prebuilt PDFs remain available for reviewers
who do not rebuild.

Both sources carry `\Description` metadata for every graphic/table float. The
supplement uses the PVLDB single-blind metadata and is compiled from the source
beside it.
