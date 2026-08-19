# TrainAudit — PVLDB Vol. 20 supplemental material

Code, data, and measurement records for **"TrainAudit: Silent Error Detection for
LLM Training via Verified Guarded Relational Constraints"**, submitted to PVLDB
Volume 20 (VLDB 2027).

PVLDB requires supplemental material for the results reported in the paper. This
repository is that material, published at
<https://github.com/vivia-an/trainaudit-vldb-artifact>: every numbered table and figure in the paper is traced
to a file here in [`docs/CLAIM_TO_ARTIFACT_MAP.md`](docs/CLAIM_TO_ARTIFACT_MAP.md).

## Layout

```
paper/            paper sources, PVLDB style files, built PDF (14 pp.) + supplementary appendix
core/             verified constraint mining: Pattern Catalog (T01-T35) -> multi-agent FSM
                  -> Accept gate (counterexample /\ confirmation) -> healthy-run replay -> SQL verifier
collector/        VTimeline: the training instrumentation that writes the DuckDB trace
benchmark/eval/   evaluation tree at the paths the paper cites (real_sdc/, template_induction/,
                  catalog_generalization/, traincheck_surrogates/, gpu_logs/, paper_v2/, v2_full/)
benchmark/injection/  fault-injection + overhead launch scripts, raw H20 collector logs, trace-DB index
experiments/      leave-one-out guard ablation (42 db x 3 lib = 126 cells), holdout mining transcripts
baselines/        baseline logs; TrainCheck itself is upstream, see TRAINCHECK_UPSTREAM.txt
figures/          the figures the paper includes, plus the generator scripts that exist
docs/             submission checklist, claim map, gap audit, data availability
scripts/          assemble_from_workspace.sh — regenerates this tree from the research workspace
```

## Quick checks (no GPU, no API key)

```bash
pip install -r core/requirements.txt

python3 core/run_smoke.py                          # offline pipeline smoke
python3 core/scripts/reproduce_funnel_counts.py    # funnel 420 -> 5334 -> 3436 -> 357 -> 45
sha256sum -c core/config/frozen_template_catalog.sha256
```

## Mining with the multi-agent pipeline

```bash
export SDC_PAPER_ALIGN=1        # paper path: catalog templates + FSM + Accept gate
export DEEPSEEK_API_KEY=...     # required for the LLM rounds
python3 core/run_miner.py
```

`SDC_PAPER_ALIGN=0` is the control arm (open-ended mining, no catalog templates) —
this is the contrast reported in §5.3.

## Rebuilding the paper

```bash
cd paper && pdflatex -shell-escape main.tex && bibtex main \
  && pdflatex -shell-escape main.tex && pdflatex -shell-escape main.tex
```

Requires `hyperxmp` and `minted` (`-shell-escape` is mandatory).

## Re-running the guard ablation on the real traces

The trace databases are published as a release asset — 129 DuckDB files from 43 runs,
39.5 MiB packed, covering every database the 126-cell ablation reads:

```bash
bash scripts/fetch_trace_dbs.sh --dest /path/for/traces
export MEGATRON=/path/for/traces SDCCHECK_ROOT=$PWD
bash core/ablation_scripts/run_d1_phase3.sh
```

Full replays of the Real-SE cases additionally need GPUs and the framework checkouts at
the commits named in the manifest; see
[`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

## Verifying the paper's numbers

```bash
python3 scripts/verify_paper_numbers.py            # 29 published numbers, recomputed
python3 scripts/verify_figures.py                 # numbers printed inside the figures
python3 benchmark/injection/parse_overhead_logs.py --check   # Table tab:overhead
```

`verify_paper_numbers.py` recomputes rather than restates: Real-SE 17+1 from the
manifest, TrainAudit 17/17, TrainCheck 5/17, Naïve 0/17, the mining funnel
420→5334→3436→357→45, the funnel-ablation arms (114/400, 0/764, 0/6,922), the
four-arm guard ablation 342→429/551/598, and every value of `tab:overhead`. It also
reports the claims the shipped data does *not* support, and flags two files that score
superseded case sets — see `benchmark/eval/DETECTION_FILES_NOTE.md`.

`verify_figures.py` covers the other direction. Most figure generators hard-code their
values, so a plot can drift from its data silently; this lifts the numbers back out of the
figure PDFs and compares them. It confirms the funnel stages, the skip-L3/L4 denominators,
the 392-record pool counts, the catalog coverage curve and its equal-size annotation, and
recomputes the amortization crossings from the measured dump costs.

## Known limits

[`docs/GAP_AUDIT.md`](docs/GAP_AUDIT.md) lists what this artifact does and does not
support — including which paper numbers are analytical bounds rather than fresh
measurements, and which figures have no generator script.
