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
core/sdccheck/    the runnable verifier: python3 -m sdccheck <trace.db> --constraints-file ...
core/config/      frozen catalog, the four guard-ablation libraries, and generated_sql.json
                  (the compiled SQL, recovered from the recorded runs)
collector/        VTimeline: the training instrumentation that writes the DuckDB trace
benchmark/eval/   evaluation tree at the paths the paper cites (real_sdc/, template_induction/,
                  catalog_generalization/, traincheck_surrogates/, gpu_logs/, paper_v2/, v2_full/)
benchmark/injection/  fault-injection + overhead launch scripts, raw H20 collector logs
                  (two sessions, kept separate), overhead_h20.csv, and the trace-DB manifest
experiments/      leave-one-out guard ablation (42 db x 3 lib = 126 cells), holdout mining transcripts
baselines/        baseline logs; TrainCheck itself is upstream, see TRAINCHECK_UPSTREAM.txt
figures/          the figures the paper includes, plus the generator scripts that exist
docs/             submission checklist, claim map, gap audit, page budget, re-run limits,
                  trace-data caveats, and patches/ for the recommended paper edits
scripts/          check_all.sh (every offline check), fetch_trace_dbs.sh, build_trace_bundle.py,
                  and assemble_from_workspace.sh — regenerates this tree from the workspace
```

## Quick checks (no GPU, no API key)

```bash
pip install duckdb            # the only third-party package the checks need

bash scripts/check_all.sh                          # all nine offline check groups
python3 core/run_smoke.py                          # offline pipeline smoke
python3 core/scripts/reproduce_funnel_counts.py    # funnel 420 -> 5334 -> 3436 -> 357 -> 45
(cd core/config && sha256sum -c frozen_template_catalog.sha256)   # names a bare filename
```

## Mining with the multi-agent pipeline

This is the only part that needs more than `duckdb` — it pulls in the AutoGen stack:

```bash
pip install -r core/requirements-mining.txt
```

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
39.5 MiB packed, covering every database the 126-cell ablation reads. Note that the
ablation also needs a DeepSeek API key, because the shipped constraint libraries carry
rule specifications rather than SQL — see [`docs/RERUN_LIMITS.md`](docs/RERUN_LIMITS.md):

```bash
bash scripts/fetch_trace_dbs.sh --dest /path/for/traces
export MEGATRON=/path/for/traces SDCCHECK_ROOT=$PWD
bash core/ablation_scripts/run_d1_phase3.sh
```

Full replays of the Real-SE cases additionally need GPUs and the framework checkouts at
the commits named in the manifest; see
[`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

## Verifying the paper's numbers

Everything offline, in one command:

```bash
bash scripts/check_all.sh                            # 9 offline check groups
bash scripts/check_all.sh --traces DIR --events DIR2 # + the trace-dependent checks
```

The two trace bundles are separate releases:

```bash
bash scripts/fetch_trace_dbs.sh --dest DIR                        # coredump schema
TAG=trace-events-v2 bash scripts/fetch_trace_dbs.sh --dest DIR2   # events schema
```

`coredump(step, stage, data)` is what the Megatron collector writes and what the guard
ablation reads. `events(…, hookpoint, payload)` is the schema §4.4 documents; its
`build.snapshot` payloads carry the per-parameter `group_size`/`all_equal` records the
topology guard reasons over — see `docs/SCHEMA_AND_GUARD_MECHANISM.md`.

Individually:

```bash
python3 scripts/verify_paper_numbers.py            # 33 published numbers, recomputed
python3 scripts/verify_figures.py                  # numbers printed inside the figures
python3 benchmark/injection/parse_overhead_logs.py --check   # Table tab:overhead
python3 benchmark/injection/audit_rank_captures.py --root <traces>  # per-rank captures
```

`verify_paper_numbers.py` recomputes rather than restates: Real-SE 17+1 from the
manifest, TrainAudit 17/17, TrainCheck 5/17, Naïve 0/17, fixed-side 0/17, the mining funnel
420→5334→3436→357→45, the funnel-ablation arms (114/400, 0/764, 0/6,922), the
four-arm guard ablation 342→429/551/598, the two measured DB-baseline false-positive
rates, and every value of `tab:overhead`. It also
reports the claims the shipped data does *not* support, and flags two files that score
superseded case sets — see `benchmark/eval/DETECTION_FILES_NOTE.md`.

`verify_figures.py` covers the other direction. Most figure generators hard-code their
values, so a plot can drift from its data silently; this lifts the numbers back out of the
figure PDFs and compares them. It confirms the funnel stages, the skip-L3/L4 denominators,
the 392-record pool counts, the catalog coverage curve and its equal-size annotation, and
recomputes the amortization crossings from the measured dump costs.

## The compiled SQL

The constraint libraries carry rule specifications, not SQL — the translation is an LLM
call at check time. The SQL that actually ran during the recorded experiments has been
recovered from the interaction logs into `core/config/generated_sql.json` (228
constraints), and it can be executed directly:

```bash
python3 core/validate_generated_sql.py --db <trace>/Collector/merged_coredump.db
```

94% executes without error. On a clean trace 203 rules return empty results; on a
fault-injected trace 44 return violation sets. See
[`core/config/GENERATED_SQL.md`](core/config/GENERATED_SQL.md).

## Known limits

[`docs/GAP_AUDIT.md`](docs/GAP_AUDIT.md) lists what this artifact does and does not
support — including which paper numbers are analytical bounds rather than fresh
measurements, and which figures have no generator script.
