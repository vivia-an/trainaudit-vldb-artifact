# Audit state (updated by the submission-readiness loop)

- **Target venue** PVLDB Vol. 20 / VLDB 2027 · single-blind · 12 pp. excluding references
- **Paper** `overleaf/sdc_llm_icml_2025` @ `main` `6f58c25` "Format paper for VLDB 2027 submission"
- **Existing public repo** `github.com/vivia-an/sdccheck-opensource` — mining skeleton only (52 files, 1.1 MB)
- **This artifact** assembled by `scripts/assemble_from_workspace.sh` (idempotent, re-runnable)

## Iteration log

### 2026-08-19 — iteration 1
- Identified the VLDB submission source: the `overleaf/` checkout is the VLDB
  reformat (Aug 15); the sibling `sdc_llm_icml_2025/` checkout is the older SIGMOD
  version but holds the 9.8 GB evaluation tree including `real_sdc/`.
- Fetched the authoritative PVLDB Vol. 20 requirements; recorded verbatim in
  `VLDB_SUBMISSION_CHECKLIST.md`.
- Built `paper/main.pdf` from scratch with the bundled TeX Live (added `hyperxmp`):
  0 errors, 0 undefined refs/citations, 14 pages — content spills onto p.13, so it
  is **over the 12-page limit**.
- Assembled the artifact: 1.1k files, 66 MB, covering 14 of the 14 identified gaps
  in the public repo (`GAP_AUDIT.md` G1–G14).
- Wrote `CLAIM_TO_ARTIFACT_MAP.md` covering all 17 main + 15 appendix floats and the
  `numbers.tex` macros.

## Open items, highest first

1. O1 — trim content to 12 pages (~12–15 lines).
2. O11/O12 — de-anonymise the artifact and set `\vldbavailabilityurl`.
3. O2 — decide the appendix's status now that appendices count toward the limit.
4. O4 — tidy overhead CSV + parser for `tab:overhead`.
5. O9 — publish the ~600 MB trace DBs as an archival release.
6. O3, O5, O6, O7, O8, O10 — see `GAP_AUDIT.md`.

### 2026-08-19 — iteration 2
- **O4 closed.** Found and fixed a provenance bug in the assembler: two overhead
  measurement sessions share log basenames (`Megatron-LM/` 2026-06-30 and
  `logs_gpu_lsk32/` 2026-07-12), so the July run was silently overwriting the June run
  the paper actually reports. Logs now live in `session_*/` subdirectories.
  Wrote `benchmark/injection/parse_overhead_logs.py`: every value in `tab:overhead`
  recomputes from the raw logs within 0.1% (732 ms, 191.9 s, 27.5 s, 25.0 s, 262×, 38×,
  34×, 7.7×), and the July session independently replicates it. Also labelled the
  unrelated gpt-tiny CPU overhead file that would otherwise mislead a reviewer.
- **O11 closed inside this artifact.** De-anonymisation is now a step of the assembler,
  not a manual edit, so re-running cannot reintroduce the double-blind framing.
- **O1 quantified.** The overflow is exactly 25 lines / 252 pt, not the ~12–15 first
  estimated; ~90 pt is recoverable from figure widths alone. Written up in `PAGE_BUDGET.md`.
- **O13 new.** Verified the upstream state of the production discovery
  (`Megatron-LM#4641` closed and maintainer-confirmed; PR #4642 closed in favour of a
  duplicate). Single-blind review permits naming it — see `UPSTREAM_BUG_EVIDENCE.md`.

## Blocked on a decision
Publishing this repository is outward-facing, so it is not done automatically:
`\vldbavailabilityurl` (O12) cannot be filled until the artifact has a public URL.

### 2026-08-19 — iteration 2 (continued): number-level verification
Wrote `scripts/verify_paper_numbers.py`, which recomputes the paper's numbers from the
shipped files rather than asserting them. **22 verified, 0 mismatched, 5 unbacked.**

Verified: Real-SE 17+1=18 from the manifest; TrainAudit 17/17; TrainCheck 5/17 with 1
tool failure; Naïve 0/17; funnel 420/5334/3436/357/45; the four-arm guard ablation
342 → 429 / 551 / 598 (551 reproduces exactly from the no-adversarial arm); and all
eight values of `tab:overhead`.

Three findings that only surface when the numbers are actually recomputed:
- **O14** three generations of detection data ship together; `results.csv` and
  `results_gpu.csv` score different case sets from the paper (9 and 6 cases shared out
  of 17), and `paper_table_baseline_traincheck.md` reports TrainCheck 10/17 against the
  paper's 5/17. The current set was prose-only, so it is now generated as
  `real_sdc/real_se_detection.csv` and the older files are labelled.
- **O15** the "0/17 vs 1/17" discrepancy carried in `experiment_registry.md` since May
  traces to one mislabelled row: `results.csv:35` has `phase=buggy` on what its message
  shows to be the fixed side of `OC-NEW-3`, firing the sqrt-decay rule. `OC-NEW-3` is
  not in the current set, so the reported 0/17 stands.
- **O16/O17** `fig:tier-coverage` has a header-only data file, and
  `paper_v2/portability.csv` was transcribed from `main_cn.tex` — it cannot verify the
  figure it appears to back.

### 2026-08-19 — iteration 2, published
Per the author's decision: a **new** repository, and **no edits to `main.tex`**.

- Published at <https://github.com/vivia-an/trainaudit-vldb-artifact> (public).
  Pre-publication scan found no secrets — every `api_key` in `core/agents/llm_config.yaml`
  is an `${ENV}` placeholder, and the only email addresses in the tree are the paper's own
  author addresses plus the ACM/VLDB template ones. 145 files contain workspace-absolute
  paths and one internal GPU hostname (`verl-lsk32-0`); these were left byte-identical
  because they are run records, and rewriting them would falsify the evidence.
  Note that `paper/` publishes the manuscript and its PDF; drop that directory if the
  paper should not be public before a decision.
- `\vldbavailabilityurl` (R6/O12) is a one-line edit left to the authors, recorded in the
  checklist.
- Paper-side findings written up in `PAPER_ACTIONS.md` instead of applied.
- Funnel-ablation arms verified: 114/400 = 28.5%, and `funnel_skip_l4_results.csv` holds
  both quoted arms in one file (`cohort=L4_kept` → 0/764, `cohort=L3_passed` → 0/6,922).
  **29 numbers verified, 0 mismatched.**

## Remaining open items
O1 (25-line overflow), O2 (appendix status), O3/O16/O17 + `\NumFixedReplay` (unbacked
numbers), O5/O6 (figure generators), O7 (temporal holdout), O9 (unshipped trace DBs),
O12 (availability URL), O13 (cite #4641), O15/O18 (mislabelled row, footnote case name).

### 2026-08-19 — iteration 3: figure-level verification
Most figure generators hard-code their values, so instead of writing generators that
would not reproduce the existing plots, `scripts/verify_figures.py` checks the other
direction: it lifts the numbers back out of the figure PDFs with `pdftotext` and compares
them to the data. **12 verified, 0 mismatched, 3 unbacked, 6 noted.**

Confirmed: the funnel stages in both funnel figures, skip-L3 114/400 = 28.5%, both
skip-L4 cohort denominators (764 and 6,922), all 13 pool counts in `bug_distribution`
(summing to 392), the catalog coverage curve, and the amortization overheads at K=1000.

Two presentation findings (added to `PAPER_ACTIONS.md` as O19/O20):
- The amortization crossings are derived two different ways — naive K≈2630 is the exact
  crossing from the measured 192 s dump, but optimised K≈380/760 are that value divided by
  the rounded 7×; from the measured 25 s they would be 342/683. No claim breaks (K=380 is
  really 9.0%), but the 7× reads as a result when it is the input.
- `fig_catalog_generalization`'s "+19.7 pts at equal size" verifies exactly — catalog at 15
  templates (67.5%) minus free-form frozen (47.8%) — and is *not* the headline gap (35.3 /
  39.9 pts), which a reviewer could easily misread.

### 2026-08-19 — iteration 4: trace databases published (O9)
DuckDB traces compress about 7×, so what the index described as an unshippable ~600 MB
is a **39.5 MiB** release asset once the run logs are left behind.

Published as [`trace-dbs-v1`](https://github.com/vivia-an/trainaudit-vldb-artifact/releases/tag/trace-dbs-v1):
129 `.db` files from 43 runs — 40 fault-injected Megatron-LM runs plus 6 clean baselines.
Verified that all 42 databases referenced by `experiments/guard_ablation/d1_results.csv`
are covered, including the four clean runs nested under `normal_db/` that a first pass
missed because they sit one directory deeper than the rest.

- `benchmark/injection/trace_db_manifest.csv` carries a SHA-256 per file.
- `scripts/fetch_trace_dbs.sh` downloads, extracts and verifies; `--verify-only` re-checks
  an existing copy.
- `core/ablation_scripts/run_d1_phase3.sh` hardcoded two workspace paths, so the assembler
  now rewrites them to `${SDCCHECK_ROOT:-...}` / `${MEGATRON:-...}`, keeping the original
  values as defaults. The 126-cell ablation is therefore re-runnable off this repository.

Excluded from the bundle: the `VLog/`, `TracePoint/` and `Collector/logs/` directories
(~400 MB) that sit beside the databases. They are run logs, not inputs to any reported
number, and remain listed in `trace_db_index.csv`.

### 2026-08-20 — iteration 5: two findings closed, one withdrawn
- **O7 unblocked.** The temporal holdout was recorded as blocked for missing dates; the
  dates were only unresolved. 385 of 392 records carry an issue URL or commit hash, and
  `resolve_record_dates.py` dated **374 (95%)** from upstream — 255 from commit dates, 119
  from issue/PR creation. The corpus spans **2021-01-26 to 2026-05-06**; a median cut at
  2025-03-14 gives 186/188 with all 13 categories on both sides (`temporal_split.json`).
- **O22 new.** That split partly confounds with framework: OLMo is 69 train / 5 test and
  OLMo-core 5 train / 63 test, because OLMo-core is the newer project. A temporal holdout
  here would partly measure framework transfer, which §5.4 already reports separately.
- **O18 withdrawn.** I had flagged the §5.2 footnote as contradicted by the data. It was
  not — I had read `results.csv`, which scores a superseded case set.
  `logs/smoke/M-020_smoke.log` records `Fixed: ASSERT fired (expected — bug prevented)`,
  and `extract_replay_outcomes.py` over the 17 current cases gives **17 detected, 16 clean,
  1 assertion-fired, 0 false positives** — the footnote and all three macros exactly as
  published. This also closes the `\NumFixedReplay` gap: those outcomes are now a CSV.
- **O21 sharpened.** The 25.8 / 83.3 FP/1M pair implies ≈13 and ≈42 violating evaluations
  over 504 K (ratio 3.2×, matching the figure's annotation). That is a row-level
  denominator, so it cannot be reconciled with the shipped leave-one-out data, whose five
  clean databases give 9 and 35 over 438 rule-level evaluations. The 504 K run needs its
  own file.

**33 numbers verified, 0 mismatched, 4 unbacked.**

## Lesson for later iterations
Two of my findings came from `results.csv` and both were wrong, because it scores a
superseded case set. Check `benchmark/eval/DETECTION_FILES_NOTE.md` before treating any
per-case file as authoritative.

### 2026-08-20 — iteration 6: the artifact could not actually run anything
Tried to reproduce the 25.8 / 83.3 FP/1M pair (O21) by running the verifier against the
now-published clean traces. It could not be done, for two reasons found in order.

- **O24.** `core/` held only the mining pipeline — the `sdccheck` verifier package and the
  four ablation constraint libraries were never assembled, so `python3 -m sdccheck` did not
  exist in the artifact. Fixed: `core/sdccheck/` (31 modules, 268 KB) and
  `core/config/ablation_libraries/` (6 libraries) are now copied by the assembler.
- **O23.** With those in place the verifier runs, connects to the trace (370,270 rows in
  `coredump(step, stage, data)`) and emits **93 report entries against `lib_full` — exactly
  the recorded cell's `total=93`** — but all 93 error. Cause: every one of the 249 rules in
  every library has an **empty `logic` field**, and the `--constraints-file` path
  synthesises SQL per constraint via `LLMSQLGenerator` → `SQLAgent`, an LLM call. Without a
  key the SQL is empty, `execute("")` returns `None`, and `.fetchdf()` fails.

That qualifies what I claimed last iteration. "§5.3 is re-runnable off this repository" was
wrong: the traces, libraries, harness and drivers are all present, but the SQL is not, so
the ablation needs an API key and is not bit-reproducible. Written up in `RERUN_LIMITS.md`
with a table of what does run offline (eight commands) versus what needs a key or GPUs.

It also raises a question for the authors rather than a defect: §4.5 describes the verifier
as compiling each constraint into parameterized SQL, with the 2–3 ms per query being DuckDB
planning. In the shipped code that translation is an LLM call. The two reconcile if the LLM
step is a one-off compilation whose output is then executed deterministically — but the
compiled SQL is precisely what the artifact lacks, so **shipping the generated SQL per rule
is the highest-value addition remaining.**

### 2026-08-20 — iteration 7: the compiled SQL was recoverable (O23 largely closed)
Last iteration's conclusion — that the deterministic SQL §4.5 describes is simply absent —
was half right. It is absent from the libraries, but **every generation call was logged**
during the runs that produced the paper's numbers.

`core/extract_generated_sql.py` recovers it: **228 constraints from 16,408 logged
`SQLAgent` interactions across 150 logs**, into `core/config/generated_sql.json` (0.9 MiB,
up to three variants per rule since generation is per run). Coverage is 151 of the
library's 242 named rules; the other 91 were never instantiated under the tested
topologies, consistent with each recorded cell evaluating 93 constraints rather than 249.
So every rule that contributed to a reported number is covered.

The recovered SQL has exactly the shape §4.5 claims — stage and name filters in `WHERE`,
`HAVING COUNT(DISTINCT cksum) > 1` as the violation condition — which turns that section
from described into inspectable.

`core/validate_generated_sql.py` then executes it against real traces:

| trace | empty | non-empty | error |
|---|---:|---:|---:|
| `normal_db/dp_normal` (clean) | 203 | 12 | 13 |
| `dp_normal_db` (clean) | 203 | 12 | 13 |
| `tp_router_test_db` (fault-injected) | 171 | **44** | 13 |

94% executes cleanly, and the injected fault yields 3.7× the violating rules of a clean
run. **O25**: the 12 non-empty on a clean trace are guard-excluded rules (TP-specific
checks on a `TP=1` run) executed without their guard — §4.4's SwitchMLP argument
reproduced incidentally, and useful as direct evidence that the guards carry weight.
The 13 errors are all `Referenced column "data" not found`: pipeline-parallel activation
rules needing trace fields above the S0 tier in these databases, i.e. schema-tier gated.

### 2026-08-20 — iteration 8: duplicate per-rank captures (O26), and a correction
Executing the recovered SQL against the published traces surfaced a data problem, and my
first diagnosis of it was wrong.

**What I first said:** the collector mislabelled the second DP rank in 4 clean traces.
**What is actually true:** the two per-rank files are *byte-identical* (same MD5, same
185,135 rows, identical row by row), so the second rank's capture is missing and a copy of
rank 0 stands in its place. And the scope is **7 of 43 runs**, not 4 — four clean baselines
plus `requires_grad_test_db`, `requires_grad_before_backward_test_db`, `shape_test_db`.

The impact is correspondingly wider than the 31 rank-counting queries I first named: every
cross-rank comparison on these traces compares rank 0 with itself and passes vacuously,
which is the dominant rule shape in the library.

Quantified against `d1_results.csv`: these seven databases contribute **147 of 1,369 false
positives (11%)** across the 126 cells, and **6 of the 9** clean-arm false positives for
`lib_full` — only `tp_normal`'s 3 come from a database with two genuine ranks. The
direction is against the paper (real rank-1 data could only add fires), and detection
results are untouched since they come from the Real-SE method-level replays.

I checked whether the live collector explains it: `initialize()` builds the filename from
the same `ranks_info_` dict that `param_info.update(cls.ranks_info_)` stamps into each
row, so a genuine two-rank run cannot produce this. The duplication happened outside the
collector — a copied file, or a second rank that never wrote.

### 2026-08-20 — iteration 9: the rank finding, settled properly
My two previous diagnoses were both wrong, for the same underlying reason: **hashing a
DuckDB `.db` file is not a sound content check.** DuckDB keeps uncommitted data in a
`.db.wal` sidecar, so for 7 of the 43 runs the `.db` is a 12 KB header — and all such
headers are byte-identical, which is what produced the phantom "duplicates" I reported as a
collector labelling bug and then as seven duplicated runs.

Settled with an order-independent aggregate over every row:

- **O26 (real).** Four clean baselines have rank 1 identical to rank 0 in content:
  `normal_db/{dp_normal,dist_optimizer_normal,mixed_precision_normal}` and `dp_normal_db`.
  Cross-rank rules pass vacuously there, and **67–80% of clean-arm false positives** rest on
  them (6/9 `lib_full`, 9/13 without π_precond, 28/35 without π_topo). Only `tp_normal` has
  two genuine ranks. Across all 126 cells it is 43/1,369 (3%) — the aggregate is dominated
  by fault-injected databases.
- **The three fault-injected runs I flagged are clear.** `requires_grad_test_db`,
  `requires_grad_before_backward_test_db` and `shape_test_db` keep their data in WALs of
  differing sizes; their ranks differ.
- **O27 (new).** `dp_normal_db` and `normal_db/dist_optimizer_normal` are the *same trace*
  (`18229edd5147`), so the ablation's 42 databases are 41. They recorded different results
  from identical input — 9 false positives against 4 without π_topo — which is the clearest
  evidence in the artifact of the nondeterminism O23 implies.
- **O28 (my defect).** `trace-dbs-v1`, which I published, omitted the sidecars, so per-rank
  files for those 7 runs shipped as empty headers. Fixed in `trace-dbs-v2` (139 files, 10
  sidecars, 40.6 MiB) via a proper `scripts/build_trace_bundle.py`. The ablation path was
  never affected — merged databases are all self-contained.

`benchmark/injection/audit_rank_captures.py` now performs this check, so it does not depend
on me remembering the WAL caveat.
