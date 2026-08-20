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
