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
