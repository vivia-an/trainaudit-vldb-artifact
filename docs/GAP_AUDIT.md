# Gap audit — what the paper needs vs what the public artifact provides

Compared: `paper/main.tex` + `paper/appendix.tex` (VLDB build) against
`github.com/vivia-an/sdccheck-opensource` @ HEAD (cloned 2026-08-19: 52 files, 1.1 MB).

## Verdict

The published repo covers exactly one of the paper's five evaluation questions —
the verified-mining pipeline and its funnel counts. It contains **no benchmark, no
detection results, no baselines, no overhead measurement, and no collector**, so a
reviewer cannot check any headline number from it. PVLDB requires supplemental
material for "the results reported in the paper", so this is a submission blocker,
not a nice-to-have.

## What the published repo has

`run_miner.py`, `run_smoke.py`, `agents/` (FSM, Accept gate, prompts),
`config/frozen_template_catalog.json` (+ sha256), `data/funnel_counts.csv`,
`data/funnel_skip_l3_results.csv`, `scripts/reproduce_funnel_counts.py`,
`data/template_induction/`.

## What it is missing (all now present in this repository)

| # | Missing from public repo | Paper claim it backs | Now at |
|---|---|---|---|
| G1 | Real-SE benchmark manifest and admission record | §5.1 construction, §5.2 17/18, `tab:realse-class-coverage` | `benchmark/eval/real_sdc/` (32 files) |
| G2 | 392-record evidence corpus + taxonomy + IRR | §3 `tab:taxonomy`, appendix `tab:silent-evidence`, `tab:irr` | `benchmark/eval/silent_evidence_392.json`, `v2_full/`, `irr_50_*` |
| G3 | Per-case detection results and replay logs | `tab:summary_3way`, appendix `tab:detection-results` | `benchmark/eval/results.csv`, `results_gpu.csv`, `gpu_logs/` |
| G4 | Baseline detectors (TrainCheck, Naïve) + surrogates | §5.1 baselines, `tab:summary_3way` | `benchmark/eval/baseline_{traincheck,naive}*`, `traincheck_surrogates/` (122 files) |
| G5 | Fault-injection drivers | §5 replay protocol, injected-fault evaluation | `benchmark/injection/launch_scripts/` (51 scripts) |
| G6 | Training instrumentation ("vtime" collector) | §4.4 `tab:trace-schema`, §5.5 overhead | `collector/vtimeline/` |
| G7 | Overhead measurement evidence | `tab:overhead`, `fig:amortization` | `benchmark/injection/overhead_raw/` (14 H20 logs) |
| G8 | Leave-one-out guard ablation (42 db × 3 lib = 126 cells) | §5.3 `fig:predicate-ablation`, `fig:guard-ablations` | `experiments/guard_ablation/d1_results.csv` |
| G9 | L4 skip / adversarial-skip funnel arms | `fig:funnel-ablation`, `tab:funnel-dual-axis` | `benchmark/eval/funnel_skip_l4_results.csv`, `catalog_ablation/` |
| G10 | Catalog generalization / held-out coverage | §5.4 `fig:catalog_generalization` | `benchmark/eval/catalog_generalization/` |
| G11 | Portability data | §5.6 `fig:portability_matrix` | `benchmark/eval/paper_v2/portability.csv`, `p9_p16_deployment/` |
| G12 | Diagnosis-accuracy study (two raters + rubric) | appendix `fig:diagnosis-mechanism` | `benchmark/eval/diagnosis_s3_*` |
| G13 | Figure PDFs and generator scripts | all 15 floats with graphics | `figures/` + `figures/generators/` |
| G14 | Paper sources and a reproducible build | reviewer cross-checking | `paper/`, builds clean at 14 pp. |

## Gaps that remain open

| # | Gap | Why it matters | Recommended action |
|---|---|---|---|
| O1 | **Content exceeds 12 pages by a measured 25 lines** (252 pt ≈ 0.43 column): the tail of §6 and all of §7 sit on p.13 before `REFERENCES` at y=364.9 pt. PVLDB: appendices included, references excluded. | desk-reject risk | see `PAGE_BUDGET.md` — ~90 pt is recoverable by shrinking four figures 10%, the rest needs prose or a dropped float |
| O2 | **The appendix is a separate PDF but PVLDB counts appendices toward the 12 pages.** The main text makes 20+ `Appendix~\ref{}` calls into it, so the main paper is not self-contained. | reviewers are not obliged to read supplemental material; any claim that lives only in the appendix reads as unsupported | make every main-text claim stand on its own, and re-label the appendix as supplemental material rather than "Appendix" |
| O3 | **Narrower than first recorded.** `tab:db-baselines`' false-positive row *was* executed and the runs are in `baselines/logs/`: Manual SQL 4.8×10⁵ (`manual_sql_baseline.json`, cross-rank subset) and Daikon-style 1.3×10⁵ (`daikon_style_baseline_loo.json`, leave-one-configuration-out) both reproduce. What is unbacked is (a) the ≤3/13 and ≤5/13 class-coverage bound, which the paper states as analytical, and (b) **TrainAudit's own 25.8 FP/1M** in the same row. | the third column of the table is the one without data | locate or re-run the six-case 504K-evaluation set; `baselines/logs/README.md` records the two denominators that differ between protocols |
| O4 | ~~`tab:overhead` has raw logs but no parser~~ **CLOSED.** `benchmark/injection/parse_overhead_logs.py --check` recomputes all eight values from the raw logs and asserts them against the paper: 732 ms / 191.9 s / 27.5 s / 25.0 s / 262× / 38× / 34× / 7.7× all reproduce within 0.1%. The 2026-07-12 session independently replicates it. The confusing gpt-tiny CPU file is labelled in `benchmark/eval/OVERHEAD_FILES_NOTE.md`. | — | — |
| O5 | **6 of 15 figures have no generator script** (+ `fig_training_panorama`, hand-drawn). | figures cannot be regenerated from data | **mitigated**: `scripts/verify_figures.py` lifts the numbers back out of the figure PDFs with `pdftotext` and checks them against the data, which is what a generator would have guaranteed. 12 checks pass, 0 mismatch. Three figures stay unbacked (below) and three are illustrative. |
| O6 | Figure generators that do exist **hard-code their numbers** instead of reading the result CSVs. | a number can drift between figure and table | **mitigated the same way** — `verify_figures.py` catches drift after the fact for every figure that renders numbers, including the funnel stages, the skip-L3/L4 denominators, the 392 pool counts, the catalog curve, and the amortization crossings. |
| O7 | **Temporal holdout is not blocked by the data.** 385 of 392 records carry an issue URL or commit hash; `resolve_record_dates.py` dated **374 (95%)** from upstream. The corpus spans 2021-01-26 to 2026-05-06 and a median cut at 2025-03-14 splits it 186/188 with all 13 categories on both sides. What remains is the re-mining run itself. | an overfitting question a reviewer will ask, now answerable | run the holdout on `temporal_split.json`, **and see O22 first** |
| O22 | **A temporal split on this corpus partly confounds with framework.** At the median cutoff OLMo is 69 train / 5 test and OLMo-core is 5 train / 63 test, because OLMo-core is the newer project. DeepSpeed (70/52) and Megatron-LM (42/68) are better balanced. | a temporal holdout would partly measure framework transfer, which §5.4 already reports separately | either stratify the cutoff per framework, or report the confound alongside the result rather than letting a reviewer find it |
| O8 | **H20 vs H200 mismatch**: §5.1 says replays run on 8×H200, §5.5 microbenchmarks are 1.2B/H20. The text does say "H20 harness", but the two platforms sit two paragraphs apart. | a reviewer may read the 7.7× as an H200 result | keep the caveat adjacent to `tab:overhead` |
| O9 | ~~Trace DBs indexed, not shipped~~ **CLOSED.** They compress 7×: 129 `.db` files from 43 runs are 39.5 MiB packed, published as release `trace-dbs-v1` with a per-file SHA-256 manifest and `scripts/fetch_trace_dbs.sh`. All 42 databases the guard ablation reads are covered, and `run_d1_phase3.sh` now honours `MEGATRON`/`SDCCHECK_ROOT`, so §5.3 is re-runnable off this repository. | — | — |
| O10 | **9.2 GB `rebuttal_v1/` + 550 MB `hunt_log/` excluded.** | prior-round evidence unavailable | leave out; they are not cited by the paper |
| O11 | ~~Public repo framed for double-blind~~ **CLOSED in this artifact.** `ANONYMOUS_RELEASE.md` removed; `CITATION.cff`, `LICENSE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `README.md` all de-anonymised by step 10 of the assembler, so a re-run cannot silently reintroduce it. | — | still to do on the *published* repo |
| O12 | `\vldbavailabilityurl` is **empty** (`main.tex:16`). | the artifact URL never appears in the paper | needs the published URL — blocked on the decision to push (see `AUDIT_STATE.md`) |
| O13 | The production discovery is described **without an upstream identifier** (a double-blind artefact). | the strongest claim in the paper is unverifiable as written | cite `NVIDIA/Megatron-LM#4641`; evidence and exact state verified in `UPSTREAM_BUG_EVIDENCE.md` |
| O14 | **Three generations of detection data ship together and only one matches the paper.** `real_sdc/SMOKE_REPORT.md` (17 cases) is what `numbers.tex` and appendix `tab:detection-results` are computed from; `results.csv` scores a different 17-case set (9 shared) and `results_gpu.csv` an earlier 14-case set (13/14). The superseded ones have the more obvious filenames. | a reviewer opening `results.csv` or `paper_table_baseline_traincheck.md` (TrainCheck 10/17) sees numbers that contradict the paper | **addressed**: `benchmark/eval/DETECTION_FILES_NOTE.md` explains the three sets; the current set is now machine-readable as `real_sdc/real_se_detection.csv`, generated by `extract_detection_csv.py`; `scripts/verify_paper_numbers.py` flags the superseded files by name |
| O15 | **Origin of the long-standing "0/17 vs 1/17" FP discrepancy found.** `results.csv:35` records `T1-sqrt-decay-front-loaded` firing on the *fixed* side of `OC-NEW-3` (`\|slope\|=0.5`) while its `phase` column still says `buggy` — a mislabelled row that reads as one fixed-side false positive. | the unresolved item in `docs/experiment_registry.md` | it does not affect the reported result: `OC-NEW-3` is not in the current case set. Documented in `DETECTION_FILES_NOTE.md`; the mislabelled `phase` value should be corrected or the file marked historical |
| O21 | **The 25.8 / 83.3 FP/1M pair needs its own source file.** Working back from the published rates, they imply ≈13 and ≈42 violating evaluations over 504 K — a ratio of 3.2×, matching the figure's own annotation. That is a *row-level* denominator, so it is a different measurement from the shipped leave-one-out data, whose five clean databases give 9 and 35 false positives over 438 *rule-level* evaluations. The two cannot be reconciled by arithmetic; the 504 K run simply is not here. | the number appears three times (`fig:predicate-ablation`, `tab:db-baselines`, the `fig:three-predicate-sql` caption) | locate the clean-trace run that produced it, or recompute both rates on the shipped clean databases and update all three sites together |
| O16 | **`fig:tier-coverage` has no data at all.** `paper_v2/overhead_c3_schema.csv` is header-only (0 rows), so the coverage 28→78% / overhead 1.5→7.5% curve and the "~8% → ~1.5%" claim in §4.4 rest on nothing shipped. | an appendix figure and a §4.4 sentence are unsupported | measure the tiers, or mark the figure as illustrative |
| O17 | **`paper_v2/portability.csv` provenance is circular**: its own `source` column reads `main_cn.tex portability section`, i.e. it was transcribed *from* the paper. Its `a0_detected`/`a1_detected` columns also do not match the mined/reused/failed cells rendered in `fig_portability_matrix.pdf` (only the adapter-LoC column agrees). | the file cannot verify the figure it appears to back | regenerate from `p9_p16_deployment/` and `mining_baseline_result_*.json` |
| O18 | The paper's §5.2 footnote attributes the non-clean fixed side to **M-020**, but in the shipped data M-020's fixed side is `CLEAN`; the case without a clean fixed row is `OC-NEW-3` in the superseded set. Under the current set the footnote may well be right, but nothing shipped confirms which case it is. | a checkable footnote that cannot be checked | name the case as it appears in `real_se_detection.csv` |
| O19 | The amortization crossings are derived two different ways. The naive K≈2630 is the exact crossing from the measured 192 s dump (recomputed: 2622), but the optimised K≈380 and K≈760 are the naive value divided by the rounded 7× speedup — recomputed from the measured 25 s they are **342 and 683**. | no claim breaks: at K=380 the real overhead is 9.0%, so "<10%" holds with room to spare. But the "~7× shorter period" reads as an independent result when it is actually the input to the other two numbers. | either quote 342/683 from the measurement, or say plainly that the optimised period is the naive one scaled by the measured speedup |
| O20 | `fig_catalog_generalization`'s "+19.7 pts at equal size" is **not** the headline gap (35.3 pts overall, 39.9 pts on observable records) but the catalog's coverage at the free-form arm's taxonomy size (67.5%) minus that arm's (47.8%). | correct, but easy for a reviewer to read as the headline gap and conclude the numbers disagree | make the equal-size basis explicit in the caption |
