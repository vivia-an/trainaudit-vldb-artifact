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
| O3 | `tab:db-baselines` (Manual SQL ≤3/13, Daikon ≤5/13) is an **analytical upper bound, never executed**. The paper says so, but there is no script a reviewer can run. | weakest evidence in §5.2 | either ship a small executable Manual-SQL/Daikon harness, or state the bound as a claim about expressivity and drop the fraction |
| O4 | ~~`tab:overhead` has raw logs but no parser~~ **CLOSED.** `benchmark/injection/parse_overhead_logs.py --check` recomputes all eight values from the raw logs and asserts them against the paper: 732 ms / 191.9 s / 27.5 s / 25.0 s / 262× / 38× / 34× / 7.7× all reproduce within 0.1%. The 2026-07-12 session independently replicates it. The confusing gpt-tiny CPU file is labelled in `benchmark/eval/OVERHEAD_FILES_NOTE.md`. | — | — |
| O5 | **6 of 15 figures have no generator script**: `fig_silent_motivation`, `fig_portability_matrix`, `fig_diagnosis_mechanism`, `fig_mining_funnel`, `tier_coverage`, `bug_distribution` (+ `fig_training_panorama`, hand-drawn). | figures cannot be regenerated from data | write generators, or state which figures are illustrative |
| O6 | Figure generators that do exist **hard-code their numbers** instead of reading the result CSVs. | a number can drift between figure and table | have generators read `benchmark/eval/*.csv` |
| O7 | **Temporal holdout is still blocked** (needs per-record commit dates); framework/pattern holdout rows are derived upper bounds, not re-mined. | an overfitting question a reviewer will ask | either re-mine on a time split or state the limitation in §5.4 |
| O8 | **H20 vs H200 mismatch**: §5.1 says replays run on 8×H200, §5.5 microbenchmarks are 1.2B/H20. The text does say "H20 harness", but the two platforms sit two paragraphs apart. | a reviewer may read the 7.7× as an H200 result | keep the caveat adjacent to `tab:overhead` |
| O9 | **Trace DBs (≈600 MB, 45 runs) are indexed, not shipped.** | reviewers cannot re-run the verifier end to end | publish as a release asset / Zenodo record and link from `docs/DATA_AVAILABILITY.md` |
| O10 | **9.2 GB `rebuttal_v1/` + 550 MB `hunt_log/` excluded.** | prior-round evidence unavailable | leave out; they are not cited by the paper |
| O11 | ~~Public repo framed for double-blind~~ **CLOSED in this artifact.** `ANONYMOUS_RELEASE.md` removed; `CITATION.cff`, `LICENSE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `README.md` all de-anonymised by step 10 of the assembler, so a re-run cannot silently reintroduce it. | — | still to do on the *published* repo |
| O12 | `\vldbavailabilityurl` is **empty** (`main.tex:16`). | the artifact URL never appears in the paper | needs the published URL — blocked on the decision to push (see `AUDIT_STATE.md`) |
| O13 | The production discovery is described **without an upstream identifier** (a double-blind artefact). | the strongest claim in the paper is unverifiable as written | cite `NVIDIA/Megatron-LM#4641`; evidence and exact state verified in `UPSTREAM_BUG_EVIDENCE.md` |
