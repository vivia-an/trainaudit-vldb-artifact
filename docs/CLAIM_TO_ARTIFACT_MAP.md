# Claim → artifact map

One row per numbered float in `paper/main.tex` and `paper/appendix.tex`.
Paths are relative to this repository. "derived" means the paper states the number
is an analytical bound or an amortisation computed from a measured quantity, not a
fresh measurement — those rows carry no raw-data file by design, and the paper says so.

## Main paper

| Float | What it claims | Backing artifact | Status |
|---|---|---|---|
| `fig:micro-step-mismatch` (§1) | motivating micro-step mismatch | `figures/figure1_vector_v5.pdf` + `figures/generators/gen_fig1_vector_v5.py` | OK |
| `fig:silent-motivation` (§2) | silent-vs-loud framing | `figures/fig_silent_motivation.pdf` | **no generator** |
| `tab:approaches-comparison` (§2) | qualitative comparison vs prior detectors | prose/citations only | OK (no data) |
| `tab:taxonomy` (§3) | 392-record corpus, 13 labels, IRR κ=0.566 | `benchmark/eval/silent_evidence_392.json`, `benchmark/eval/v2_full/category_392_v2.csv`, `benchmark/eval/irr_50_results.json`, `benchmark/eval/build_392_catalog.py` | OK |
| `fig:architecture` (§4) | system overview | `figures/fig_overview_v32.pdf` + `generators/gen_overview_v32.py` | OK |
| `fig:three-predicate-sql` (§4) | π_topo/π_precond/π_schema → SQL | `figures/fig_three_predicate_sql.pdf` + generator; `core/dp_consistency_check.sql` | OK |
| `tab:trace-schema` (§4.4) | schema tiers S0–S6 | `collector/vtimeline/src/vtimeline/megatron_collector.py`, `benchmark/SCHEMA.md`, `benchmark/eval/paper_v2/overhead_c3_schema.csv` | OK |
| `tab:summary_3way` (§5.2) | TrainAudit 17/18, TrainCheck 5/17, Naïve 0/17 | `benchmark/eval/real_sdc/SMOKE_REPORT.md` → `real_sdc/real_se_detection.csv` (**authoritative**), with `traincheck_surrogates/*_{buggy,fixed}.py` and `gpu_logs/*.log` as run evidence. `results.csv`, `results_gpu.csv`, `paper_table_baseline_*.md` score **superseded case sets** — see `benchmark/eval/DETECTION_FILES_NOTE.md` | OK |
| `tab:db-baselines` (§5.2) | Manual SQL ≤3/13, Daikon-style ≤5/13, clean FP ≥83.3/1M | `docs/derivations/draft_S6_baselines.tex` (expressivity upper bound, derived class by class), FP floor reuses the −π_topo cell of `experiments/guard_ablation/d1_results.csv` | **derived, no run** |
| `fig:predicate-ablation` (§5.3) | −π_topo FP/1M 25.8→83.3; −π_precond det 6/6→4/6 | `experiments/guard_ablation/d1_results.csv` (126 cells), `benchmark/eval/paper_v2/mechanism_topology_p3.csv`, `mechanism_precond.csv` | OK |
| `fig:funnel-ablation` (§5.3) | funnel 420→5334→3436→357→45; skip-L3 28.5% FP | `benchmark/eval/funnel_counts.csv`, `funnel_skip_l3_results.csv`, `funnel_skip_l4_results.csv`, `core/scripts/reproduce_funnel_counts.py` | OK |
| `fig:guard-ablations` (§5.3) | leave-one-out guard ranking topo>adversarial>precond | `experiments/guard_ablation/d1_results.csv`, `experiments/guard_ablation_no_adversarial/` | OK |
| `tab:funnel-dual-axis` (§5.3) | dual-axis candidate/rule counts | `benchmark/eval/paper_v2/funnel_appendix.csv`, `benchmark/eval/catalog_ablation/catalog_ablation_summary.csv` | OK |
| `fig:catalog_generalization` (§5.4) | held-out record coverage of the frozen catalog | `benchmark/eval/catalog_generalization/` (`generalization_summary.csv`, `cov_A/B1/B2.jsonl`, `heldout_bugs.json`, `judge.py`) | OK |
| `tab:overhead` (§5.5) | 732 ms baseline; dump 192 s → 27.5 s → 25 s (7.7×) on 1.2B/H20 | `benchmark/injection/overhead_raw/session_20260630/*.log` → `overhead_h20.csv` via `parse_overhead_logs.py --check`; every value reproduces within 0.1%, replicated by `session_20260712/` | **verified** |
| `fig:amortization` (§5.5) | K≈380 → <10%, K≈2630 naive | `figures/generators/gen_amortization.py` (literals from `tab:overhead`) | derived |
| `fig:portability_matrix` (§5.6) | cross-framework rule transfer | `benchmark/eval/paper_v2/portability.csv` (**circular provenance**, and only its LoC column matches the figure), `mining_baseline_result_*.json`, `p9_p16_deployment/` | **partly unbacked** (O17) |

## Supplementary appendix

| Float | Backing artifact | Status |
|---|---|---|
| `tab:app-roadmap` | — (navigation) | OK |
| `tab:dataset-construction-flow` | `benchmark/eval/build_392_catalog.py`, `benchmark/eval/manifest_summary.md` | OK |
| `tab:silent-evidence` | `benchmark/eval/silent_evidence_392.json`, `silent_evidence_rate.py`, `silent_evidence_warnings.csv` | OK |
| `tab:irr` | `benchmark/eval/irr_50_{input,groundtruth,results}.json`, `irr_50_annotations/` | OK |
| `fig:bug-distribution-detail` | `benchmark/eval/v2_full/{category,pattern,tier,hook}_392_v2.csv` | **no generator** |
| `tab:template-induction`, `fig:template-induction` | `benchmark/eval/template_induction/`, `core/data/template_induction/PROTOCOL.txt`, `core/config/frozen_template_catalog.{json,sha256}` | OK |
| `fig:mining-funnel` | `benchmark/eval/funnel_counts.csv` | **no generator** |
| `tab:json_schema` | `benchmark/SCHEMA.md`, `core/dp_consistency_check.sql` | OK |
| `fig:diagnosis-mechanism`, §diagnosis | `benchmark/eval/diagnosis_s3_{input.json,rater_A.csv,rater_B.csv,rubric.md,score.py}` | data OK, **no figure generator** |
| `fig:case-studies` | `benchmark/eval/paper_v2/case_studies.csv`, `case_studies_macros.tex`, `figures/generators/gen_case_studies_v2.py` | OK |
| `tab:detection-results` (per case) | `benchmark/eval/real_sdc/real_se_detection.csv` (all 17 rows match), `real_sdc/table6_current_audit.csv`, `real_sdc/provenance_table.tex` | **verified** |
| `fig:tier-coverage` | `benchmark/eval/paper_v2/overhead_c3_schema.csv` is **header-only, 0 rows** | **UNBACKED** (O16) |
| `fig:training-panorama` | `figures/fig_training_panorama.png` | **no generator** (hand-drawn) |
| `tab:realse-class-coverage` | `benchmark/eval/real_sdc/real_sdc_manifest.json`, `real_sdc_same_harness.csv`, `manifest_v2.json` | OK |

## Headline numbers (`paper/numbers.tex` is the single source of truth)

| Macro | Value | Backing |
|---|---|---|
| `\NumRealSE` / `\NumRealSEReal` | 18 / 17 | `benchmark/eval/real_sdc/real_sdc_manifest.json` (`cases_confirmed_real` + `cases_boundary`) |
| `\NumRealSEDet` / `\RealSEDetRate` | 17 / 94.4% | `benchmark/eval/results.csv`, `gpu_logs/` |
| `\NumTCDet` / `\TCDetRate` | 5 / 29.4% | `benchmark/eval/baseline_traincheck_results.csv`, `traincheck_surrogates/` |
| `\NaiveDetFrac` | 0/17 | `benchmark/eval/baseline_naive_results.csv` |
| `\FixedFPFrac` | 0/17 | `benchmark/eval/fp_audit.csv` |
| 45 deployed / 24 active rules, 357 L4-passed | — | `core/config/frozen_template_catalog.json`, `benchmark/eval/funnel_counts.csv`, `benchmark/eval/p9_p16_deployment/` |

## Automated check

```bash
python3 scripts/verify_paper_numbers.py        # 22 numbers recomputed from the shipped data
```

Current state: **22 verified, 0 mismatched, 5 unbacked, 2 superseded files flagged.**
The unbacked items are listed as O3, O16, O17 and the `\NumFixedReplay` accounting in
`GAP_AUDIT.md`.
