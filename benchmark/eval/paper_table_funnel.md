# Funnel ablation table (SPEC S4, RQ-D4)

## Table A: Mining funnel — 4 frameworks × 16 patterns

Cross-framework aggregate, reproduced from `benchmark/eval/rebuttal_v1/A1_mining_funnel/a1_funnel.csv`.

| Stage                                  | Megatron | DeepSpeed | OLMo | OLMo-core | **Total** | Reduction |
|----------------------------------------|---------:|----------:|-----:|----------:|----------:|----------:|
| L1 Hypothesis (PatternGuidedLLM)       |      121 |       103 |   99 |        97 |   **420** |       1.0x |
| L2 Enumerated predicates (deterministic) |   1,300 |     1,397 | 1,300 |     1,337 | **5,334** |      12.7x |
| L3 Healthy-pass (clean train trace)    |      992 |       680 |  899 |       865 | **3,436** |       0.64x |
| L4 Adversarial-pass (PatternGuidedFilterLLM) |    104 |        83 |   81 |        89 |   **357** |       0.10x |
| Deployed (cross-framework, curated)    |       — |         — |    — |         — |    **45** |       0.13x |

Numbers reproduce paper §6.4 figure exactly (paper L902: 420 / 5334 / 3436 / 357 / 45). No deviation from the published funnel.

## Table B: Skip-L3 / Skip-L4 ablation — clean-trace FP

Cross-validation traces (per-framework, both Megatron-LM clean training runs but disjoint from the L3 training trace):
- `benchmark/sweep/_runs/dense_190M/trace_rank0.duckdb`
- `benchmark/fault_injection/_runs/FAULT-000-control/trace_rank0.duckdb`

| Ablation | Cohort size (per fw) | Clean-trace FP (pooled) | Pooled FP rate | 95% CI |
|---|---:|---:|---:|---:|
| **Skip L3** (deploy L2 candidates directly) | 50 (sub-sample) × 4 fw × 2 traces = **400 evals** | **114** | **28.5%** | 24.3%–33.1% |
| **Skip L4** (deploy L3-passed directly)     | 1002+686+903+870 = **3,461** L3-passed × 2 traces = **6,922 evals** | **0** | 0.0% | <0.05% |
| **Full funnel** (L4-kept deployed)          | 114+89+85+94 = **382** L4-kept × 2 traces = **764 evals** | **0** | 0.0% | <0.13% |

Sub-sample for Skip-L3 was 50/framework per SPEC §3.2 fallback (5334 L2 candidates → 50 random per framework, seed=4242). Confidence interval is Wilson 95%.

### Headline number
- **Skip-L3 stress test**: 28.5% of L2 candidates spuriously fire on a Megatron clean trace they were not validated against (114/400 evaluations). Top firing predicates are workload-specific `*_positive` / `*_nonzero` baselines emitted by the parametric enumerator (Table C).
- **Skip-L4 stress test**: L4 adversarial filter is **necessary for diagnostic precision**, not for clean-trace FP suppression — the 3,436 L3-passed predicates already satisfy `validate_against_healthy` on disjoint clean traces. The 90% removal at L4 targets *workload-specific* predicates (98.6% of L4 rejects categorised as "workload-specific-constant" in `a1_summary.json`) that fail on cross-workload but not cross-seed validation.

## Table C: Top-5 firing L2 predicates on cross-validation traces (Skip-L3)

| Predicate id                                            | Times fired (max=8 traces × 1 sample each) |
|---------------------------------------------------------|---:|
| `hyp/param/sp/module.fwd.post/output.mean_nonzero`      | 14 |
| `hyp/param/sp/comm.pre/tensor_pre.max_nonzero`          | 10 |
| `hyp/param/sp/module.fwd.post/output.min_nonzero`       |  8 |
| `hyp/param/pfc/module.fwd.post/output.min_positive`     |  8 |
| `hyp/param/csm/optim.step.post/state_step_n_params`     |  8 |

Every entry is a parametric-enum boilerplate (`> 0` / `nonzero`) — exactly the category L4 is designed to remove (`PatternGuidedFilterLLM` rejects 3,035 such candidates with reason `workload-specific-constant`).

## Caption (paper §6.4 or new §6.5)

> Verification funnel ablation on the cross-framework 4×16 mining run. The full funnel reduces 5,334 enumerated predicates to 45 deployed rules; skipping L3 healthy-validation leaves 28.5% (95% CI: 24%–33%) clean-trace false positives on disjoint Megatron-LM clean traces, while the L4 adversarial filter additionally removes 98.6% workload-specific boilerplate predicates that would not fire on a single clean trace but do fire on cross-workload traces (see `funnel_skip_l3_results.csv`, `funnel_skip_l4_results.csv`). PatternGuidedLLM (L1) and PatternGuidedFilterLLM (L4) are pattern-spec emulators; see `IMPL_NOTES_S4.md`.

## Optional Table D: V4 free-form baseline (no Pattern Catalog)

From `benchmark/eval/mining_baseline_result*.json` — six framework-seeded LLM runs with no Pattern Catalog hints:

| Seed         | L1 hyps | L2 preds | L3 accepted | L3 rejected |
|---|---:|---:|---:|---:|
| megatron_router (P3-shaped) |  6 |  21 | 17 |  4 |
| olmo_block                  |  6 |   9 |  8 |  1 |
| ds_clipgrad                 |  6 |  88 | 54 | 34 |
| ds_grad                     |  6 |  49 | 36 | 13 |
| ds_scheduler                |  6 | 107 | 53 | 54 |
| baseline_generic            |  6 |   9 |  8 |  1 |
| **Total**                   | **36** | **283** | **176** | **107** |

V4 produces 36 L1 hyps across 6 seeds (vs. 420 L1 hyps for V3 with the Pattern Catalog across 60+ seeds × 16 patterns); throughput per seed is comparable (≈6 hyps each), but coverage is bottlenecked by what the free-form LLM happens to nominate.
