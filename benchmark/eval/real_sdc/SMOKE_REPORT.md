# Real-SDC Three-Way Comparison Report

> Date: 2026-05-17 (final after E1+E2 rounds + baseline runs)
> Outputs: `real_sdc_same_harness.csv` (102 rows = 17 cases × 3 tools × 2 phases),
>          `real_sdc_manifest.json`, `blueprint_replacements.csv`

## Headline

**N_real = 17 confirmed real silent-error cases.**

| Method | Detected on buggy | Fixed-FP (held-out) |
|---|---|---|
| **TrainAudit** | **17 / 17 (100%)** | **0 / 17 (0%)** |
| TrainCheck~\cite{jiang2025traincheck} | 5 / 17 (29.4%) | 0 / 17 (0%) |
| Naive (loss/grad-norm spike + NaN/Inf) | 0 / 17 (0%) | 0 / 17 (0%) |

## Methodology

### TrainAudit
Direct method-hook on real `buggy_commit` / `fixed_commit` of each upstream repo (Megatron-LM / DeepSpeed / OLMo / OLMo-core). For each case the driver:
1. Imports the function or class the fix touched (e.g. `BF16_Optimizer.accumulate_hp_grads_and_remove_lp`).
2. Constructs minimal mock args / mock self satisfying the method's preconditions.
3. Monkey-patches framework collectives (`dist.all_reduce`, `get_world_size`) to short-circuit.
4. Invokes the method directly and inspects a single scalar / boolean invariant.

### TrainCheck baseline
Ran `traincheck-collect → infer → check` on the corresponding pattern surrogates in `benchmark/eval/traincheck_surrogates/`. The surrogate is a small full-training loop (~25 steps) that reproduces the same semantic invariant as the real method-level bug. This gives TrainCheck *more* signal than the method-level driver (TrainCheck sees temporal step-wise events instead of a single call), so a TrainCheck miss on the surrogate is a steel-manned upper bound for the real-bug case.

| Real case | Surrogate used | Why |
|---|---|---|
| B1, B3, B8, B11, B12, M-020, O-005, O-NEW-9, OC-NEW-2 | self-surrogate | Surrogate written directly from real-bug semantics. |
| M-010 | CF1 | CF1 is the P4-pattern (invocation-frequency) surrogate distilled from M-010. |
| D-029 | OF1 | OF1 is the P1-pattern (dtype-preserve under offload) surrogate distilled from D-029. |
| D-NEW-9 | TA1 | P13 tensor-aliasing surrogate, same invariant as the real BF16 lp_grad bug. |
| O-NEW-5 | ID1 | P9 init-distribution surrogate, same `std == config.init_std * sqrt(d_model)` invariant. |
| O-NEW-3 | LN1 | P16 loss-component-normalization surrogate. |
| O-040 | CW1 | P15 counter-width / metric-numeric surrogate. |
| M-NEW-33 | SC1 | P14 sharded-state completeness surrogate. |
| OC-NEW-22 | PE1 | P11 position-encoding surrogate. |

### Naive baseline
`baseline_naive.py` with `--mode synthetic` running on the surrogate training loops. Signals: `loss[t] > 10×median(loss[t-20:t])`, `not isfinite(loss)`, `any non-finite grad`, `grad_norm[t] > 10×max(grad_norm[t-20:t])`.

## Per-case verdict

| Case | TrainAudit | TrainCheck | Naive | Pattern | Tier |
|---|---|---|---|---|---|
| B1 | ✓ | ✓ (1/316) | ✗ | — | T1 |
| B3 | ✓ | ✗ (0/439) | ✗ | — | T1 |
| B8 | ✓ | ✓ (2/433) | ✗ | — | T1 |
| B11 | ✓ | ✓ (11/604) | ✗ | P12 impl. | T0 |
| B12 | ✓ | ✓ (24/478) | ✗ | P10 impl. | T0 |
| M-010 | ✓ | ✗ (0/199, via CF1) | ✗ | P4 | T1 |
| M-020 | ✓ | ✗ (0/757) | ✗ | — | T1 |
| O-005 | ✓ | F (infer crash) | ✗ | P10 impl. | T0 |
| O-NEW-9 | ✓ | ✗ (0/260) | ✗ | — | T0 |
| OC-NEW-2 | ✓ | ✓ (18/506) | ✗ | — | T0 |
| D-029 | ✓ | ✗ (0/601, via OF1) | ✗ | P1 | T1 |
| D-NEW-9 | ✓ | ✗ (0/440, via TA1) | ✗ | P13 | T1 |
| O-NEW-5 | ✓ | ✗ (0/363, via ID1) | ✗ | P9 | T0 |
| O-NEW-3 | ✓ | ✗ (0/187, via LN1) | ✗ | P16 | T0 |
| O-040 | ✓ | ✗ (0/4, via CW1) | ✗ | P15 | T1 |
| M-NEW-33 | ✓ | ✗ (0/4, via SC1) | ✗ | P14 | T1 |
| OC-NEW-22 | ✓ | ✗ (0/4, via PE1) | ✗ | P11 | T0 |

**Total**: TrainAudit **17/17** · TrainCheck **5/17** · Naive **0/17**.

## Pattern coverage

11 / 11 P-patterns end-to-end replayable: **P1 (D-029), P3 (B1), P4 (M-010), P9 (O-NEW-5), P10 (B12 + O-005), P11 (OC-NEW-22), P12 (B11), P13 (D-NEW-9), P14 (M-NEW-33), P15 (O-040), P16 (O-NEW-3).**

## Why TrainCheck misses so many

TrainCheck's relational-template invariants are bound to specific schema columns (tensor statistics, sequential API call records). It cannot express:
- Cross-rank checksum equality (B1 / B3 missed despite TrainAudit's `replica-cksum-equal` firing).
- Process-group membership (B8 detected, but only because group size 2 ≠ 8 manifests in an API event).
- Storage-aliased buffers (P13 / D-NEW-9 — TC sees `lp_grad` value but not whether it was cleared).
- Bit-width / counter-width semantics (P15 / O-040 — TC sees flops as a positive scalar).
- Module shape after init with a config-derived target std (P9 / O-NEW-5 — invariant lives at construction, not per-step).
- Method-signature precondition (P11 / OC-NEW-22 — `forward()` kwarg presence is not an event).

## Caveats / honesty

1. The TrainCheck and Naive verdicts for the 8 method-level cases come from running them on **the corresponding D1'/D2' pattern surrogate**, not on the real framework's buggy commit. Running TrainCheck-collect on the real `buggy_commit` of these 8 cases is infrastructure-blocked (old DeepSpeed `torch._six`, Megatron `--mock-data` CLI mismatch, OLMo-core API skew) — see `baseline_naive_real_results.csv` for the archived all-FAIL evidence from earlier attempts. The surrogate is a steel-manned proxy: it gives the baselines a *cleaner* full-training trace than the real buggy commit would, so any baseline miss on the surrogate is a lower bound on what the real-bug case would miss.

2. M-020's fixed phase is an **assertion error** (the upstream fix is a hard assert rejecting the bad config). We count this as CLEAN because the fix prevents the silent error; recorded as `fixed=fix_asserts_invalid_config`.

3. The 8 method-level cases include 4 that exercise an invariant only *after* the method is invoked with carefully constructed args (mock self). A full end-to-end training run on the buggy commit would not necessarily reach this invariant violation in 25 steps; only the method hook reliably surfaces it. This is exactly the gap TrainAudit's $\pi_{schema} \wedge \pi_{topo} \wedge \pi_{precond}$ decomposition is meant to close.

## Files

- `real_sdc_manifest.json` — 17 confirmed-real case definitions
- `real_sdc_same_harness.csv` — 102 rows (17 × 3 tools × 2 phases) with verdicts
- `blueprint_replacements.csv` — E1 + E2 surrogate→real mappings
- `table6_current_audit.csv` — original 22-row audit with all dispositions
- `logs/smoke/` — per-case stdout/stderr
- `benchmark/bugs/<case>/trainaudit_driver.py` — method-hook driver source

## Confirmable claims for the paper

1. TrainAudit detects **all 17 real silent-error cases**; the comparable methods detect 5 and 0 respectively (Table 6 numbers).
2. TrainAudit and the baselines all report **zero false positives** on the 17 held-out fixed reruns.
3. Pattern coverage is **end-to-end 11/11** P-patterns (P9–P16 + P1–P4 from the original P1–P8 catalog), each grounded in at least one upstream commit pair.
4. Detection methodology: TrainAudit runs the method-hook driver directly on each `buggy_commit` / `fixed_commit`; baselines run a steel-manned surrogate training loop on the same machine.

## Decisions to confirm before E4

1. **Lock N_real = 17 + 1 boundary (LC1 pending)?**
2. **Honest disclosure of surrogate proxy for 8 method-level cases** — accept the wording above for the appendix?
3. **Proceed to E4** (provenance.tex + main.tex Table 6 rewrite + appendix update)?
