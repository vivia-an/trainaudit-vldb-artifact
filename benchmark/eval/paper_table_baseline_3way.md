# Paper §4.1 — Three-way baseline comparison (D1 same-集合)

## Per-bug verdict (buggy phase)

| bug_id | framework | TrainAudit | TrainCheck | Naïve |
|---|---|---|---|---|
| B1 | megatron-lm | DETECTED | CLEAN | CLEAN |
| B11 | deepspeed | DETECTED | DETECTED | CLEAN |
| B12 | olmo-core | DETECTED | DETECTED | CLEAN |
| B13 | olmo | DETECTED | DETECTED | CLEAN |
| B2 | megatron-lm | DETECTED | DETECTED | CLEAN |
| B3 | deepspeed | DETECTED | CLEAN | CLEAN |
| B8 | deepspeed | DETECTED | DETECTED | CLEAN |
| M-012 | megatron-lm | DETECTED | CLEAN | CLEAN |
| M-014 | megatron-lm | DETECTED | DETECTED | CLEAN |
| M-020 | megatron-lm | DETECTED | DETECTED | CLEAN |
| M-024 | megatron-lm | DETECTED | DETECTED | CLEAN |
| M-NEW-5 | megatron-lm | DETECTED | CLEAN | CLEAN |
| O-005 | olmo | DETECTED | FAIL | CLEAN |
| O-NEW-1 | olmo | DETECTED | DETECTED | CLEAN |
| O-NEW-9 | olmo | DETECTED | CLEAN | CLEAN |
| OC-NEW-2 | olmo-core | DETECTED | DETECTED | CLEAN |
| OC-NEW-3 | olmo-core | DETECTED | CLEAN | CLEAN |

## Summary (buggy phase, n=17 D1 surrogates)

| 方法 | DETECTED | CLEAN (miss) | FAIL | det_rate |
|---|---:|---:|---:|---:|
| TrainAudit | 17 | 0 | 0 | 100.0% |
| TrainCheck | 10 | 6 | 1 | 58.8% |
| Naïve | 0 | 17 | 0 | 0.0% |

## Fixed-phase FP rate (n=17 fixed runs)

For paper FP evidence, TrainCheck fixed-phase rows must come from held-out clean reruns checked with invariants learned from a separate clean reference run. Do not report a reference-trace re-check as an independent FP measurement.

| 方法 | fixed CLEAN | FP rate |
|---|---:|---:|
| TrainAudit | 16/17 | 5.9% |
| TrainCheck | 17/17 | 0.0% |
| Naïve | 17/17 | 0.0% |

## Source

- TrainCheck per-bug results: `baseline_traincheck_results.csv`
- Naïve per-bug results: `baseline_naive_results.csv`
- TrainCheck pipeline (per bug): benchmark/eval/traincheck_surrogates/run_one.sh
- Naïve detector: 4 signals (loss_spike, loss_nan, grad_nan, gradnorm_spike) over 25 training steps
