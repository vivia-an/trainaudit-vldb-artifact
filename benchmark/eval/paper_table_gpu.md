# Paper §4.1 — GPU verification (trainaudit_run.sh on eval-gpu-0)

- Bugs run on real GPU: **14** (4× H200, 4 frameworks)
- Buggy phase **DETECTED**: **13/14** = 92.9%
- Fixed phase CLEAN: **12/12** = 100%; **FP rate 0.0%**
- 2 FIXED phase(s) emitted no contract line (framework added an assertion on the buggy config in the fixed commit and aborted at init — itself evidence the bug class is gone).

## Per-bug verdicts

| bug_id | framework | category | buggy | fixed | rule fired |
|---|---|---|---|---|---|
| B1 | megatron-lm | moe_router_init | **CLEAN** | CLEAN | `` |
| B11 | deepspeed | gradient_clipping | **BUG_DETECTED** | CLEAN | `T0-clip-grad-bounded` |
| B12 | olmo-core | lr_schedule | **BUG_DETECTED** | CLEAN | `T0-initial-lr-present` |
| B13 | olmo | residual_connection | **BUG_DETECTED** | CLEAN | `T1-residual-stream-preserved` |
| M-012 | megatron-lm | dtype | **BUG_DETECTED** | CLEAN | `T1-expert-bias-fp32` |
| M-014 | megatron-lm | numerical | **BUG_DETECTED** | ? | `T0-softmax-degenerate` |
| M-020 | megatron-lm | configuration_validation | **BUG_DETECTED** | ? | `T1-layer-count-strict` |
| M-024 | megatron-lm | dtype | **BUG_DETECTED** | CLEAN | `T1-jitter-preserves-dtype` |
| M-NEW-5 | megatron-lm | loss_scaling | **BUG_DETECTED** | CLEAN | `T1-router-has-calculate-per-token-loss` |
| O-002 | olmo | numerical | **BUG_DETECTED** | CLEAN | `T1-residual-stream-preserved` |
| O-005 | olmo | control_flow | **BUG_DETECTED** | CLEAN | `T0-checkpoint-preserve-rng` |
| O-NEW-1 | olmo | normalization | **BUG_DETECTED** | CLEAN | `T0-norm-output-unit-rms` |
| OC-NEW-2 | olmo-core | optimizer | **BUG_DETECTED** | CLEAN | `T0-optim-step-counter-monotonic` |
| OC-NEW-3 | olmo-core | lr_schedule | **BUG_DETECTED** | CLEAN | `T1-sqrt-decay-front-loaded` |

## By framework

| framework | bugs | DETECTED | det_rate |
|---|---:|---:|---:|
| deepspeed | 1 | 1 | 100.0% |
| megatron-lm | 6 | 5 | 83.3% |
| olmo | 4 | 4 | 100.0% |
| olmo-core | 3 | 3 | 100.0% |
