# Paper coverage table — synthetic_17

- Buggy runs: **18**
- Buggy detected: **18** (100.0%)
- Buggy failed (driver/install/etc): 0
- Fixed-commit FP rate: 0.0% (0/16)

## By framework

| framework | buggy_runs | detected | failed | det_rate |
|---|---:|---:|---:|---:|
| deepspeed | 3 | 3 | 0 | 100.0% |
| megatron-lm | 7 | 7 | 0 | 100.0% |
| olmo | 4 | 4 | 0 | 100.0% |
| olmo-core | 4 | 4 | 0 | 100.0% |

## By framework × category

| framework | category | buggy_runs | detected | det_rate |
|---|---|---:|---:|---:|
| deepspeed | communication_dtype | 1 | 1 | 100.0% |
| deepspeed | gradient_clipping | 1 | 1 | 100.0% |
| deepspeed | moe_parallel_grouping | 1 | 1 | 100.0% |
| megatron-lm | configuration_validation | 1 | 1 | 100.0% |
| megatron-lm | dtype | 2 | 2 | 100.0% |
| megatron-lm | loss_scaling | 1 | 1 | 100.0% |
| megatron-lm | moe_router_init | 1 | 1 | 100.0% |
| megatron-lm | numerical | 1 | 1 | 100.0% |
| megatron-lm | tensor_parallel_grad | 1 | 1 | 100.0% |
| olmo | control_flow | 1 | 1 | 100.0% |
| olmo | data_processing | 1 | 1 | 100.0% |
| olmo | normalization | 1 | 1 | 100.0% |
| olmo | residual_connection | 1 | 1 | 100.0% |
| olmo-core | lr_schedule | 3 | 3 | 100.0% |
| olmo-core | optimizer | 1 | 1 | 100.0% |
