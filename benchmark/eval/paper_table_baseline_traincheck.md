# Paper §4.1 — TrainAudit vs TrainCheck (D1 same-集合)

## Per-bug verdict comparison (buggy phase)

| bug_id | framework | TrainAudit | TrainCheck | violations | note |
|---|---|---|---|---|---|
| B1 | megatron-lm | DETECTED | CLEAN | 0/315 | traincheck:0/315 failed |
| B2 | megatron-lm | DETECTED | DETECTED | 6/520 | traincheck:6/520 failed invariants |
| B3 | deepspeed | DETECTED | CLEAN | 0/439 | traincheck:0/439 failed |
| B8 | deepspeed | DETECTED | DETECTED | 2/433 | traincheck:2/433 failed invariants |
| B11 | deepspeed | DETECTED | DETECTED | 11/604 | traincheck:11/604 failed invariants |
| B12 | olmo-core | DETECTED | DETECTED | 24/478 | traincheck:24/478 failed invariants |
| B13 | olmo | DETECTED | DETECTED | 2/741 | traincheck:2/741 failed invariants |
| M-012 | megatron-lm | DETECTED | CLEAN | 0/431 | traincheck:0/431 failed |
| M-014 | megatron-lm | DETECTED | DETECTED | 3/432 | traincheck:3/432 failed invariants |
| M-020 | megatron-lm | DETECTED | DETECTED | 1/758 | traincheck:1/758 failed invariants |
| M-024 | megatron-lm | DETECTED | DETECTED | 1/430 | traincheck:1/430 failed invariants |
| M-NEW-5 | megatron-lm | DETECTED | CLEAN | 0/432 | traincheck:0/432 failed |
| O-005 | olmo | DETECTED | FAIL | infer:     assert not isinstance(            ^^^^^^^^^^^^^^^ AssertionError: Exc | infer:     assert not isinstance(            ^^^^^^^^^^^^^^^ |
| O-NEW-1 | olmo | DETECTED | DETECTED | 1/683 | traincheck:1/683 failed invariants |
| O-NEW-9 | olmo | DETECTED | CLEAN | 0/260 | traincheck:0/260 failed |
| OC-NEW-2 | olmo-core | DETECTED | DETECTED | 18/506 | traincheck:18/506 failed invariants |
| OC-NEW-3 | olmo-core | DETECTED | CLEAN | 0/381 | traincheck:0/381 failed |

## Summary

- TrainAudit DETECTED: **17/17** (100.0%)
- TrainCheck DETECTED: **10/17** (58.8%)
- TrainCheck FAIL (driver/infer crash): 1
