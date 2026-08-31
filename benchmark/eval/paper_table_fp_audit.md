# Paper §4.3 — Long clean run FP audit

3 healthy models × 2 step counts = 6 clean runs at T1_FW_METADATA tier.

| model | steps | total violations | unique rules fired |
|---|---:|---:|---:|
| mlp-2l | 200 | 0 | 0 |
| mlp-2l | 500 | 0 | 0 |
| gpt-tiny | 200 | 0 | 0 |
| gpt-tiny | 500 | 0 | 0 |
| moe-style | 200 | 0 | 0 |
| moe-style | 500 | 0 | 0 |

**Zero rule fires across all 6 clean runs — 0 FP across the audit corpus.**
