# Paper §4.3 — Clean-run FP rate

Workload: `gpt-tiny` × 200 steps, no injected bug.

| metric | value |
|---|---:|
| rules evaluated | (T0_PYTORCH tier) |
| rules that fired | 0 |
| FP rate | 0.0% |

Fired rules: none

Combined with selected_synthetic_5 fixed-commit FP=0/5, the
clean-run FP corpus has zero false positives so far.
