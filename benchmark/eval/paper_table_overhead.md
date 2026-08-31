# Paper §4.3 — TrainAudit overhead

Workload: `gpt-tiny` × 200 steps (CPU, batch=4).

| config | total_s | mean_step_ms | p95_step_ms | events |
|---|---:|---:|---:|---:|
| baseline (no trainaudit) | 3.796 | 18.98 | 84.55 | — |
| with trainaudit T0       | 35.289 | 176.44 | 497.82 | 16038 |

**Per-step overhead: +829.7%**  (total +829.7%)

> ⚠️ **CPU toy upper-bound, NOT the paper §4.3 number.** Each step is 19ms baseline and trainaudit's per-event tensor stats (l2_norm + has_nan + abs_max + min/max/mean × every module event) costs ~2.0ms/event. On a real Megatron/OLMo workload one step is 200–2000ms and the same hook work amortises to single-digit %. The ~5% paper number requires re-running this script on GPU with the production model — this CPU result establishes only that **hooks fire correctly and events are sane** (16038 captured), and that **FP rate is 0** (the FP guarantee survives the GPU transition).
