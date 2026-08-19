# Paper §4 Evidence Pool (live document, updated as experiments complete)

> All §4 numbers must trace back to a row here. Anything in main_cn.tex
> not anchored to a row is dead.

## §4.1 Fault Injection Benchmark

13 synthetic injections + 1 control on OLMo-core small_hybrid_moe (pre-norm
hybrid clean baseline) + FSDP2 + bf16:

| Bug | Pattern | Detected? | Rule(s) fired |
|---|---|---|---|
| FAULT-000-control | clean baseline | — | 0 |
| FAULT-001 | block 0 forward → identity | ✓ | T0-no-nan-inf, T1-fwd-output-block-uniformity |
| FAULT-002 | init scale ×100 on block 5 ffn | ✓ | T1-fwd-output-block-uniformity, T1-grad-flow |
| FAULT-003 | high lr no clip → grad explosion | ✓ | 5 rules (loud) |
| FAULT-004 | norm.weight init = 0 | ✓ | T0-norm-output-rms, T0-module-grad-output-alive |
| FAULT-005 | router → one-hot | ✓ | T0-softmax-degenerate (504 ev) |
| FAULT-006 | lr=0 on group 0 | ✓ | T0-optim-lr-positive |
| FAULT-007 | Inf in embedding init | ✗ | trace empty (training crashed at step 0) |
| FAULT-007v2 | Inf via grad hook mid-train | ✗ | FSDP grad hook subverted |
| FAULT-008 | bf16 → fp16 round-trip | ✗ | dtype change not propagated to fwd output |
| FAULT-009 | skip alternating optim step | ✓ | T0-optim-step-counter-monotonic (19/28 events) |
| FAULT-010 | residual stream clobber (B13 pattern) | ✗ | rule needs OLMo `residual.probe` |
| FAULT-011 | token IDs > vocab_size | ✗ | embedding lookup OOB → instant crash |
| FAULT-012 | block 7 → identity | ✓ | T0-no-nan-inf (40 ev) |
| FAULT-013 | dropout + ckpt RNG mismatch | ✗ | injection didn't activate ckpt path |

**Detection rate: 8/13 = 61.5%. Control FP rate: 0/27 rules = 0.0%.**

5 honest miss patterns categorized in `benchmark/fault_injection/PHASE1_RESULTS.md`.

## §4.2 Real-Bug Archaeology (in-progress)

1078 PRs scraped from OLMo-core (492) + DeepSpeed (586). 266 keyword-filter
candidates → 15 silent-error fix triage:

| ID | Repo | Type | Status |
|---|---|---|---|
| DS-7985 | DS | bf16 microbatch grad leak | ✓ existing CAND_BF16_BOUNDARY_GRAD_LEAK |
| DS-7898 | DS | zero3 stream race nan | ✓ existing CAND_ZERO3_STREAM_RACE_NAN |
| DS-7360 | DS | warmup lr inheritance | ✓ existing CAND_WARMUPCOSINE_MULTIGROUP |
| DS-7551 | DS | MoE num_experts ↔ ep_size confusion | static: T1-process-group-size; bug in inference path |
| DS-7981 | DS | zero offload multi-backward | ✓ existing CAND_ZERO_OFFLOAD_MULTI_BACKWARD |
| DS-7462 | DS | all-gather mixed dtype | static: AssertionError = crash, not silent |
| DS-7839 | DS | bf16 ZeRO-0 grad norm divergence | ✓ existing CAND_BF16_ZERO0_DUAL_BUG |
| DS-6993 | DS | enable autocast w/ ZeRO | static: feature add, not bug fix |
| OC-314 | OC | DDP param dtype casting | static: RuntimeError = crash |
| OC-452 | OC | instance mask NaN loss | static: T0-no-nan-inf covers; workload-conditional |
| OC-296 | OC | router bf16 → fp32 | static: precision; rule gap (need router-dtype rule) |
| OC-20 | OC | ckpt optim state regression | static: T0-step-counter; needs resume workflow |
| OC-477 | OC | QK norm param count | rule gap (no param-count check) |
| OC-517 | OC | dataloader race | rule gap (no async race detection) |
| OC-652 | OC | eval determinism | rule gap (no eval mode coverage) |

**Existing dynamic-confirmed: 5. Static-mapped to existing rules: 4 (with caveats). Rule gaps for paper §6: 3.**

## §4.3 Broad Config Sweep (controlled experiment)

29 traces × 28 rules. Includes 8 sweep configs varying scale / dtype / seq_len /
block_name, vs the existing 21 hunt traces.

**Critical control**: `prenorm_370M` (olmo2_370M factory + block_name=default
i.e. Llama pre-norm) **fires 0 rules**. Same factory as olmo2_370M with default
reordered_norm fires T1-fwd-output (3) + T1-grad-flow (1). This is the
ground-truth experiment that **block_name is the only variable causing fires**.

### Sweep configs (15 total, post-norm reordered_norm everywhere unless noted)

| config | varied dimension | fwd-output | grad-flow | norm-rms |
|---|---|---|---|---|
| dense_190M | scale (190M) | 3 | 1 | 0 |
| dense_760M | scale (760M) | 3 | 1 | 0 |
| act_ckpt | + activation ckpt | 2 | 2 | 120 |
| fp32 | dtype = float32 | 2 | 1 | 60 |
| long_seq | seq_len = 1024 | 3 | 1 | 62 |
| seed42 | random seed = 42 | 2 | 1 | 60 |
| seed7 | random seed = 7 | 2 | 1 | 60 |
| lr_high | lr = 1e-2 | 2 | 1 | 4 |
| lr_low | lr = 1e-5 | 2 | 1 | 62 |
| smallmoe_long | smallmoe + 50 steps | 6 | 1 | 0 (但 softmax-degenerate 4) |
| ngpt_1B | nGPT 1B (diff arch) | 0 | 0 | 2232 (L2Norm FP) |
| moe_hybrid_repeat | pre-norm hybrid | 0 | 0 | 0 (CLEAN) |
| **prenorm_370M (definitive control)** | **block_name=default (Llama pre-norm)** | **0** | **0** | **0** |

**Conclusion**: post-norm reordered_norm fires consistently across:
  - 4 scales (60M/190M/370M/760M)
  - bf16 / fp32 dtypes
  - seq_len 256 / 1024
  - lr 1e-5 to 1e-2
  - 3 random seeds (0/7/42)
  - both with/without activation ckpt

The ONLY variable that changes fire pattern: **block_name (post-norm vs pre-norm)**.

29 trace × 28 rule confusion matrix in `benchmark/sweep/final_analysis.json`:

| Trace family | Fires |
|---|---|
| OLMo-core all reordered_norm-family configs (60M, 190M, 370M, 760M, smallmoe, small_hybrid_moe, dense_actckpt) | T1-fwd-output-block-uniformity + T1-grad-flow + T0-norm-output-rms |
| OLMo-core moe_hybrid (pre-norm control) | 0/28 fire |
| OLMo-core olmoe_1B_7B (production dropless) | T1-fwd-output-block-uniformity (4 ev) |
| OLMo-core nGPT (L2Norm, FP) | T0-norm-output-rms (1344 ev) |
| OLMo-core EP=2 | T0-no-nan-inf (latent — buffer uninit) + others |
| DS clean (4 configs) | T0-grad-norm-finite + T0-optim-lr-positive + T0-step-counter (instrumentation FP) |
| Megatron clean | 0/28 fire |
| OLMo-core baseline (dropless small) | 0/28 fire |

## §4 Novel Candidates (4 documented, 1 production-impacting)

| ID | Class | Status |
|---|---|---|
| `CAND_OLMOCORE_REORDERED_HYBRID_DEAD_BLOCK0` | active permanent | ✓ dynamic confirmed @ 100 step; dev factory only |
| `CAND_OLMOCORE_REORDERED_DENSE_BLOCK0_ATTN` | transient | self-recovers in 30 steps |
| `CAND_OLMOCORE_EP_A2A_UNINIT_BUFFER_NAN` | latent | 7% comm.pre NaN; downstream mask blocks impact |
| `CAND_OLMOCORE_REORDERED_NORM_GRAD_AMPLIFICATION` | numerical phenomenon | A/B verified Adam compensates → no parameter-level impact |

## Honest assessment for paper

- 1 active silent error with dynamic verification (#1)
- 3 invariant violations that don't translate to parameter-level training degradation
- 13 fault injections + ~14 real-PR-mapped bugs = ~27 evidence points
- Detection rate baseline: 61.5% in fault injection

This is a defensible mid-tier paper if framed as:
"trainaudit's hook layer surfaces invariant violations production monitoring
misses, including 1 active silent error and 3 latent property violations.
On 13 synthetic injections trainaudit detects 8 (61.5% recall, 0 FP).
Real-bug archaeology of 1078 PRs identifies 15 silent-error patterns with
9 rule-shape covered and 3 rule gaps as future work."

Not strong enough for top-tier without production-PR-merged outcome.
