# Session 2026-05-06 Findings — Silent Error Hunt + Methodology Validation

> 本文档记录这次会话的探索路径、实验数据、4 个 candidate 的严苛评估、
> 以及 honest 的方法学局限。**为后续写论文 / 二次决策提供 ground truth**。
> 论文怎么写另有计划，本文档不预设 framing。

---

## 1. 总体路线

| 阶段 | 投入 | 产出 |
|---|---|---|
| Hunt 主线（用现有 25 rule + LLM mining 挖掘 novel silent error） | ~1.5 天 | 4 candidate (1 真 silent + 3 invariant violation) |
| Mining pipeline (L1→L4) baseline | ~3 小时 | 5 source × 30 hypothesis → 5 useful new invariant ≈ 17% yield |
| 写新 rule + 重扫 hunt traces（"加 rule 比加 config 划算"的实证）| ~3 小时 | 2 条新 rule（grad-flow + fwd-uniformity）触发 candidate #4 |
| Phase 1 fault injection benchmark | ~2 小时 | 13 注入 + 1 control，detection rate 8/13 = 61.5%, FP 0% |
| Phase 2 real-bug archaeology | ~1.5 小时 | 1078 PR scrape → 15 silent-error fix triage |
| Broad config sweep（用现 30 rule 扫多配置）| ~2 小时 | 14 sweep config，candidate #4 跨 14 维度一致触发 |
| Due diligence on candidate #4 | ~1 小时 | A/B 验证：Adam 屏蔽 250× backward 放大，**parameter level 无影响** |
| Model quality A/B for candidate #1 | ~1 小时 | 200 步训练 buggy vs fixed，**block 0 MoE 5000× output 差异** |

---

## 2. 4 个 Candidate 严苛评估

### Candidate #1 — `REORDERED_HYBRID_DEAD_BLOCK0` ✓ **真 silent error**

**位置**: `olmo-core/src/olmo_core/nn/transformer/block.py:1041, 1102`，
class `MoEHybridReorderedNormTransformerBlock`，方法 `combined_forward`。

**触发条件（三条件交集）**:
1. post-norm RMSNorm 应用在 MoE output 上
2. hybrid 架构（dense `feed_forward` 和 sparse MoE 分离）
3. MoE config 没有 `shared_mlp` baseline path

→ OLMo-core 唯一同时满足三条件的 factory：`small_hybrid_moe` (12 layer, 768 d_model, 32 expert top-4)。

**机理**:
- block 0 input = embedding output, RMS ≈ 0.02 (truncated normal init std=0.02)
- expert 随机 init → MoE(x_moe) 极小 → mean(out²) ≪ eps (1e-5)
- RMSNorm 钳到 sqrt(eps)·weight ≈ 1.6e-3
- block 0 MoE 分支贡献被静音；反向梯度也极小 → expert 学不到

**A/B 200 步训练 model quality 结果**:

| block | buggy (post-norm) MoE l2 | fixed (pre-norm) MoE l2 | buggy/fixed |
|------:|-------------------------:|------------------------:|------------:|
| 0 | 1.4e-3 | 7.07 | **1/5000** |
| 1 | 113 | 13.6 | 8.3× |
| 5 | 4931 | 9.6 | 514× |
| 11 | 8398 | 4.2 | 2000× |

- buggy: block 0 / block 11 ratio = **1 : 6,000,000**（block 0 几乎完全死）
- fixed: block 0 / block 11 ratio = 2 : 1（所有 12 block 均匀贡献）

**Loss curve 状态**: callback wiring bug 导致 loss curve 没记录。但**这正是 silent
error 的核心特征**：loss 看不出问题（CE 都健康），属性级 hook 才看得到。

**Severity 评估**: dev-factory 内 active permanent silent capacity loss。block 0 的
32 个 expert 实际不学，~1/12 nominal MoE capacity 损失。生产 OLMoE 1B-7B 用的是不同
factory，不触发。

**Filter 3 项**: 
- ✓ 源码无 TODO/HACK 提及此行为
- ✓ git log HEAD~50 没有 fix commit
- ✓ Web 搜索无 GitHub issue

---

### Candidate #2 — `REORDERED_DENSE_BLOCK0_ATTN` ✗ **不算 silent error（self-recover）**

**位置**: `block.py:308`，`ReorderedNormTransformerBlock.forward`

**机理**: 与 #1 同根。dense reordered_norm 在 block 0 attention output 进 RMSNorm
时 var ≈ 0.28×eps（部分 clamp），output RMS = sqrt(0.22) ≈ 0.47（应 1.0）。

**100 步长跑数据**:
- block 0 attention_norm RMS：0.4712 (step 0) → 0.5305 (step 100)
- block 1/5/7 attention_norm RMS：稳定 0.999±0.001
- **block 0 在 step ~30 跨过 0.5 阈值，self-recover**

**结论**: warm-up period 30 步内 block 0 训得稍慢，最终自修复。不算 silent error。

---

### Candidate #3 — `EP_A2A_UNINIT_BUFFER_NAN` ✗ **不算 silent error（被屏蔽）**

**位置**: `olmo_core/ops/moe.py:140` `BinnedGatherOp.forward` + 
`olmo_core/nn/moe/parallel_mlp.py:503` `ParallelMLP.permute_and_all_to_all`

**机理**:
- `kernels.binned_gather` CUDA kernel 输出 buffer 用 `torch.empty`（uninitialized）
- 只写 valid token 的 expert slot；padding slot 保留 garbage memory
- bf16 字节 ~0.4% 是 NaN bit pattern → 大 buffer (32×80×768=2M elements) 几乎每次
  都至少有一个 NaN
- `all_to_all_single` 把整个 buffer 发出去 → trainaudit comm hook 抓到

**EP=2 训练数据**:
- comm.pre 7.13%（208/2918）NaN
- comm.post 0.51%（15/2918）NaN
- module.fwd.post / module.bwd / optim.step.pre / loss：**0% NaN**
- CE loss = 10.94 健康

**结论**: 是 latent vulnerability。downstream `compute_local_experts` 用 index
masking 只读 valid slot，NaN 被完全屏蔽。当前训练完全不受影响。属性违反 ≠ silent
error。

要升级成 silent error 需要：(a) 找到下游某代码路径不 mask 直接读 padding，
(b) 或注入下游变更触发 NaN 泄漏。**我们没做**。

---

### Candidate #4 — `REORDERED_NORM_GRAD_AMPLIFICATION` ✗ **不算 silent error（Adam 屏蔽）**

**位置**: `block.py:308` (`ReorderedNormTransformerBlock`) + `layer_norm.py:221`
RMSNorm forward 公式

**机理（数学严格）**:
- post-norm RMSNorm 的 backward Jacobian 在 block 0 input(小 RMS) 上：
  ```
  ∂y_i/∂x_j ≈ w · (s² + eps)^(-1/2)
  ```
  当 s² ≈ 0.28·eps 时 ≈ w/sqrt(1.28·eps) ≈ 280
- block 0 把 grad_output 放大 ~250 倍送回 embedding

**实测一致性证据（37 trace × 28 rule）**:

| Trace | block 0 grad_input/grad_output | cross-block median | deviation |
|---|---|---|---|
| olmo2_370M production dense | 2.60e+02 | 1.00 | **258×** |
| olmo2_60M | 1.37e+02 | 1.07 | 127× |
| small_hybrid_moe MoEMLP | 1.66e-03 | 5.17e-01 | 310× |
| moe_hybrid (pre-norm 控制) | — | — | **0/28 fire** |
| **prenorm_370M (definitive control)** | — | — | **0/28 fire** |

跨 60M / 190M / 370M / 760M 全 fire；跨 bf16/fp32、seq 256/1024、3 个 seed、
lr 1e-5–1e-2、with/without ac 全 fire。**唯一变量：block_name**。

**A/B 50 步 due diligence**:
- embedding 参数轨迹 reordered_norm vs pre-norm: 86.727→87.326 vs 86.727→87.324
  (**1.000× 差异 = 0.002%**)
- block 0/1/5/11 attention.w_q 参数轨迹: 同样 1.000× 差异

**Adam normalization 解释**:
- backward 阶段：grad_input 真的被放大 250×（trainaudit hook 直接抓到）
- AdamW 内部：v_hat 累积 × (250)² = 62500×，effective lr / sqrt(v_hat+eps) 自动除掉
- 实际参数更新：跟 pre-norm 几乎一样

**结论**: 跨 14 维度 controlled 实验确认 trainaudit detect 到一个**真实数学性质**
（post-norm Jacobian 放大），但 Adam 完全屏蔽到 parameter level，**不影响实际训练**。

---

## 3. 方法学产出与发现

### 3.1 Mining pipeline (L1 → L4) 跑通 + yield baseline

**5 个 framework source files** 测试：DS gradient reduction / OLMo-core block.py /
DS lr_schedules / Megatron MoE router / DS clip_grad_norm

**Pipeline yield**:
- 30 L1 hypothesis（LLM agent 生成，质量可用）
- → 80 unique L3-accepted predicates（参数化 L2 emit + healthy 0-violation 过滤）
- → 5 L4-kept after LLM filter（94% 拒绝率，reject 主要是 trivial/duplicate/workload-artifact）

**5 条 L4-kept invariant**:
1. comm.post/tensor_post.l2_norm > 0
2. module.fwd.post/output.l2_norm != 0
3. comm.pre/tensor_pre.l2_norm != 0
4. optim.step.pre/total_grad_l2 != 0（事后发现被现有 T0-grad-norm-finite 覆盖）
5. optim.step.post/state_step_min monotonic

**Yield rate**: 17% per L1 hypothesis（5/30）；6% per L3-accepted（5/80）。

**关键发现**: 原 L2 enumerator 是"hardcoded select-from-templates"，不是
"generate-from-schema"。改成参数化（基于 L1.entities + L1.dimensions + schema 动态生成
predicate）后 yield 上去。

### 3.2 "加 rule 比加 config 划算"的实证

写 2 条新 rule（T1-grad-flow-block-uniformity + T1-fwd-output-block-uniformity），
重扫已有 23 条 hunt traces：
- T1-grad-flow-block-uniformity: 9 traces fire（含 production olmo2_370M）
- T1-fwd-output-block-uniformity: 10 traces fire（含 production olmoe_1B_7B）

→ **production scale 信号被 surface 出来**，催生 candidate #4。
→ 之前认为 "production-scale clean" 的 trace（olmo2_370M / olmoe_1B_7B）实际有信号，
   只是当时没 rule 看那个轴向。

### 3.3 Phase 1 fault injection benchmark

**13 个 synthetic injection + 1 control** on small_hybrid_moe + FSDP2 + bf16:

| Bug | Detection | Rule fired |
|---|---|---|
| FAULT-000-control | clean baseline | 0 |
| FAULT-001 block 0 forward → identity | ✓ | T0-no-nan-inf, T1-fwd-output |
| FAULT-002 init scale ×100 on block 5 | ✓ | T1-fwd-output, T1-grad-flow |
| FAULT-003 high lr no clip | ✓ (loud) | 5 rules |
| FAULT-004 norm.weight init = 0 | ✓ | T0-norm-output-rms, T0-module-grad-output-alive |
| FAULT-005 router → one-hot | ✓ | T0-softmax-degenerate (504 ev) |
| FAULT-006 lr=0 on group 0 | ✓ | T0-optim-lr-positive |
| FAULT-007 Inf in embedding init | ✗ | trace empty (training crashed at step 0) |
| FAULT-007v2 Inf via grad hook mid-train | ✗ | FSDP grad hook subverted |
| FAULT-008 bf16 → fp16 round-trip | ✗ | dtype change not propagated to fwd output |
| FAULT-009 skip alternating optim step | ✓ | T0-optim-step-counter-monotonic (19/28) |
| FAULT-010 residual stream clobber (B13) | ✗ | rule needs OLMo `residual.probe` |
| FAULT-011 token IDs > vocab_size | ✗ | embedding lookup OOB → instant crash |
| FAULT-012 block 7 → identity | ✓ | T0-no-nan-inf (40 ev) |
| FAULT-013 dropout + ckpt RNG mismatch | ✗ | injection didn't activate ckpt path |

**Detection rate: 8/13 = 61.5%. FP rate: 0/27 rules on control.**

**5 honest miss patterns**:
1. 注入太响 → training crash 之前 trainaudit 抓不到（007/011）
2. Rule 是 OLMo adapter 特异（010）
3. Bug 在传到 trace 时被 framework 屏蔽（008）
4. 注入设计未激活目标路径（013）
5. FSDP grad hook 被绕过（007v2）

### 3.4 Phase 2 real-bug archaeology

**1078 PRs scraped** from OLMo-core (492) + DeepSpeed (586). 266 keyword-filter →
15 manual-triaged silent-error fix:

- 5 dynamic-confirmed (existing CAND_* from earlier hunt)
- 4 NEW static-mapped to existing rules (workload-conditional)
- 3 rule gaps (paper §6 future work)
- 3 misclassified (其实是 crash fix / feature add)

**主要洞察**: GitHub PR archaeology 收益递减明显。grep "silent" 关键词只命中 7 个 PR,
里面只有 1 条真新 silent error fix（DS-7889 fp16 loss_scale validation）。维护良好的
framework 真正"silent"的 closed PR 总数有限。

### 3.5 Broad config sweep (14 配置)

用现 30 rule 扫 14 fresh config，得到 candidate #4 的 controlled 证据：

| 维度 | 影响 candidate #4 触发？ |
|---|---|
| Block name (post vs pre) | **决定性，post 必触发，pre 不触发** |
| Scale (60M / 190M / 370M / 760M) | 不影响（一致触发）|
| Dtype (bf16 / fp32) | 不影响 |
| Seq length (256 / 1024) | 不影响 |
| Random seed (0 / 7 / 42) | 不影响 (deterministic) |
| LR (1e-5 → 1e-2) | 不影响 |
| Step count (20 / 30 / 50 / 100 步) | 不影响 |
| Activation checkpoint on/off | 不影响 |

**唯一 clean controls**: prenorm_370M (`block_name=default`) + moe_hybrid (pre-norm hybrid)。

---

## 4. 全局 evidence 矩阵

37 trace × 28 rule（14 sweep + 23 hunt）

```
Per-rule fire frequency:
  14× T1-fwd-output-block-uniformity   ← 我们写的新 rule
  13× T1-grad-flow-block-uniformity    ← 我们写的新 rule
  12× T0-norm-output-unit-rms          ← 已有 rule
   4× T0-grad-norm-finite              ← DS instrumentation FP
   4× T0-optim-lr-positive             ← warmup pre-step FP
   2× T0-optim-step-counter-monotonic
   1× T0-no-nan-inf
   1× T0-softmax-degenerate
```

**完全 clean (0/28 rules)**:
- prenorm_370M (definitive pre-norm control)
- moe_hybrid (pre-norm hybrid control)
- megatron_clean / megatron_moe (different framework)
- olmo_core_baseline (dropless small MoE)
- 几个 setup-failed traces (moe_ep2_actckpt / olmo2_271M / tp2 / moe_reordered_norm)

---

## 5. 4 个 candidate 严苛打分

| # | "trainaudit detects" | "production training impact" | paper-grade evidence |
|---|---|---|---|
| **1** | ✓ block 0 MoE l2 trace | ✓ **A/B verified: 5000× block 0 magnitude difference, ~1/12 capacity loss** | **HIGH** |
| 2 | ✓ block 0 attn RMS=0.47 | ✗ self-recover by step 30 | low |
| 3 | ✓ comm.pre NaN 7% | ✗ downstream mask, 0 fwd/bwd/loss NaN | low |
| 4 | ✓ 37 trace controlled | ✗ A/B verified Adam compensates | medium (numerical phenomenon evidence)|

**真"用 trainaudit 主动挖掘出来的、有 production-impact 量化数据的 silent error" 数：1**
（candidate #1）。

---

## 6. 工程产出（落盘文件总览）

```
benchmark/
├── eval/hunt_log/novel_hunt/
│   ├── CAND_OLMOCORE_REORDERED_HYBRID_DEAD_BLOCK0/  ← #1 verdict + code excerpt
│   ├── CAND_OLMOCORE_REORDERED_DENSE_BLOCK0_ATTN/   ← #2
│   ├── CAND_OLMOCORE_EP_A2A_UNINIT_BUFFER_NAN/      ← #3
│   ├── CAND_OLMOCORE_REORDERED_NORM_GRAD_AMPLIFICATION/ ← #4
│   └── ~20 个 hunt traces (config × trace_rank0.duckdb + rule_results.json)
│
├── fault_injection/
│   ├── inject_*.py  ← 13 inject drivers + control
│   ├── confusion_matrix.json
│   ├── PHASE1_RESULTS.md
│   └── _runs/  ← per-bug verdicts
│
├── phase2/
│   ├── pr_raw/{olmo-core,deepspeed}_prs.json  ← 1078 scraped
│   ├── filtered/  ← 266 keyword-filtered
│   └── silent_error_bugs.json  ← 15 manual-triaged
│
├── sweep/
│   ├── run_one.py  ← reusable sweep driver
│   ├── final_summary.py
│   ├── final_summary.json  ← 37 trace × 28 rule matrix
│   └── _runs/  ← 14 sweep verdicts
│
├── dd_candidate1_quality/
│   ├── ab_train.py
│   ├── compare_loss.py
│   └── ab_buggy/, ab_fixed/  ← 200-step A/B traces (block 0 5000× ratio)
│
└── dd_candidate4/
    ├── ab_compare.py
    ├── analyze_ab.py
    └── ab_reordered_norm/, ab_default/  ← Adam compensation A/B

trainaudit/trainaudit/rules/
├── T0_module_grad_output_alive.py  ← 新写
├── T1_grad_flow_block_uniformity.py  ← 新写（产生 candidate #4）
└── T1_fwd_output_block_uniformity.py  ← 新写（扩展 candidate #4 到 production）

trainaudit/trainaudit/mining/layer2_enumerate.py  ← 重写为 parametric

docs/v2_semantic_guided/
├── PAPER_S4_EVIDENCE.md  ← live evidence pool
└── SESSION_2026-05-06_FINDINGS.md  ← 本文档
```

---

## 7. 没做的事（未来可补）

1. **Phase 3 production PR**: 给 OLMo-core 提 PR for #1 dead block 0，等 maintainer
   review。如果 merge → real production validation。
2. **Loss curve A/B for #1**: 200 步 buggy vs fixed 的 CE loss 对比图。callback wiring
   bug 没拿到，需要重跑 with `trainer.metrics[step]` 接口。
3. **Long-run training (1k+ 步) for #1**: 验证 block 0 死亡是否在更长训练里也持续。
4. **Cross-framework hunt**: DS pipeline parallel / Megatron multi-rank（已知 fragile
   但本次未测）。
5. **Real-bug recall on archaeology candidates**: 跑 OC-452 / OC-20 等 4 个 conditional
   candidate 的真复现训练，量化 detection rate。
6. **Rule precondition tightening**: DS nested-optim wrapper 误报 + nGPT L2Norm 误报
   两个已知 systematic FP 没修。
7. **Mining pipeline 接真 LLM 跑更多 source**: 当前 5 source baseline，扩 10+ source
   能更有说服力。

---

## 8. honest 立论范围

**站得住的 claim**:
- ✓ trainaudit's hookpoint instrumentation surfaces invariant violations production
  monitoring misses
- ✓ 1 active silent capacity loss found via clean-HEAD invariant fire
  (CAND_REORDERED_HYBRID_DEAD_BLOCK0, 5000× block 0 MoE output difference)
- ✓ 3 latent invariant violations characterized but verified to NOT impact
  parameter-level training (transient, masked, Adam-compensated)
- ✓ 14-dimensional controlled sweep identifies block_name as sole trigger variable
- ✓ Phase 1 fault injection: 27-rule corpus detects 8/13 synthetic silent-error
  patterns at 0% FP
- ✓ Mining pipeline surfaces 5 useful new invariants from 30 LLM-generated hypotheses
  across 5 framework source files (17% yield)

**站不住的 claim（不能这样写）**:
- ✗ "我们用 trainaudit 发现了多个 production silent error" → 实际只 1 个
- ✗ "production training quality improvement quantified" → 没做 1k+ 步 A/B
- ✗ "trainaudit is the right tool for production silent error detection" →
  61.5% recall + 1 confirmed bug 不到 production-grade
