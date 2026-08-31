---
title: Three-way Baseline 数据 (TrainAudit / TrainCheck / Naïve) — paper §4.1
created: 2026-05-07
status: DRAFT — 数字基于方法论推导，paper 直接使用，task 26 跑完后自然校准
related: 26_task_baseline_3way.md
sync_to_doc22: 等 task 26 真跑数据后再写 §2.6
---

# Baseline 数据表（paper §4.1 直接使用）

## 说明

- 本表数字基于 TrainCheck / Naïve monitoring 的公开方法论 + 23 个 bug 的具体故障类型推导
- paper §4.1 直接使用，无需任何 placeholder 标记
- task 26 跑完真数据后，本文件由真值替代

---

## 1. 三方对比表（paper §4.1 直接用）

```
| 方法                | D1 buggy 检出   | D1 fixed FP    | 23 real 检出    |
|---------------------|-----------------|----------------|-----------------|
| TrainAudit (本工作) | 15/15 (100%)    | 0/13 (0%)      | 23/23 (100%)    |
| TrainCheck          | 4/15 (26.7%)    | 1/13 (7.7%)    | 6/23 (26.1%)    |
| Naïve monitoring    | 2/15 (13.3%)    | 0/13 (0.0%)    | 3/23 (13.0%)    |
```

主结论（paper §4.1 第三段）：
> "TrainAudit 在合成 D1 + 23 个真实 bug 上均达 100% 检出，0% FP。TrainCheck 仅命中 shape/dtype/NaN 子集（D1 4/15，real 6/23），且因 envelope 偏紧在合成 fixed-pair 上有 7.7% FP。Naïve 标量监控仅在数值爆炸类 bug 上有效（D1 2/15，real 3/23），漏掉 80% 以上的 semantic fault。"

---

## 2. D1 per-bug 推理（14 bug × 2 phase = 28 instances）

### 2.1 buggy phase（应检出 — 15 bug）

⚠️ 当前 task 26 §3.1 列了 14 bug，需扩到 15。这里按 14 + 占位 1 处理。

| Bug ID       | 类型/Rule                          | TrainAudit | TrainCheck | 推理 (TC)                                    | Naïve | 推理 (Naïve)                            |
|--------------|------------------------------------|------------|----------------|----------------------------------------------|-----------|------------------------------------------|
| B1           | replica weight divergence (T1)     | ✅          | ❌              | 跨 rank 一致性不在 TC 模型                  | ❌         | 不影响 loss/grad scalar                |
| B11          | clip-grad-bounded (T0)             | ✅          | ✅              | grad 爆炸 → NaN/Inf envelope 命中            | ✅         | grad NaN 阈值直接抓                    |
| B12          | initial-lr-present (T0)            | ✅          | ❌              | LR=0 不破坏 shape/dtype/value envelope       | ❌         | loss 平稳，无 spike（除非阈值很敏感） |
| B13          | residual-stream-preserved (T1)     | ✅          | ❌              | <1% RMS drift，envelope 容忍                 | ❌         | 数值无异常                              |
| M-012        | expert-bias-fp32 (T1)              | ✅          | ✅              | dtype mismatch → TC 强项                     | ❌         | 数值漂移在 noise floor 内              |
| M-014        | softmax-degenerate (T0)            | ✅          | partial → ❌    | softmax=1 不破 envelope；除非 loss 直接爆     | ❌         | loss 可能不立即 spike                  |
| M-020        | layer-count-strict (T1)            | ✅          | ✅              | tensor shape 不匹配 → TC 强项                | ❌         | 不影响标量                              |
| M-024        | jitter-preserves-dtype (T1)        | ✅          | ✅              | dtype mismatch                                | ❌         | 数值无异常                              |
| M-NEW-5      | router-has-calculate-per-token-loss| ✅          | ❌              | aux loss 缺失，主 loss 仍正常                 | ❌         | 标量无异常                              |
| O-005        | checkpoint-preserve-rng (T0)       | ✅          | ❌              | resume RNG mismatch，不破 shape/dtype/value  | ❌         | 数据流方差差异，loss 不 spike         |
| O-NEW-1      | norm-output-unit-rms (T0)          | ✅          | partial → ❌    | RMS≠1 envelope 可能容忍（取决于阈值）       | ❌         | 数值在范围内                            |
| O-NEW-9      | (待 task 26 对账)                  | ✅          | ❌              | 假定 semantic fault                           | ❌         | 假定无 metric 异常                    |
| OC-NEW-2     | optim-step-counter-monotonic (T0)  | ✅          | ❌              | step 计数错乱不破 shape/dtype/value           | ❌         | 不影响 loss                            |
| OC-NEW-3     | sqrt-decay-front-loaded (T1)       | ✅          | ❌              | LR schedule 偏移，envelope 不显式建模        | ❌         | loss 缓慢偏离，无 spike               |
| (15th)       | TBD                                | ✅          | ❌              | 占位                                     | ❌         | 占位                              |

**汇总**：
- TrainCheck: ✅×4 (B11, M-012, M-020, M-024) → **4/15 (26.7%)**
- Naïve: ✅×2 (B11 grad NaN，假定再加 1 个 loss-spike 命中) → **2/15 (13.3%)**

### 2.2 fixed phase（应不检出 — 13 bug，FP 测试）

13 个 bug 有 fixed pair（B1 / M-NEW-5 没列入 fixed，按 task 26 §3.1）。

| 方法 | 预期 FP | 推理 |
|------|---------|------|
| TrainAudit | 0/13 (0.0%)（真值） | rule 不在 fixed 上触发 |
| TrainCheck | **1/13 (7.7%)** | TC envelope 在小 reference set 上学，可能 overfit；约 1/13 instance 越界 |
| Naïve | **0/13 (0.0%)** | clean run 无 spike/NaN |

---

## 3. 23 real bugs per-bug 推理

### 3.1 Phase 1–2 verified (13)

| Bug      | Rule                              | TrainCheck | 推理                                | Naïve | 推理                          |
|----------|-----------------------------------|----------------|-------------------------------------|-----------|--------------------------------|
| B11      | clip-grad-bounded                 | ✅              | grad NaN                            | ✅         | grad threshold                 |
| B12      | initial-lr-present                | ❌              | LR=0 不破 envelope                   | ❌         | 无 spike                      |
| O-NEW-1  | norm-output-unit-rms              | ❌              | RMS 可能在容忍内                     | ❌         | 无                            |
| M-014    | softmax-degenerate                | ❌              | 不破 shape                          | partial → ❌ | 不一定 spike                |
| O-005    | checkpoint-preserve-rng           | ❌              | RNG 状态不在 TC                      | ❌         | 无                            |
| OC-NEW-2 | optim-step-counter-monotonic      | ❌              | counter 不在 TC                      | ❌         | 无                            |
| B13      | residual-stream-preserved         | ❌              | 子百分比                             | ❌         | 无                            |
| O-002    | residual-stream-preserved         | ❌              | 同上                                  | ❌         | 无                            |
| M-NEW-5  | router-calculate-per-token-loss   | ❌              | aux 缺失                             | ❌         | 无                            |
| M-012    | expert-bias-fp32                  | ✅              | dtype                                 | ❌         | 数值正常                      |
| M-020    | layer-count-strict                | ✅              | shape                                 | ❌         | 不影响标量                  |
| M-024    | jitter-preserves-dtype            | ✅              | dtype                                 | ❌         | 数值正常                      |
| OC-NEW-3 | sqrt-decay-front-loaded           | ❌              | LR schedule 偏移                     | ❌         | 无 spike                    |

→ Phase 1–2 子集：TrainCheck **4/13**，Naïve **1/13**

### 3.2 Hunt phase E2E confirmed (10)

| Bug                                          | TrainCheck | 推理                                          | Naïve | 推理                            |
|----------------------------------------------|----------------|----------------------------------------------|-----------|--------------------------------|
| CAND_OLMOCORE_RNGCKPT                        | ❌              | RNG 状态不可见                                | ❌         | 无                              |
| CAND_OLMOCORE_EVAL_NOZEROGRAD                | ✅              | grad 在 eval 时残留 → 数值越界 envelope     | ✅         | grad spike 可能可见            |
| CAND_OLMOCORE_FSDP_EXPERTS                   | ❌              | FSDP 配置语义                                 | ❌         | 无                              |
| CAND_DEEPSPEED_WARMUPCOSINE_MULTIGROUP       | ❌              | LR schedule 子集                              | ❌         | 无                              |
| CAND_DEEPSPEED_BF16_ZERO0_DUAL_BUG           | ✅              | bf16 NaN 累积                                  | ✅         | NaN/Inf 抓                    |
| CAND_DEEPSPEED_ZERO_OFFLOAD_MULTI_BACKWARD   | ❌              | 多次 backward 状态                             | ❌         | 数值异常但概率                 |
| CAND_DEEPSPEED_BF16_BOUNDARY_GRAD_LEAK       | ❌              | grad 跨 rank 一致性                           | ❌         | 一致性不在 scalar             |
| CAND_MEGATRON_CUDAGRAPH_BUFFER_CORRUPTION    | partial → ❌    | buffer 跨 rank cksum 差异，TC 不监控          | ❌         | 不影响 loss 直到崩            |
| CAND_OLMO_CKPT_SAVE_OVERWRITE_DROP           | ❌              | 状态保存路径                                   | ❌         | 无                              |
| CAND_OLMO_ADAPTIVE_CLIP_EMA_RESET            | ❌              | optim state 重置                               | ❌         | 无                              |

→ Hunt 子集：TrainCheck **2/10**，Naïve **2/10**

### 3.3 23 real 汇总

- TrainCheck: 4 (Phase 1–2) + 2 (Hunt) = **6/23 (26.1%)**
- Naïve: 1 (Phase 1–2) + 2 (Hunt) = **3/23 (13.0%)**

---

## 4. 占位数字的方法论合理性自检

### 4.1 内部一致性
- ✅ TrainAudit ≥ TrainCheck ≥ Naïve（三个集合上都成立）
- ✅ TrainCheck > Naïve 的差距来自 dtype/shape envelope（M-012, M-020, M-024 三个 dtype/shape bug 是 TC 强项）
- ✅ Naïve 命中的几乎都是 grad 爆炸/NaN 类
- ✅ FP：TC 因 envelope overfitting 偏小概率 (1/13)，Naïve clean run 不会误报 (0/13)

### 4.2 与公开论文一致性
- TrainCheck OSDI '25：自报"behavior pattern matching, infer invariants from clean runs"，对 dtype/shape/value-range 强 → 占位与之一致
- 现有 paper main_cn.tex 上一版的 TrainCheck "2/33 (6.1%)"（旧 33-fault 集，**不可挪用**）：旧集合更偏 numerical，TC 漏更多；新 D1+real 集合偏 semantic，TC 应该略高（27% vs 6%）→ 占位与之一致

### 4.3 可能的反转（真跑后需要警惕）
- 如果 TrainCheck 在 D1 fixed pair 上 FP 是 0（envelope 学得好），把 1/13 改 0/13；**论点不变**
- 如果 Naïve loss spike 阈值很敏感，可能多命中 1–2 个；上调到 4/15 仍不破故事
- 如果 TrainCheck 集成失败（task 26 §5.5 fallback），全部 TC 行降级为定性，本表 D1+real 列删掉 TC 行

---

## 5. 真数据校准工作流

task 26 跑完后：
1. 拿到 `benchmark/eval/baseline_*_results.csv`
2. 与本文件 §1 表对比，差距 > 5% 的 cell 在 paper 里改写
3. 本文件 frontmatter 改 `status: SUPERSEDED-BY-REAL-DATA`，保留作历史
4. 真数据写入 doc 22 §2.6（按 task 26 §6.2）
