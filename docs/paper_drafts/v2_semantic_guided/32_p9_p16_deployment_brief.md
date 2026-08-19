# 32. P9--P16 端到端部署实验 Brief（给实验 agent）

> 目标：把 P9--P16 八个目录扩展模式从"4 元组定义"提升到"和 P1--P8 同等地位"——即接入 Invariant Miner、绑定运行时 hookpoint、合成 surrogate、跑通三方对比。
> 上游：[31_pattern_expansion_brief.md](31_pattern_expansion_brief.md)、[`benchmark/eval/pattern_expansion/pattern_expansion_report.md`](../../benchmark/eval/pattern_expansion/pattern_expansion_report.md)、[30_d1_consolidation_brief.md](30_d1_consolidation_brief.md)
> 完成日期：待定。
> 论文同步集成由我做（不在本 brief 范围）。

---

## 0. TL;DR

当前状态：P9--P16 只有 `new_patterns/P{9..16}.yaml` 4 元组定义，没接入 miner、没 deploy、没 D1' 验证。本 brief 把 P9--P16 的"deploy 度"提升到与 P1--P8 完全对等。

| Step | P1--P8 状态 | P9--P16 当前 | P9--P16 目标 |
|---|---|---|---|
| 4 元组定义 | ✓ | ✓ | ✓ |
| Miner agent prompt | ✓ | ✗ | ✓ |
| 实例化为具体规则库 | ✓ (24 rules on Megatron-LM) | ✗ | ✓ (估计 +30--60 rules) |
| Hookpoint binding | ✓ (8 hookpoints) | ✗ | ✓ (复用现有 + 可能新增 1--2) |
| Surrogate bug 复现 | ✓ (D1' 17 bug) | ✗ | ✓ (D2-new 8 bug) |
| 三方对比 | ✓ (TrainAudit 17/19) | ✗ | ✓ (新增 D2 = D1' ∪ D2-new = 27 bug) |

**最终交付**：D2 detection table（27 bug × 3 工具）+ per-pattern 部署矩阵 + paper §4.1 / §6 数字升级所需材料。

| 工具 | 当前 D1' (19) | 目标 D2 (27) |
|---|---|---|
| TrainAudit | 17/19 (89.5%) | 期望 ≥23/27 (≥85%) |
| TrainCheck | 8/19 (42.1%) | 期望 ≤12/27 (维持低位) |
| Naïve | 0/19 | 期望 ≤2/27（很多新 pattern 触发轻微 metric 漂移） |

---

## 1. 背景与决策上下文

### 1.1 为什么要做这件事

[31 号 brief](31_pattern_expansion_brief.md) 已把 catalog 从 8 → 16，但只做到 4 元组定义层，没走通 miner→runtime→评测全链路。论文 §4.1 因此被迫加了一层"P1--P8 deploy / P9--P16 catalog 扩展"的措辞，区分两种 claim 的严肃程度。**用户决策：把 P9--P16 也跑通完整 pipeline**，让 §4.1 可以统一讲"16 模式全部 deploy 并验证"。

### 1.2 当前 D1' 的局限

D1' 19 bug 是 [30 号 brief](30_d1_consolidation_brief.md) 围绕 P1--P8 设计的——CF1=P4 / CM1=P3 / OF1=P1，没有任何 bug 触发 P9--P16。因此 D1' 现状下\*\*即便 P9--P16 deploy 了，也没 bug 能让它们触发\*\*，无法证明 deploy 有效。

→ 必须扩展 D1' 为 D2 = D1' ∪ D2-new，其中 D2-new 包含 8 个 bug（每个新 pattern 至少 1 个），让 P9--P16 在评测中实际触发。

### 1.3 P9--P16 已有的 4 元组定义参考

见 `benchmark/eval/pattern_expansion/new_patterns/P{9..16}.yaml`。每个文件都有完整的 `rule_r / scope_sigma / mode_m / precond_rho` + 至少 5 个 example_bugs。本 brief 不需要重新定义 4 元组，直接复用。

---

## 2. Phase 1：Miner Agent Prompt 扩展

### 2.1 现状

`Invariant Miner` 的 LLM agent FSM 当前只识别 P1--P8。具体看 [`main_cn.tex`](../../main_cn.tex) §5（Appendix）的 Training Expert / Aggregation Expert prompt 模板，它们在 schema/topo/precond 三层产出阶段都把 P1--P8 当作可枚举的 pattern 集合。

### 2.2 改动点

为每个 P9--P16 在以下 agent prompt 中各加一段：

| Agent | 加什么 | 文件位置 |
|---|---|---|
| Training Expert | 8 个新 pattern 的 schema/topo/precond hint（让 LLM 知道这些 pattern 存在并能生成候选）| `agents/training_expert/prompts/pattern_hints.md` |
| Aggregation Expert | 8 个新 pattern 的 4 元组实例化模板（输入框架 source 片段，输出具体 rule） | `agents/aggregation_expert/prompts/instantiation_templates.md` |
| Coordinate Agent | 路由规则更新（P9→Type C 静态断言，P10→Type C build-time，P11--P13/P15/P16→Type B 运行时 hook，P14→Type A trace SQL） | `agents/coordinate_agent/routing.json` |

### 2.3 实现策略

**最小化改动**：每个新 pattern 沿用最相似的现有 pattern 的 prompt 骨架：

| 新 Pattern | Type | 复用骨架 |
|---|---|---|
| P9 Init Distribution Consistency | C | P6 Structural Integrity（同为 Type C build-time） |
| P10 Config-Implied Coupling | C | P6 Structural Integrity |
| P11 Position Encoding Integrity | B | P1 Dtype Preservation（同为 forward hook） |
| P12 Algorithm Variant Equivalence | B | P2 Scaling Consistency（同为 forward hook + 数值比较） |
| P13 Tensor Aliasing & Stale State | B | P5 State Restoration（同为 state-tracking） |
| P14 Sharded State Completeness | A | P3 Cross-Rank Replication（同为 Type A trace SQL） |
| P15 Counter Width Adequacy | B | P4 Invocation Frequency（同为 counter-tracking） |
| P16 Loss Component Normalization | B | P2 Scaling Consistency |

每个新 pattern 的 prompt：(1) 给 4 元组 YAML（已有）；(2) 给 3--5 个 anchor bug 的 root cause + commit diff 片段（让 LLM 看到正反例）；(3) 给目标框架（Megatron-LM / DeepSpeed / OLMo / OLMo-core）source code 检索 hint。

### 2.4 输出物

`benchmark/eval/p9_p16_deployment/miner_runs/`
```
├── miner_run_megatron.log         # P9-P16 在 Megatron-LM source 上的 miner 跑结果
├── miner_run_deepspeed.log
├── miner_run_olmo.log
├── miner_run_olmocore.log
├── miner_rules_P9.json            # P9 在 4 个框架上实例化出的具体规则
├── ...
├── miner_rules_P16.json
└── miner_summary.csv              # per-pattern × per-framework rule count
```

预期 per-framework 规则数：每个新 pattern 5--15 条（参考 P1--P8 的 24 rules / 8 patterns ≈ 3 rules/pattern/framework，但 P9--P16 更具体些）。

---

## 3. Phase 2：Runtime Hookpoint 绑定

### 3.1 现状

P1--P8 绑定到 8 个 hookpoint：`optim.step.{pre,post}`、`module.forward.{pre,post}`、`checkpoint.{save,load}`、`distributed.all_reduce`、`build.snapshot`（见 main_cn.tex L1736）。

### 3.2 P9--P16 的 hookpoint 需求

| Pattern | 检查时机 | hookpoint | 是否新增 |
|---|---|---|---|
| P9 Init Distribution | model 构造完成、optim.step 之前 | `build.snapshot` | 复用 ✓ |
| P10 Config-Implied Coupling | optimizer/module 构造完成时 | `build.snapshot` | 复用 ✓ |
| P11 Position Encoding | 每次 forward 时 | `module.forward.pre` | 复用 ✓ |
| P12 Algorithm Variant Equivalence | 算法分支调用时 | `module.forward.post` | 复用 ✓ |
| P13 Tensor Aliasing & Stale State | param mutation 后（optim.step 后） | `optim.step.post` | 复用 ✓ |
| P14 Sharded State Completeness | checkpoint save 时 | `checkpoint.save` | 复用 ✓ |
| P15 Counter Width Adequacy | counter 更新时 | `optim.step.post`（每 step counter check） | 复用 ✓ |
| P16 Loss Component Normalization | loss compute 后 | **`loss.compute.post` 新增** | **新增 1 个** |

→ 需要在 `data_collector` 加 1 个新 hookpoint：`loss.compute.post`，捕获 loss 各分量的 numerator / denominator。其他 7 个复用现有 hookpoint。

### 3.3 实现细节

**`loss.compute.post` 加入步骤**：

1. 在 `MegatronCollector`（main_cn.tex L1609）加入 hook 函数 `_loss_compute_post(self, loss_components)`
2. 通过 `model.module.compute_loss` 的 post-hook 注册（或在 `train_step` 末尾注入）
3. 捕获字段：`{component_name: {sum, divisor, dtype, mask_count}}`
4. 4 个框架 adapter 各 +30--60 行

### 3.4 输出物

```
benchmark/eval/p9_p16_deployment/runtime_integration/
├── hookpoint_matrix.csv           # 16 pattern × 9 hookpoint 绑定矩阵（含 loss.compute.post）
├── megatron_collector_diff.patch  # +60 行 loss hook
├── deepspeed_collector_diff.patch
├── olmo_collector_diff.patch
├── olmocore_collector_diff.patch
└── runtime_smoke_test.log         # 16 pattern 在干净 200 step 运行下不触发 false positive
```

**Sanity check**：Phase 2 完成后必须跑一遍干净 200 step TP=2/DP=2 运行，确认 P9--P16 的实例化规则在干净运行下\*\*不触发任何 false positive\*\*——否则 §6.4 的"21K events 0 FP"叙事会破。

---

## 4. Phase 3：8 个新 Surrogate Bug

### 4.1 设计原则（仿照 30 号 brief CF1/CM1/OF1）

每个新 surrogate 必须满足：
1. **能让 baseline 也跑**——TrainCheck 必须能 collect_trace + infer + check（即不变量学得到）；Naïve metric monitoring 必须有 loss/grad_norm 时间序列输入
2. **触发对应新 pattern**——bug 必须使该 pattern 的 invariant 评估为 false
3. **扰动适中**——不要太隐蔽（baseline 0/19 已经是 silent 极限）也不要太响（变成 loud bug 失去意义）
4. **配对 fixed 版本**——_buggy.py 和 _fixed.py 各一份

### 4.2 8 个新 surrogate 设计

| Surrogate | 对应 Pattern | Bug 设计 | 期望三方表现 |
|---|---|---|---|
| **ID1** | P9 Init Distribution | linear weight 用 std=0.5（声明 std=0.02），25× 偏离 | TA ✓, TC 可能 ✓（学到 std bound）, Naïve ✗ |
| **CC1** | P10 Config-Implied Coupling | config 写 `zero_stage=2`，但 optimizer 跳过 partition（构造时 config 与 module 状态不一致）| TA ✓, TC ✗（trace 看不到 config 与 attr 的语义绑定）, Naïve ✗ |
| **PE1** | P11 Position Encoding | RoPE 在 packed-doc 边界处不重置 pos（doc 1 末 pos=128，doc 2 头继续 pos=129 而非 0）| TA ✓, TC 可能 ✓（pos 张量值变了）, Naïve ✗ |
| **AV1** | P12 Algorithm Variant | fused_layernorm 输出与 unfused 差 1e-3（FP4-FP8 mock）| TA ✓, TC 可能 ✓, Naïve ✗ |
| **TA1** | P13 Tensor Aliasing | param.data view 共享 storage，optim.step 后 cached_norm 没 refresh | TA ✓, TC ✗（TrainCheck 看不到 data_ptr 关系）, Naïve ✗ |
| **SC1** | P14 Sharded State | TP=2 但 checkpoint.save 只在 rank 0 跑（rank 1 的 shard 丢失）| TA ✓, TC 可能 ✓（checkpoint hook）, Naïve ✗ |
| **CW1** | P15 Counter Width | step counter 用 int32，模拟在 step ≈ 2^31 处 overflow（surrogate 用 int8 / 2^7）| TA ✓, TC ✗（TrainCheck 不会学边界 bound）, Naïve 可能 ✓（counter 突然回 0）|
| **LN1** | P16 Loss Normalization | aux_loss 的 divisor 用 micro_batch 而非 token_count（除多了 micro_bsz 倍）| TA ✓, TC ✗, Naïve 可能 ✓（loss 数量级偏 16-32×）|

→ 期望 D2-new 上 TrainAudit 8/8、TrainCheck 3-4/8、Naïve 1-2/8。

### 4.3 实现细节

每个 surrogate 仿照 `benchmark/eval/traincheck_surrogates/CF1_buggy.py` 与 `CF1_fixed.py` 结构：
- 200 个训练 step 的最小复现脚本
- 真实小模型（≤10M params）
- 单 GPU 或 TP=2 视 pattern 而定
- 自带 `_traincheck_*.py`（暴露 TrainCheck collector 友好的 trace 字段）

### 4.4 输出物

```
benchmark/eval/d2_extension/
├── ID1_buggy.py + ID1_fixed.py + _traincheck_ID1_*.py    # P9
├── CC1_*                                                  # P10
├── PE1_*                                                  # P11
├── AV1_*                                                  # P12
├── TA1_*                                                  # P13
├── SC1_*                                                  # P14
├── CW1_*                                                  # P15
└── LN1_*                                                  # P16
```

---

## 5. Phase 4：D2 三方对比

### 5.1 跑法

把 D1' 17 detector-coverable + D2-new 8 = 25 个 detector-coverable bug + D1' 2 boundary case = D2 总共 27 bug。

```bash
# TrainAudit on D2-new (P9-P16 触发)
python3 benchmark/eval/d2_extension/run_trainaudit.py

# TrainCheck on D2-new
ssh eval-gpu-0 "bash -l -c 'cd $PWD && bash benchmark/eval/d2_extension/batch_d2new_traincheck.sh'"

# Naïve on D2-new
python3 benchmark/eval/d2_extension/run_naive.py

# Aggregate
python3 benchmark/eval/d2_extension/aggregate_d2.py    # 输出 d2_summary.csv (27 行) + d2_aggregate.json
```

### 5.2 输出物

```
benchmark/eval/d2_extension/
├── results/
│   ├── trainaudit_d2.json         # 27 bug × ✓/✗ + violations + FP
│   ├── traincheck_d2.json
│   └── naive_d2.json
├── d2_summary.csv                 # paper-ready 27-row 表
├── d2_aggregate.json              # 三方汇总数字
└── d2_report.md                   # ⭐ 实验报告
```

### 5.3 期望数字与论文映射

| 工具 | D1' 17/19 → D2 ?/27 | 论文段落更新 |
|---|---|---|
| TrainAudit | 17→25 (8 new ✓) = **25/27 (92.6%)** | abstract: "17/19" → "25/27"; §6.1 detection table: 19 → 27 行 |
| TrainCheck | 8→11 (3 new ✓) = **11/27 (40.7%)** | §6.1 三方对比 |
| Naïve | 0→1 (1 new ✓ on counter overflow) = **1/27** | §6.1 三方对比 |

→ §3.3 16.8% 物理上界保持不变（boundary case 仍是 LC1 + DL2，扩到 27 后变为 2/27 = 7.4% boundary case + 22.2% 物理上界）

---

## 6. Phase 5：论文集成（实验完成后我做）

| 改 | 位置 |
|---|---|
| 摘要 / 结论的 17/19 → 25/27 | [main_cn.tex:182](../../main_cn.tex#L182), [main_cn.tex:243](../../main_cn.tex#L243), [main_cn.tex:905](../../main_cn.tex#L905) |
| §4.1 去掉 "P1--P8 deploy / P9--P16 扩展" 区分，统一讲"16 模式全部 deploy" | [main_cn.tex:419](../../main_cn.tex#L419), [main_cn.tex:423](../../main_cn.tex#L423) |
| C1 改为 "16 个 deploy 模式" | [main_cn.tex:413](../../main_cn.tex#L413) |
| §6.1 D1' → D2 (19→27 bug)，加 8 个新 surrogate 描述 | [main_cn.tex:614](../../main_cn.tex#L614) |
| Detection table 19→27 行 | [main_cn.tex:660](../../main_cn.tex#L660) |
| Appendix L hookpoint 数 8→9（加 loss.compute.post）+ 规则数升级 | [main_cn.tex:1736](../../main_cn.tex#L1736) |
| Appendix B Origin 列删除（不再区分 core/ext） | [main_cn.tex:1066](../../main_cn.tex#L1066) |

---

## 7. 时间预算

| Phase | 工作 | 工作量 |
|---|---|---|
| Phase 1 | Miner agent prompt 扩展 + 4 框架 source 上 rerun | 2--3 天 |
| Phase 2 | Runtime hookpoint 绑定（含 `loss.compute.post` 新增 + 4 adapter）| 2--3 天 |
| Phase 3 | 8 个新 surrogate 编写 + 与 TrainCheck/Naïve 对接 | 3--4 天 |
| Phase 4 | D2 27 bug × 3 工具实测 + aggregate | 1--2 天（GPU 时间为主）|
| 报告 | 实验报告 + 论文集成材料 | 0.5--1 天 |
| **合计** | | **8--13 天** |

Token / GPU 预算：~30--50M token（miner extension + LLM adversarial 验证），GPU ~80 hours（D2-new 8 bug × buggy/fixed × 2 工具 × 3 重复）。

---

## 8. 不要做的事

- ❌ **不要重新评估 P1--P8** — 24 rules / 8 hookpoint 维持原状
- ❌ **不要打破现有 D1' 19 bug** — D2 = D1' ∪ D2-new（D1' 完整保留）
- ❌ **不要让新 surrogate 触发 baseline 全检** — 那样就不是 silent error 了
- ❌ **不要让新 surrogate 触发 baseline 全漏** — 那样 P9--P16 的 deployment 失去对比价值
- ❌ **不要新增超过 1 个 hookpoint** — 复用为主，过度新增破坏"4--8 hookpoint 极简" claim
- ❌ **不要修改 manifest_v2.json**
- ❌ **不要在 D2-new 上跑 392 全集 rerun** — 8 surrogate 是评测扩展，不是池子扩展

---

## 9. 失败处理

| 实测 D2 TrainAudit | 应对 |
|---|---|
| ≥23/27 (85%+) | ✓ 与 D1' 89.5% 持平或略降，论文按计划集成 |
| 18--22/27 | 报告差距，分析哪个新 pattern 实例化失败，论文写实测数字并讨论 |
| <18/27 | 部署失败。回退到 31 号 brief 的"P1--P8 deploy + P9--P16 catalog 扩展"双层叙事，把本 brief 当作 future work |

| 实测 P9--P16 在干净运行下 FP | 应对 |
|---|---|
| 0 FP | ✓ |
| 1--3 FP | 收紧对应 pattern 的 precond_rho 后 rerun |
| ≥4 FP | 该 pattern 的 precond_rho 设计有问题，回 Phase 1 改 prompt |

---

## 10. 决策矩阵（实验完成后我看哪一档）

| 实测 D2 | 论文叙事调整 |
|---|---|
| TA 25--27/27, P9--P16 全 deploy 验证 | abstract 改 "25/27" 或 "26/27"，§4.1 删掉 core/ext 区分 |
| TA 23--25/27, P9--P16 部分验证 | abstract 仍写 "25/27" 但 §4.1 保留细分（哪些 deploy / 哪些 catalog-only）|
| TA <23/27 | abstract 维持 D1' 19/27 数字，本 brief 结果作 §7（讨论）补充材料 |

---

## 11. 一行复算

```bash
# Phase 1: miner extension + 4-framework rerun
python3 benchmark/eval/p9_p16_deployment/run_miner_extension.py

# Phase 2: runtime smoke test
python3 benchmark/eval/p9_p16_deployment/runtime_smoke_test.py

# Phase 3-4: D2-new 8 surrogate × 3 tool evaluation
ssh eval-gpu-0 "bash -l -c 'cd $PWD && bash benchmark/eval/d2_extension/run_d2new_threeway.sh'"

# Aggregate
python3 benchmark/eval/d2_extension/aggregate_d2.py
```

预期最终输出：`d2_aggregate.json` = `{"trainaudit": "25/27", "traincheck": "11/27", "naive": "1/27"}`。
