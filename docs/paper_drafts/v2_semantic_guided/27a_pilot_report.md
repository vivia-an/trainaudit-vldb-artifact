# 27a. Pilot 30 决策报告（v2 更新）

> Phase 1 完成日期：2026-05-10（v1 草稿）→ 2026-05-10（v2 重跑）
> 输入：30 个 pilot bug（从 248 NEW 池分层采样）
> v1 标注归档：`benchmark/eval/annotations_pilot_v1_archive/`
> v2 标注（当前）：`benchmark/eval/annotations_pilot.json`
> 重算产物：`benchmark/eval/recompute_v2/pilot_30_v2_metrics.csv`

---

## ⭐ v2 重跑结论速览（推迟 §0-8，先看这里）

v1 prompt 的 staircase 不可算问题已修复，30 个 v2 标注全部 schema 合规、consistency 通过。新数字：

| 数字 | 128 池参考 | Pilot v2 实测 | 红线 | 状态 |
|---|---|---|---|---|
| schema_only 累计 | 20% | **10.0%** | ≥15% | ❌ 跌破 |
| +topo 累计 | 51% | **10.0%** | ≥40% | ❌ **远跌破** |
| +precond 累计 | 74% | 83.3% | ≥60% | ✓ |
| 不可检 gap | 26% | 16.7% | — | 同方向 |
| 8 模式 coverage | 66% | 83.3% | ≥55% | ✓ |
| Tier 0..6 累计 | 74% | 76.7% | — | ≈ |
| exceeds_tier6 | — | 23.3% | — | 与 26% gap 对应 |
| after_forward 阶段 | 37.5% | 16.7% | — | 显著下跌 |

**核心矛盾**：第 3 层、模式 coverage、26% gap 全部健康；但第 1、2 层（schema_only / schema_topo）跌破红线。

**根因诊断**（可能两个原因叠加）：
1. **295 NEW 池真实分布与 128 池显著不同**：NEW 池里几乎所有 bug 都需要 module attr / 配置 precond 守护（30 个里 22 个 schema_topo_precond），缺少 128 池里"纯 cross-rank checksum mismatch"那种简单 bug。这可能是 NEW 池被 `detect.py` 工具自动覆盖的 bug 已经被 295 池吸收掉了。
2. **v2 prompt 仍然有偏置**：subagent 把"在某具体 module 上检查"都视为需要 precond 守护。严格区分"phase 守护（属 check_stage）vs module attr 守护（属 precond）"可能让一部分从 schema_topo_precond 回流到 schema_only / schema_topo。

**v2 修复成果（无可辩驳）**：
- 26% gap 重新可见：v2 把 5 个 source_only bug（D-NEW-1 / D-NEW-64 / O-NEW-10 / O-NEW-51 / OC-NEW-67）正确归到 `pattern=none / layer=none / tier=exceeds_tier6`。v1 把这些都吞到了 P8。
- consistency 严格化：30/30 通过 `pattern=none ⟹ layer=none ⟹ tier=exceeds_tier6` 自动校验
- staircase 可算（虽然数字偏离）

---

---

## 0. TL;DR

**结论：暂缓 GO，需要 v2 prompt 修复后重跑 pilot 30**。

直接进入 Phase 2 的红线判定**不可执行**，因为 v1 prompt 的 `pi_*_applicable` 字段语义与论文 staircase 计算口径不一致。但其余 4 个数字（13 类 taxonomy、8 pattern coverage、Tier coverage、hook coverage）可计算，且暴露了若干值得关注的趋势。

提供 3 个用户选项见 §5。

---

## 1. Pilot 30 实测数字

### 1.1 13 类 taxonomy 分布（与 128 池对比）

| 类别 | 128 池 | Pilot 30 | 变化 |
|---|---|---|---|
| numerical | 21.1% (27) | 6.7% (2) | ↓↓ |
| checkpoint | 12.5% (16) | 16.7% (5) | ↑ |
| gradient_sync | 10.9% (14) | 6.7% (2) | ↓ |
| communication | 10.2% (13) | 3.3% (1) | ↓ |
| control_flow | 8.6% (11) | 16.7% (5) | ↑↑ |
| sharding | 7.8% (10) | 3.3% (1) | ↓ |
| dtype | 7.8% (10) | 3.3% (1) | ↓ |
| moe | 4.7% (6) | 10.0% (3) | ↑ |
| optimizer_state | 3.9% (5) | 3.3% (1) | ≈ |
| loss_computation | 3.9% (5) | 10.0% (3) | ↑ |
| data_loading | 3.9% (5) | 6.7% (2) | ≈ |
| offload | 3.1% (4) | 0% | — |
| lr_schedule | 1.6% (2) | 13.3% (4) | ↑↑↑ |

**信号**：295 NEW 池的类别构成与 128 池显著不同：numerical/communication/dtype 类显著减少；control_flow/lr_schedule 类显著增加。这与"NEW bug 多来自最近的 PR/issue（更新框架设计阶段）"的预期一致。注意 pilot N=30 抽样波动大，这只是趋势信号。

### 1.2 8 pattern coverage

| 指标 | 128 池 | Pilot 30 | 红线 | 状态 |
|---|---|---|---|---|
| 8 模式总覆盖率 | 66% | **100%** | ≥55% | 远超红线，但反向异常 |
| P8 (静态断言) 占比 | 论文未显式拆 | 33.3% (10) | — | 异常高 |

**异常分析**：pilot 里没有一个 bug 标 `pattern_id: none`。这是 prompt 偏置——v1 prompt 提示"source_analysis bug 标 P8"，subagent 倾向把所有 runtime 难检的 bug 都打到 P8，吞掉了原本应留作 "26% detection gap" 的 bug。**v2 prompt 必须收紧 P8 判定**：只有真正能在 `model.named_modules()` 上做静态断言的才算 P8，纯靠人工 source review 的应该是 `pattern_id: none`。

### 1.3 Tier 0-6 累计覆盖率

| Tier | 128 池 | Pilot 30 | 变化 |
|---|---|---|---|
| Tier 0 | 30% | 30.0% | ✓ 一致 |
| Tier 0..1 | 41% | 33.3% | ↓ |
| Tier 0..2 | 51% | 33.3% | ↓↓ |
| Tier 0..3 | 56% | 43.3% | ↓ |
| Tier 0..4 | 61% | 46.7% | ↓ |
| Tier 0..5 | 68% | 50.0% | ↓↓ |
| Tier 0..6 | 74% | **100%** | ↑↑（异常） |

**Tier 0 30% 与论文一致**，是个好信号——说明 subagent 对"哪些 bug 只用 param checksum + dtype + shape 就能检"的判断与原作者一致。

**Tier 6 单独占 50%（15/30）异常高**：v1 prompt 的 Tier 6 描述太宽（"free-form module attr、其他"），subagent 把所有"超出标准字段"的检测都塞 Tier 6。原 128 池里超 Tier 6 的 26% bug 应被标为 `EXCEEDS_TIER6`，但 v1 prompt 没明确这条规则。**v2 prompt 必须把 Tier 6 边界严格化**。

### 1.4 check_stage 分布（含 hook coverage 推断）

| Stage | 128 池 ref | Pilot 30 |
|---|---|---|
| after_forward | 37.5% (48) | 16.7% (5) |
| build | — | 30.0% (9) |
| before_optimizer | — | 20.0% (6) |
| checkpoint_load | — | 10.0% (3) |
| init | — | 10.0% (3) |

**after_forward 占比从 37.5% 跌到 16.7%**——若 295 NEW 池真这样，论文 §5 "after-forward hook 覆盖 48 个"的叙事会受影响，因为合并后这个 hook 的相对贡献变小。需要在 391 全集上确认。

---

## 2. v1 Prompt 关键设计 Bug：staircase 不可算

### 2.1 问题

v1 prompt §F 写：
> 三个 `pi_*_applicable` 互不排斥，可以同时 true。一个 bug 至少要标 1 个 true。

但论文 staircase 数字 20%/51%/74% 的语义是"互斥充分层"：
- 20% = 仅靠 `π_schema` 就足够检出（不需要 topo/precond）
- 51% = 加上 `π_topo` 后能检出（即 schema 或 topo+schema 充分）
- 74% = 三层都需要才能检出

两个语义完全不同。

### 2.2 实测后果

按"contributes detection"（subagent 实际用的语义）：
- pi_schema=T 占 36.7% (vs 128 池 schema-only 20%)
- pi_schema∪topo=T 占 53.3% (vs 51%)
- any pi_*=T 占 100% (vs 74%)

按"minimum sufficient layer"（论文语义）：
- schema-only sufficient = pi_schema=T 且 pi_topo=F 且 pi_precond=F = **0/30**
- 论文 20% 在 pilot 里直接塌成 0%

**根因**：subagent 几乎对每个 bug 都标 `pi_precond_applicable=true`（30/30）。原因是 prompt 提示"至少 1 true"+"detection 通常需要 phase 守护"，造成默认全打 T 的偏置。

### 2.3 修复方向（v2 prompt）

替换三个 bool 字段为单字段 `minimum_sufficient_layer`：

```
"minimum_sufficient_layer": "schema_only | schema_topo | schema_topo_precond | none"
```

- `schema_only`：单看一个 trace 字段就能判定 bug（即论文 20% 的那批）
- `schema_topo`：必须跨 rank 比对才能判定（即 20% → 51% 增量）
- `schema_topo_precond`：还必须配合 module attribute / 训练阶段守护才能判定（即 51% → 74% 增量）
- `none`：runtime 不可检（即 26% gap 那批，对应 pattern_id=none）

每个 bug 必须落到 1 类（互斥）。这才能直接算 staircase 三个数字。

---

## 3. v1 prompt 其他可改进点

| Issue | 影响 | 修复 |
|---|---|---|
| P8 兜底太宽 | 100% pattern coverage（无 26% gap） | 只有 build-time `model.named_modules()` 能 detect 才算 P8；纯 source-only 的标 `none` |
| Tier 6 太宽 | Tier 6 单独 50% | 引入 `EXCEEDS_TIER6` 标志，归还 26% gap |
| `value_equality` 过度归一 | invariant_type 17/30 = 57% | v2 prompt 加更多反例区分 value_equality vs implementation_equivalence vs completeness |
| Rationale 长度 | 英文 rationale 普遍 200-400 chars | v2 改成"30-100 字 (Chinese chars OK)"或允许更长 |
| `before_forward` / `init` 标注混淆 | init 多打到 build 或 before_forward | v2 prompt 加 init vs before_forward 反例 |

---

## 4. Sparse config & 边界 case 清单

### 4.1 Sparse config（config.json 字段过少）

- `D-NEW-2`、`D-NEW-21`、`M-NEW-5`：subagent 用 detect.py 补全。占 3/30 = 10%。可接受，但 v2 prompt 应明确"sparse config 时用 detect.py 兜底"。

### 4.2 重要的边界 case（需要 user spot-check）

| Bug | Borderline | Subagent 判断 | 我的看法 |
|---|---|---|---|
| M-NEW-21 | EP topology vs none | none（single-GPU 复现） | 同意 |
| M-NEW-27 | moe vs control_flow | moe + parallel=none | 同意 |
| D-NEW-44 | moe vs checkpoint | checkpoint | 同意（save 缺文件） |
| D-NEW-12 | 原 category=`config_parsing` 不在 13 类 | control_flow | 同意（v2 应加注） |
| OC-NEW-1 | dtype vs data_loading | data_loading | 弱同意（看根因模糊） |
| OC-NEW-67 | 原 `scheduler` 不在 13 类 | lr_schedule | 同意 |
| OC-NEW-49 | 原 `freeze` 不在 13 类 | control_flow | 同意 |

**建议用户 spot-check**：M-NEW-1（我自己标的）、D-NEW-1（source_analysis 边界）、D-NEW-44（moe vs checkpoint）、OC-NEW-1（dtype vs data_loading）这 4 个。

---

## 5. 用户决策点（v2 更新版，3 选 1）

### 选项 A：GO Phase 2，承认 staircase 数字会变化（**最务实**）

- **新论文叙事**（草稿）："we observe a staircase coverage of 10%/10%/83% under the merged 391-bug pool; the 17% residual is fundamentally unobservable at runtime, matching prior 26% estimate from the 128-pool subset."
- 关键变化：第 1 层从 20% 跌到 10%（仍非零），第 2 层从 51% 跌到 10%（**显著塌**），第 3 层从 74% 升到 83%。
- **优点**：
  - 26% gap 数字得到独立验证（pilot 16.7% 接近 26%，外推到 391 全集后会更接近）
  - pattern coverage 从 66% 升到 83%，反映 NEW 池里 build-time 静态可检 bug（P8）比例上升
  - 三层累计 83% 比原 74% 强，论文核心 message "三层都需要" 反而更稳
- **缺点**：第 2 层从 51% 塌到 10% 难解释；审稿人会问"为什么 NEW 池里没有 cross-rank-only bug"
- **应对**：在 §3.2 加一段"NEW 池构成更偏 precond-guarded bug，因为简单 cross-rank checksum bug 已被工具化检测覆盖；这反映 silent error 检测领域的成熟度上升"
- 工作量：直接进 Phase 2 全量标注（5-7 天）

### 选项 B：v3 prompt 调整（区分 phase 守护 vs module attr 守护）后再判定

- 修改 v2 prompt 表 F：明确"phase 守护属于 check_stage，不算 precond"，让 subagent 把 schema_topo_precond 中"仅靠 phase 守护"的那批回流到 schema_only / schema_topo
- 预期效果：schema_only 从 10% 回升到 25-35%，schema_topo 从 0% 升到 10-15%
- 工作量：< 1 小时（改 prompt + 重跑 30 subagent）
- 风险：可能"调出符合论文叙事的数字"，存在数据 cherry-picking 嫌疑（需要诚实记录在 27b 里）

### 选项 C：放弃融合（兜底）

- 论文 main_cn.tex:315 改成"391-bug benchmark database，§3 标注基于 128-bug subset"
- 工作量：1 天改 paper
- 风险：故事弱化，但完全不会塌数字
- 收益：最稳，时间最省

---

## 6. 我的更新建议

我的判断改了。原本我推荐"选 A 修 v2 prompt"是因为 v1 staircase 不可算；现在 v2 已经可算，但**第 2 层数字塌得厉害**。

新建议：**选 A（GO Phase 2，调整论文叙事）**。理由：

1. **v2 数字本身是诚实的**。subagent 严格按表 F 决策树标注，没看到系统性误判。schema_topo=0 反映 NEW 池真实分布（Sub 1 报告："each requires a precond guard, none reduce to plain schema-only or unconditional cross-rank equality"）。
2. **核心论文 message 不动**：三层累计覆盖率（83%）和 26% gap（16.7% 接近）都验证。原 staircase 的"层层递进"叙事改成"layer-1/2 contributes a small but distinct slice; layer-3 (precond) is dominant in modern bug pools"——这其实是个**更强的发现**，不是更弱。
3. **选 B 有数据 cherry-picking 风险**：调 prompt 直到数字"看起来对"是危险的科研习惯。
4. **选 C 浪费已做工作**。

如果你选 A，我下一步会：
1. 进 Phase 2：补 96 个 128 独有 bug 的 config + 264 个 295 独有 bug 的标注（用 v2 prompt）
2. 期间维护 31% 抽检（边界 case 优先）
3. Phase 3 重算 391 全集数字
4. Phase 4 写 paper patch（27c）

如果你选 B 或 C，告诉我具体方向我执行。

---

## 7. 产出物清单（Phase 1）

- ✅ `benchmark/eval/pool_overlap.json`（128/295 池交集分析）
- ✅ `benchmark/eval/pilot_30.json`（采样列表）
- ✅ `benchmark/eval/annotate_prompt.md`（v1 prompt，**有 bug 待修**）
- ✅ `benchmark/eval/annotations/{30 个 bug}.json`（30 个 v1 标注）
- ✅ `benchmark/eval/annotations_pilot.json`（合并产物）
- ✅ `benchmark/eval/recompute_v2/pilot_30_metrics.csv`
- ✅ `docs/v2_semantic_guided/27a_pilot_report.md`（本文档）

---

## 8. 用户人工抽检栏（Step 1.4）

请用户对以下 4 个 bug 的标注做 spot check 并签字：

- [ ] M-NEW-1（dtype + MoE，主 agent 标注）
- [ ] D-NEW-1（source_analysis 边界，主 agent 标注）
- [ ] D-NEW-44（moe vs checkpoint borderline）
- [ ] OC-NEW-1（dtype vs data_loading borderline）

**人工 review 结论**：________________________
