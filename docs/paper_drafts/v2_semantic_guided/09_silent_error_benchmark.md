# Silent Error Benchmark：128 个真实 Bug 的分析

> 从 Megatron-LM (42)、DeepSpeed (46)、OLMo + PyTorch FSDP (40) 的 GitHub issue/PR 中收集。
> 原始数据：`exp/megatron_silent_errors.json`、`exp/deepspeed_silent_errors.json`、`exp/olmo_silent_errors.json`

---

## 1. 总览

| 仓库 | 数量 | 数据文件 |
|------|------|---------|
| NVIDIA/Megatron-LM | 42 | `exp/megatron_silent_errors.json` |
| microsoft/DeepSpeed | 46 | `exp/deepspeed_silent_errors.json` |
| allenai/OLMo + PyTorch FSDP | 40 | `exp/olmo_silent_errors.json` |
| **总计** | **128** | |

---

## 2. 按 Bug 类别分布

| 类别 | 数量 | 说明 |
|------|------|------|
| **numerical** | 27 | 数值计算错误（loss scaling、normalization、溢出等） |
| **checkpoint** | 16 | Checkpoint 保存/恢复时状态不一致 |
| **gradient_sync** | 14 | 梯度同步/归约错误 |
| **communication** | 13 | 通信操作错误（all-reduce/all-gather 参数、group 配置） |
| **control_flow** | 11 | 控制流错误（counter、条件分支、初始化顺序） |
| **sharding** | 10 | 参数切分/分布错误 |
| **dtype** | 10 | 数据类型不匹配 |
| **moe** | 6 | MoE 专用 bug（routing、expert gradient、EP group） |
| **optimizer_state** | 5 | Optimizer state 错误 |
| **loss_computation** | 5 | Loss 计算逻辑错误 |
| **data_loading** | 5 | 数据加载/采样错误 |
| **offload** | 4 | CPU offload 相关的同步错误 |
| **lr_schedule** | 2 | Learning rate 调度错误 |

### 关键发现

1. **numerical（27 个）是最大类别**——不是简单的 NaN/Inf，而是值"看起来合理但不正确"的 bug
2. **checkpoint（16 个）数量超出预期**——保存/恢复是 silent corruption 的高频入口
3. **gradient_sync + communication（27 个）合计占 21%**——通信相关 bug 是主力

---

## 3. 按并行维度分布

| 维度 | 数量 | 占比 |
|------|------|------|
| **none**（与并行无关） | 47 | 37% |
| **DP** | 33 | 26% |
| **FSDP** | 14 | 11% |
| **TP** | 10 | 8% |
| **EP** | 10 | 8% |
| **CP** | 4 | 3% |
| **PP** | 4 | 3% |
| **SP** | 2 | 2% |
| 组合（DP+PP, DP+TP 等） | 4 | 3% |

### 关键发现

1. **37% 的 bug 与并行策略无关**——纯逻辑/数值错误，不需要 cross-rank 检查
2. **DP 是并行相关 bug 的重灾区**（33 个）——ZeRO 各 stage 的梯度/optimizer state 同步问题
3. **FSDP（14 个）独立于 DP**——PyTorch 原生分布式训练有自己的一套 bug 模式
4. **EP（10 个）意外地多**——MoE expert parallelism 的通信 group 配置是高风险区

---

## 4. 按 Invariant 类型分布

| 类型 | 数量 | 说明 |
|------|------|------|
| **bounded_change** | 43 | 值的变化在合理范围内（最大类别） |
| **cross_rank_equality** | 25 | 跨 rank 的值应该一致 |
| **value_equality** | 11 | 某个值应该等于特定值 |
| **dtype_consistency** | 8 | 数据类型应该一致 |
| **numerical_consistency** | 5 | 数值精度一致性 |
| **value_range** | 4 | 值在特定范围内 |
| **implementation_equivalence** | 4 | 两种实现应产生相同结果 |
| 其他（monotonic, completeness 等） | 28 | 各种特殊类型 |

### 关键发现

1. **bounded_change（43 个）是最大类别**——这意味着很多 bug 不是"值完全错了"，而是"变化幅度异常"
2. **cross_rank_equality（25 个）** 仍然重要，但只占 20%——不是所有 bug 都能通过 cross-rank 比较发现
3. **37% 的 bug 需要非 cross-rank 的检查方式**——这验证了你的直觉：只做 cross-rank equality 不够

---

## 5. 最关键的发现：required_trace_fields 频率

**高频字段（出现 >10 次）**：

| 字段 | 被需要的 bug 数 | 当前 schema 有吗 |
|------|---------------|-----------------|
| param_name | 40 | ✅ |
| dp_rank | 34 | ✅ |
| cksum | 33 | ✅ |
| grad_norm | 25 | ⚠️ 需要确认 |
| loss_value | 22 | ❌ 当前未采集 |
| grad_cksum | 14 | ⚠️ 需要确认 |
| dtype | 11 | ✅ |
| iteration/step_count | 21 | ✅ |
| tp_rank | 10 | ✅ |

**中频字段（5-10 次）**：

| 字段 | 被需要的 bug 数 | 当前 schema 有吗 |
|------|---------------|-----------------|
| layer_id | 9 | ⚠️ 可从 param_name 推导 |
| optimizer_state_cksum | 7 | ⚠️ 需要单独采集 |
| zero_stage | 7 | ✅（在 topology 表） |
| learning_rate | 6 | ❌ 需新增 |
| micro_step_id | 6 | ❌ 需新增 |
| world_size | 6 | ✅（在 topology 表） |
| shape | 5 | ✅ |
| pp_rank | 5 | ✅ |
| num_tokens | 5 | ❌ 需新增 |
| offload_enabled | 5 | ❌ 需新增（在 topology 表） |

### Schema 覆盖分析

**当前 schema 能直接覆盖的 bug**：
- 需要 cksum + rank 字段的 cross_rank_equality bug → ~25 个
- 需要 dtype + rank 字段的 dtype_consistency bug → ~8 个
- 需要 shape 字段的 bug → ~5 个
- **合计约 38 个（30%）**

**补充以下字段后能覆盖的 bug**：
- +loss_value → +22 个
- +grad_norm → +25 个（部分重叠）
- +learning_rate → +6 个
- +micro_step_id → +6 个
- +optimizer_state_cksum → +7 个
- **补充后合计约 80-90 个（63-70%）**

**仍然无法覆盖的 bug（~30-40 个）**：
- 需要 activation 级数据的 bug
- 需要 reference implementation 对比的 bug（implementation_equivalence 类）
- 需要 checkpoint 文件内容比对的 bug
- 需要数据加载行为追踪的 bug

---

## 6. 按 check_stage 分布

| Stage | 数量 | 说明 |
|-------|------|------|
| **after_forward** | 48 | 最多——很多 numerical/loss bug 在 forward 后就能观测到 |
| **after_backward** | 37 | 梯度相关 bug |
| **after_optimizer** | 19 | 参数更新后的状态检查 |
| **checkpoint 相关** | 12 | 保存/加载前后 |
| **data_loading** | 6 | 数据加载后 |
| **initialization** | 6 | 初始化阶段 |

### 关键发现

当前 TrainAudit 的 hook 主要在 after_backward 和 after_optimizer，但 **after_forward（48 个）** 是最大的检查点——需要增加 forward 后的采集。

---

## 7. 对 P0-3 问题的回答

### Q: B1-B15（原始 benchmark）在当前 schema 下能否覆盖？
→ 见前次对话中的逐个分析。6 个确定能检测，7 个有条件能检测，2 个不可检测。

### Q: 扩展到 128 个 bug 后，schema 的覆盖率？
→ 当前 schema 约覆盖 30%。补充 5 个关键字段（loss_value, grad_norm, learning_rate, micro_step_id, optimizer_state_cksum）后可覆盖 63-70%。剩余 30% 需要更深层数据（activation、checkpoint 内容、数据加载行为）。

### Q: Schema 的最小完备集是什么？
→ **核心字段**（覆盖 70% 的 bug）：

```
必须有：cksum, param_name, dp_rank, tp_rank, pp_rank, ep_rank, step, stage, dtype, shape
强烈建议加：grad_norm, grad_cksum, loss_value, learning_rate, micro_step_id
建议加：optimizer_state_cksum, num_tokens, has_nan, has_inf, offload_enabled
```

### Q: 哪些 bug 是方法论上覆盖不了的？
→ 三类：
1. **Sub-percent 数值偏差**（如 B14/B15）——信号太弱，且所有 rank 受同样影响
2. **Checkpoint 文件级 corruption**——需要对比 checkpoint 文件内容，不在 training loop trace 范围内
3. **数据加载逻辑错误**——需要追踪每个 rank 看到了哪些 data sample，当前 schema 不覆盖

---

## 8. 对方案设计的启示

### 8.1 不能只做 cross-rank equality

128 个 bug 中只有 25 个（20%）是 cross_rank_equality 类型。最大的类别是 bounded_change（43 个），意味着需要：
- 跨 step 的变化幅度检查
- 数值范围检查
- 与 baseline/历史数据的对比

### 8.2 Invariant type 的优先级

按覆盖 bug 数量排序的 predicate template 优先级：

| 优先级 | Template 类型 | 覆盖 bug 数 | 实现难度 |
|--------|-------------|------------|---------|
| P0 | bounded_change（值变化在合理范围内） | 43 | 中（需要 baseline） |
| P0 | cross_rank_equality（跨 rank 一致） | 25 | 低 |
| P1 | value_equality（值应等于特定值） | 11 | 低 |
| P1 | dtype_consistency（类型一致） | 8 | 低 |
| P2 | value_range（值在范围内） | 4 | 低 |
| P2 | numerical_consistency | 5 | 中 |

### 8.3 最大的 gap：bounded_change 类 invariant

这是当前方案最薄弱的地方。bounded_change 需要：
- "正常的变化幅度是多少？" → 需要从 healthy trace 中学习 baseline
- "什么时候变化幅度大是正常的？"（比如 warmup 期间 LR 变化大）→ 需要 training phase awareness
- "变化幅度的阈值怎么设？" → 需要统计方法，不是简单的等式检查

这正是 **semantic knowledge 能发挥价值的地方**：LLM 理解"warmup 期间 gradient norm 波动大是正常的"，从而给 bounded_change 检查设定合适的 context。

---

## 9. 数据文件说明

每个 bug 的完整记录格式：
```json
{
  "id": "M-001 / D-001 / O-001",
  "repo": "NVIDIA/Megatron-LM / microsoft/DeepSpeed / allenai/OLMo",
  "issue_or_pr": "Issue #xxx / PR #xxx",
  "url": "https://github.com/...",
  "title": "...",
  "description": "一句话描述",
  "category": "...",
  "parallel_dimension": "...",
  "detection_signal": "什么可观测的现象能暴露这个 bug",
  "required_trace_fields": ["field1", "field2", ...],
  "check_stage": "...",
  "invariant_type": "...",
  "severity": "high / medium / low"
}
```

**注意**：这些 bug 是通过 GitHub 搜索 + LLM 分析得到的，部分 issue URL 和细节需要人工核实。特别是：
- 确认 issue/PR 确实存在且描述准确
- 确认 bug 确实是 silent error（不是 crash/显式报错）
- 确认 detection_signal 和 required_trace_fields 的分析正确
