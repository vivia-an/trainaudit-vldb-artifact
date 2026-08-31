# Semantic Hypothesis Generation：LLM 怎么生成搜索方向

---

## 1. 核心定位

LLM 在新方案中**不生成 invariant**，只生成 **semantic hypothesis**——一个结构化的搜索指令，指明"在 trace 的哪个子空间里可能存在有意义的 invariant"。

关键转变：
- 旧方案：LLM 输出 → "DP ranks 间 optimizer state 应该一致"（NL invariant，需要翻译成 SQL）
- 新方案：LLM 输出 → `{entities, relation_type, rank_dimension, stage, rationale}`（搜索指令，不需要翻译）

---

## 2. Semantic Hypothesis 的结构化格式

```json
{
  "hypothesis_id": "H-042",
  
  "entities": {
    "target_fields": ["cksum"],
    "param_filter": "param_name LIKE '%.exp_avg' OR param_name LIKE '%.exp_avg_sq'",
    "description": "Adam optimizer 的一阶和二阶矩估计"
  },
  
  "relation_type": "cross_rank_equality",
  
  "rank_dimension": {
    "comparison": "same_tp_rank AND same_pp_rank AND different_dp_rank",
    "description": "同一 TP/PP 位置下的不同 DP rank 之间"
  },
  
  "stage": "after_optimizer",
  
  "temporal": {
    "scope": "per_step",
    "description": "每个 step 结束时独立检查"
  },
  
  "precondition_hints": [
    "可能只在 zero_stage < 3 时成立（ZeRO-3 会 shard optimizer state）",
    "frozen parameters 没有 optimizer state",
    "distributed optimizer 可能有不同的同步时机"
  ],
  
  "rationale": {
    "source": "ZeRO 论文 (Rajbhandari et al., 2020) + Megatron-LM distributed_optimizer.py",
    "reasoning": "ZeRO-1/2 只 shard optimizer state 的 computation 不 shard storage，所以最终 state 应该在 DP ranks 间一致"
  },
  
  "confidence": 0.8
}
```

### 2.1 字段说明

| 字段 | 含义 | 为什么需要 |
|------|------|-----------|
| `entities` | 要检查的目标字段和参数范围 | 缩小搜索空间——不在全部参数上枚举 |
| `relation_type` | 期望的关系类型 | 决定用哪组 predicate 模板（equality / monotonicity / bounded 等） |
| `rank_dimension` | 在哪个 rank 维度上比较 | 决定 SQL 的 GROUP BY 和 JOIN 方式 |
| `stage` | 在哪个 training stage 检查 | 决定 WHERE 条件 |
| `temporal` | 时间范围（per_step / cross_step / window） | 决定是否需要 window function |
| `precondition_hints` | LLM 猜测的可能限制条件 | **不作为硬约束**，而是提示 data-driven 验证阶段关注这些维度 |
| `rationale` | 语义依据 | 用于 Layer 4 的 semantic filtering |

### 2.2 关键设计决策

**`precondition_hints` 是 hints 而非 hard constraints**：

这是和当前 TrainAudit 最大的区别。当前方案让 LLM 写出完整的 precondition（写不全就 false positive）。新方案中：
- LLM 只给出 hints（"可能在 ZeRO-3 下不成立"）
- 真正的 precondition 由 data-driven validation 阶段从 trace 中**自动发现**
- 如果 healthy trace 中某些 config 下 predicate 不成立，系统自动添加对应的 WHERE 条件

---

## 3. Relation Type 的分类体系

LLM 的 hypothesis 中 `relation_type` 从以下预定义类型中选择：

### 3.1 Cross-Rank Relations（跨 rank 关系）

| 类型 | 含义 | 典型场景 |
|------|------|---------|
| `cross_rank_equality` | 某字段在指定 rank group 内完全一致 | DP 参数一致性、gradient all-reduce 结果 |
| `cross_rank_partitioned` | 某字段在 rank group 内各不相同且互补 | TP shard、ZeRO-3 partitioned optimizer state |
| `cross_rank_bounded_diff` | 字段值差异在允许范围内 | 数值精度导致的微小差异 |

### 3.2 Cross-Step Relations（跨 step 关系）

| 类型 | 含义 | 典型场景 |
|------|------|---------|
| `monotonic_increase` | 随 step 单调增加 | step counter、累积 loss |
| `monotonic_decrease` | 随 step 单调减少 | learning rate decay |
| `periodic_reset` | 每 N 步重置一次 | micro_step counter、gradient accumulation buffer |
| `bounded_change` | 相邻 step 变化幅度有界 | 参数更新量、gradient norm |

### 3.3 Cross-Stage Relations（跨 stage 关系）

| 类型 | 含义 | 典型场景 |
|------|------|---------|
| `stage_unchanged` | 在两个 stage 之间不变 | forward 前后参数不变（no in-place update） |
| `stage_changed` | 在两个 stage 之间一定改变 | optimizer step 前后参数改变 |
| `stage_conditional_change` | 特定条件下才改变 | gradient accumulation 期间只有最后一个 micro_step 才做 update |

### 3.4 Value-Level Relations（值约束）

| 类型 | 含义 | 典型场景 |
|------|------|---------|
| `no_nan_inf` | 不含 NaN / Inf | 数值稳定性 |
| `bounded_range` | 值在某范围内 | gradient clipping 后 norm ≤ max_grad_norm |
| `statistical_property` | 满足某统计性质 | 初始化后 mean ≈ 0、std ≈ 1/√n |

---

## 4. LLM Prompt 设计思路

### 4.1 输入

给 LLM 的输入包括：

1. **框架源码片段**：相关模块的代码（optimizer、communication、gradient handler）
2. **框架文档/论文**：相关算法的描述
3. **Trace schema**：DuckDB 表结构——让 LLM 知道哪些字段可用
4. **Relation type 列表**：上面 §3 的分类体系——限制 LLM 的输出空间
5. **已有 hypothesis 列表**：避免重复

### 4.2 Prompt 结构

```
你是一个分布式训练系统专家。你的任务是从框架源码和文档中识别可能的正确性关系。

【输入】
- 框架源码片段：{code}
- 相关文档：{docs}
- 可用的 trace 字段：{schema}
- 可选的关系类型：{relation_types}
- 已有的 hypothesis：{existing}

【任务】
请识别源码/文档中蕴含的正确性关系，输出结构化的 semantic hypothesis。
注意：
1. 你不需要给出精确的检查规则，只需要指出"哪些实体之间应该存在什么类型的关系"
2. precondition_hints 只需列出你能想到的可能限制，不需要穷举
3. 每个 hypothesis 必须给出语义依据（来自哪段代码/哪个算法描述）

【输出格式】
{structured JSON}
```

### 4.3 和当前 TrainAudit Invariant Miner 的区别

| 方面 | 当前 Invariant Miner | 新方案 Hypothesis Generator |
|------|---------------------|---------------------------|
| LLM 输出 | 完整的 NL invariant + precondition + SQL | 只输出搜索指令（entities + relation_type + rank_dim + stage） |
| Precondition | LLM 必须写全（写不全就 FP） | LLM 只给 hints，真正的 precondition 由数据发现 |
| 验证方式 | Bidirectional adversarial（LLM 构造反例） | Data-driven（在 healthy trace 上验证） |
| 对 LLM 的要求 | 高——需要同时理解语义 + 精确描述 + 写 SQL | 低——只需要理解"哪里可能有关系" |
| 失败模式 | Hallucinated invariant → 检测时 FP 或 FN | Hallucinated hypothesis → 数据验证阶段自动排除（只是浪费搜索时间） |

最后一行是关键：**hypothesis hallucination 的代价从"检测错误"降级为"浪费搜索时间"**——这是本质性的安全性改进。

---

## 5. 多次采样策略

参考 LaM4Inv (ASE'24) 的核心观察：

> LLM 单次可能给不出完整的 hypothesis 集，但多次采样可以覆盖大部分有意义的关系。

策略：
1. 对同一段源码，用不同 prompt variant（不同角度的问题）多次采样
2. 对 hypothesis 做去重（基于 entities + relation_type + rank_dimension 的语义相似度）
3. 保留所有非重复的 hypothesis——即使某个 hypothesis 看起来可疑，也让 data-driven 阶段来判断

这进一步降低了对 LLM 单次输出质量的依赖。

---

## 6. 开放问题

1. **Hypothesis 的粒度如何控制**：太粗（"参数应该在 DP 间一致"）导致搜索空间仍然很大；太细（"transformer.layers.0.attention.query.weight 的 exp_avg 在 step=100 时应该一致"）失去泛化性。
2. **Relation type 体系是否完备**：§3 的分类是否覆盖了所有已知 silent error 类型？需要对照 B1-B15 逐个验证。
3. **源码切片策略**：Megatron-LM 有数百个文件，给 LLM 哪些文件？怎么切片？这影响 hypothesis 的覆盖率。
