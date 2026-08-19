# Data-Driven Validation：用 Healthy Trace 验证并精化 Predicate

---

## 1. 核心思路

这一层是整个方案的 **precision oracle**——替代 SE 领域的 SMT solver。

输入：~2700 个候选 predicate（来自 Layer 2）
输出：verified invariant library（在 healthy trace 上始终成立、且语义有意义的 predicate）

核心假设：**如果一个 predicate 在多个不同配置的 healthy execution 上始终成立，那它大概率是一个真正的 invariant。**

---

## 2. Healthy Trace 的来源

### 2.1 什么是 "healthy"

一次 healthy execution 满足：
1. 训练正常完成（没有 crash、没有 NaN）
2. Loss 曲线符合预期（稳步下降）
3. 没有已知 bug（使用 verified 版本的框架）
4. 配置是典型的（常见的并行策略组合）

### 2.2 需要多少种 healthy trace

**关键洞察：多种不同配置的 healthy trace 能自动发现 precondition。**

| Trace | 配置 | 作用 |
|-------|------|------|
| H1 | DP=4, TP=1, PP=1, ZeRO-0 | 纯 DP 基线 |
| H2 | DP=2, TP=2, PP=1, ZeRO-0 | TP 混合 |
| H3 | DP=2, TP=1, PP=2, ZeRO-0 | PP 混合 |
| H4 | DP=4, TP=1, PP=1, ZeRO-1 | ZeRO-1 |
| H5 | DP=4, TP=1, PP=1, ZeRO-2 | ZeRO-2 |
| H6 | DP=4, TP=1, PP=1, ZeRO-3 | ZeRO-3（optimizer state sharded） |
| H7 | DP=2, TP=2, PP=2, ZeRO-0 | 3D 并行 |
| H8 | DP=4, TP=1, PP=1, ZeRO-0, grad_accum=4 | Gradient accumulation |

每个 trace 只需要跑几十到几百个 step（不需要完整训练）。

### 2.3 Precondition 自动发现机制

假设一个候选 predicate 是 "DP ranks 间 optimizer state cksum 一致"：

| Trace | 配置 | 该 predicate 是否成立 |
|-------|------|--------------------|
| H1 | ZeRO-0 | ✅ 成立 |
| H4 | ZeRO-1 | ✅ 成立 |
| H5 | ZeRO-2 | ✅ 成立 |
| H6 | ZeRO-3 | ❌ 不成立 |

系统自动推断 precondition：`zero_stage < 3`

这个 precondition **不是 LLM 猜的，是从数据中发现的**。

---

## 3. Validation Pipeline

### Step 1: 在每个 healthy trace 上执行所有候选 predicate

```python
results = {}  # predicate_id -> {trace_id -> pass/fail/partial}

for trace in healthy_traces:
    load_trace_to_duckdb(trace)
    config = get_topology(trace)
    
    for predicate in candidates:
        violations = execute_sql(predicate.sql)
        if len(violations) == 0:
            results[predicate.id][trace.id] = "pass"
        elif len(violations) < threshold:
            results[predicate.id][trace.id] = "partial"  # 少数 step 违反
        else:
            results[predicate.id][trace.id] = "fail"
```

### Step 2: 分类候选 predicate

根据跨 trace 的结果，分成四类：

| 类别 | 定义 | 处理 |
|------|------|------|
| **Universal** | 在所有 healthy trace 上都 pass | 直接接受为 invariant |
| **Conditional** | 在部分 trace 上 pass、部分 fail | 分析 pass/fail 的配置差异，自动生成 precondition |
| **Noisy** | 在某些 trace 上 partial（少量 step 违反） | 分析违反的 step 特征（warmup？ckpt？），可能需要排除特殊 step |
| **Spurious** | 在所有/大部分 trace 上 fail | 丢弃 |

### Step 3: Precondition 自动生成

对 Conditional 类的 predicate，自动推断 precondition：

```python
def infer_precondition(predicate, results, traces):
    """
    对比 pass 和 fail 的 trace 配置，找出区分条件
    """
    pass_configs = [t.config for t in traces if results[predicate.id][t.id] == "pass"]
    fail_configs = [t.config for t in traces if results[predicate.id][t.id] == "fail"]
    
    # 对每个配置维度，检查是否能区分 pass/fail
    preconditions = []
    for dim in ["zero_stage", "tp_size", "pp_size", "grad_accum", ...]:
        pass_values = set(c[dim] for c in pass_configs)
        fail_values = set(c[dim] for c in fail_configs)
        
        if pass_values.isdisjoint(fail_values):
            # 这个维度完美区分 pass/fail
            preconditions.append(f"{dim} IN {pass_values}")
    
    return preconditions
```

**举例**：

predicate: "DP ranks 间 exp_avg cksum 一致"
- Pass on: H1(ZeRO-0), H4(ZeRO-1), H5(ZeRO-2)
- Fail on: H6(ZeRO-3)
- 自动推断: `zero_stage IN (0, 1, 2)` → 简化为 `zero_stage < 3`

### Step 4: Noisy predicate 的特殊 step 排除

对 Noisy 类的 predicate（大部分 step pass，少数 step fail）：

```python
def analyze_noise(predicate, trace):
    """分析哪些 step 违反，找出 pattern"""
    violation_steps = execute_sql(predicate.sql)  # 返回违反的 step 列表
    
    # 检查是否是特殊 step
    patterns = [
        ("first_step", lambda s: s == 0),
        ("warmup", lambda s: s < warmup_steps),
        ("checkpoint_step", lambda s: s % ckpt_interval == 0),
        ("last_micro_step", lambda s: s.micro_step == grad_accum - 1),
    ]
    
    for name, check in patterns:
        if all(check(s) for s in violation_steps):
            return f"排除 {name} 步后成立"
    
    return "无法解释的 noise，降低置信度"
```

---

## 4. Confidence Scoring

每条通过验证的 invariant 计算置信度：

```
confidence(p) = w1 * coverage_score + w2 * consistency_score + w3 * specificity_score
```

| 分量 | 定义 | 含义 |
|------|------|------|
| `coverage_score` | 在多少种不同配置的 trace 上测试过 | 测试范围越广越可靠 |
| `consistency_score` | 在适用的 trace 上成立的比例 | 1.0 = 每个 step 都成立 |
| `specificity_score` | predicate 的区分度（不是所有 predicate 都 trivially true） | 排除"永远为真"的无用 predicate |

阈值：confidence ≥ 0.8 才进入 final library。

---

## 5. 完整产出格式

一条 verified invariant 的完整记录：

```json
{
  "invariant_id": "INV-042-E1-cksum",
  "source_hypothesis": "H-042",
  "template": "cross_rank_equality_exact",
  "field": "cksum",
  
  "sql": "SELECT step, param_name, COUNT(DISTINCT cksum) ... HAVING COUNT(DISTINCT cksum) > 1",
  
  "preconditions": {
    "auto_discovered": {
      "zero_stage": "< 3",
      "requires_grad": "= true"
    },
    "step_exclusions": ["step = 0"],
    "topology_requirements": {
      "dp_size": "> 1"
    }
  },
  
  "validation_evidence": {
    "tested_on": ["H1", "H2", "H4", "H5", "H7", "H8"],
    "excluded_on": ["H6 (ZeRO-3)"],
    "total_steps_checked": 4800,
    "violations_in_healthy": 0
  },
  
  "semantic_context": {
    "hypothesis_rationale": "ZeRO-1/2 不 shard optimizer state storage",
    "relation_type": "cross_rank_equality",
    "rank_dimension": "dp"
  },
  
  "confidence": 0.95
}
```

---

## 6. 和当前 TrainAudit 验证方式的对比

| 方面 | 当前 TrainAudit | 新方案 |
|------|----------------|--------|
| Precondition 来源 | LLM 写 + bidirectional adversarial 补充 | 从多配置 healthy trace 自动发现 |
| 验证 oracle | LLM 构造反例（可能遗漏） | Healthy trace 数据（ground truth） |
| 覆盖的配置空间 | 取决于 LLM 能想到多少种 corner case | 取决于 healthy trace 覆盖多少种配置 |
| False positive 来源 | Precondition 不全 | 配置空间未覆盖（可通过增加 trace 解决） |
| 可扩展性 | 每条 invariant 需要多轮 LLM 对话 | 批量 SQL 执行，可并行 |

---

## 7. 开放问题

1. **Healthy trace 的获取成本**：需要跑 8+ 种配置 × 几百步。在 8-GPU 机器上大约需要多少时间？
2. **配置空间的覆盖率**：8 种 trace 是否足够？是否存在某些 edge case 配置只有在特定组合下才暴露 precondition？
3. **"Healthy" 的定义**：如果 healthy trace 本身包含一个未被发现的 silent error 怎么办？→ 这个 bootstrapping 问题需要讨论。
4. **跨框架迁移**：在 Megatron 上验证的 invariant，在 DeepSpeed 上需要重新跑 healthy trace 吗？→ 可能只需要重新做 Step 2-4，hypothesis 可以复用。
