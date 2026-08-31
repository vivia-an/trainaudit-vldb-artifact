# Predicate Enumeration：自动枚举候选 invariant

---

## 1. 核心思路

收到一个 semantic hypothesis 后，系统在其指定的子空间内**自动枚举**所有可能的 predicate。

枚举是**确定性的、不依赖 LLM 的**——基于一组预定义的 predicate template。

---

## 2. Predicate Template 库

### 2.1 Cross-Rank Equality Templates

当 hypothesis 的 `relation_type = cross_rank_equality` 时，枚举以下候选：

```sql
-- Template E1: 精确一致（cksum）
SELECT step, param_name,
       COUNT(DISTINCT {field}) AS distinct_count
FROM coredump
WHERE stage = '{stage}'
  AND {param_filter}
GROUP BY step, param_name, {other_rank_dims}
HAVING COUNT(DISTINCT {field}) > 1

-- Template E2: 数值近似一致（norm/mean/std）
SELECT step, param_name,
       MAX({field}) - MIN({field}) AS spread
FROM coredump
WHERE stage = '{stage}'
  AND {param_filter}
GROUP BY step, param_name, {other_rank_dims}
HAVING MAX({field}) - MIN({field}) > {epsilon}

-- Template E3: 条件一致（只在特定 step 类型）
-- 针对 gradient accumulation：只在 sync step 检查
SELECT step, param_name,
       COUNT(DISTINCT {field}) AS distinct_count
FROM coredump
WHERE stage = '{stage}'
  AND {param_filter}
  AND micro_step = (SELECT gradient_accumulation_steps - 1 FROM topology)
GROUP BY step, param_name, {other_rank_dims}
HAVING COUNT(DISTINCT {field}) > 1
```

其中 `{field}`、`{stage}`、`{param_filter}`、`{other_rank_dims}` 从 hypothesis 中填充。

**枚举维度**：对每个 hypothesis，系统生成 E1/E2/E3 三个变体 × 可能的 field（cksum / norm / mean）= ~9 个候选。

### 2.2 Cross-Rank Partition Templates

当 `relation_type = cross_rank_partitioned` 时：

```sql
-- Template P1: 各 rank 的值互不相同
SELECT step, param_name,
       COUNT(DISTINCT {field}) AS distinct_count,
       COUNT(*) AS total_count
FROM coredump
WHERE stage = '{stage}'
  AND {param_filter}
GROUP BY step, param_name, {other_rank_dims}
HAVING COUNT(DISTINCT {field}) < COUNT(*)
-- 如果 distinct < total，说明有 rank 重复了（不应该）

-- Template P2: 各 rank 的 shape 按某维度分割
SELECT step, param_name,
       -- 检查 shape 的某个维度是否被 tp_size 整除
FROM coredump
WHERE stage = '{stage}'
  AND {param_filter}
```

### 2.3 Cross-Step Templates

当 `relation_type` 涉及跨 step 关系时：

```sql
-- Template T1: 单调递增/递减
SELECT c1.step, c1.param_name,
       c2.{field} - c1.{field} AS delta
FROM coredump c1
JOIN coredump c2 ON c1.param_name = c2.param_name
  AND c1.global_rank = c2.global_rank
  AND c2.step = c1.step + 1
  AND c1.stage = '{stage}' AND c2.stage = '{stage}'
WHERE delta < 0  -- 违反单调递增

-- Template T2: 周期性重置
SELECT step, param_name, {field}
FROM coredump
WHERE stage = '{stage}'
  AND {param_filter}
  AND step % {period} = 0
  AND {field} != {reset_value}
-- period 从 topology.gradient_accumulation_steps 获取

-- Template T3: 相邻 step 变化有界
SELECT c1.step, c1.param_name,
       ABS(c2.{field} - c1.{field}) AS abs_delta
FROM coredump c1
JOIN coredump c2 ON ...
WHERE ABS(c2.{field} - c1.{field}) > {threshold}
```

### 2.4 Cross-Stage Templates

```sql
-- Template S1: 两个 stage 之间值不变
SELECT step, param_name, global_rank
FROM coredump c1
JOIN coredump c2 ON c1.step = c2.step
  AND c1.param_name = c2.param_name
  AND c1.global_rank = c2.global_rank
WHERE c1.stage = '{stage_before}'
  AND c2.stage = '{stage_after}'
  AND c1.{field} != c2.{field}

-- Template S2: 两个 stage 之间值一定改变
SELECT step, param_name, global_rank
FROM coredump c1
JOIN coredump c2 ON ...
WHERE c1.stage = '{stage_before}'
  AND c2.stage = '{stage_after}'
  AND c1.{field} = c2.{field}
  AND {param_filter}  -- e.g. requires_grad = true
```

### 2.5 Value-Level Templates

```sql
-- Template V1: 无 NaN/Inf
SELECT step, param_name, global_rank
FROM coredump
WHERE (has_nan = TRUE OR has_inf = TRUE)
  AND stage = '{stage}'

-- Template V2: Norm 在合理范围内
SELECT step, param_name, global_rank, norm
FROM coredump
WHERE stage = '{stage}'
  AND {param_filter}
  AND (norm > {upper_bound} OR norm < {lower_bound})
```

---

## 3. 枚举算法

```python
def enumerate_predicates(hypothesis, schema):
    """
    输入: semantic hypothesis + trace schema
    输出: 候选 predicate 列表
    """
    candidates = []
    
    # 1. 根据 relation_type 选择 template 族
    templates = TEMPLATE_REGISTRY[hypothesis.relation_type]
    
    # 2. 确定 field 候选集
    if hypothesis.entities.target_fields:
        fields = hypothesis.entities.target_fields
    else:
        fields = infer_relevant_fields(hypothesis.relation_type)
    
    # 3. 对每个 template × 每个 field 生成候选
    for template in templates:
        for field in fields:
            sql = template.instantiate(
                field=field,
                stage=hypothesis.stage,
                param_filter=hypothesis.entities.param_filter,
                rank_dimension=hypothesis.rank_dimension,
            )
            candidates.append(Predicate(
                sql=sql,
                source_hypothesis=hypothesis.hypothesis_id,
                template=template.name,
                field=field,
            ))
    
    # 4. 如果 hypothesis 有 precondition_hints，
    #    生成 WITH / WITHOUT precondition 的变体
    for hint in hypothesis.precondition_hints:
        where_clause = hint_to_sql_condition(hint)
        if where_clause:
            for base in candidates.copy():
                candidates.append(base.add_where(where_clause))
    
    return candidates
```

### 3.1 枚举规模估算

对一个典型的 hypothesis：
- 3 个 template 变体
- 2-3 个 field 候选
- 2-3 个 precondition 变体（有/无某条件）
- ≈ 3 × 3 × 3 = **~27 个候选 predicate**

如果 LLM 生成 100 个 hypothesis，总共约 **~2700 个候选**。

这个规模在 DuckDB 上是完全可行的——每个 SQL 查询在百万行 trace 上执行时间在毫秒级。

---

## 4. 参数化与阈值

某些 template 包含需要确定的参数（epsilon、threshold、upper_bound 等）。

策略：**不预设，从 healthy trace 中统计确定**。

```python
def determine_threshold(field, healthy_trace):
    """从 healthy trace 的统计分布确定阈值"""
    values = query(f"SELECT {field} FROM coredump WHERE ...")
    
    # 用 percentile 而非 hard-coded 值
    return {
        "epsilon": percentile(values, 99.9) - percentile(values, 0.1),
        "upper_bound": percentile(values, 99.9) * safety_margin,
        "lower_bound": percentile(values, 0.1) / safety_margin,
    }
```

这确保阈值是 data-driven 的，不是 LLM 猜的。

---

## 5. 和当前 TrainAudit 的 SQL 编译对比

| 方面 | 当前 TrainAudit | 新方案 |
|------|----------------|--------|
| SQL 谁写 | LLM（通过 SQL Agent） | 模板引擎自动生成 |
| SQL 正确性 | 依赖 LLM 对 schema 的理解 | 模板经过预验证，保证语法正确 |
| 覆盖多种解读 | 一条 invariant → 一条 SQL | 一条 hypothesis → 多条候选 SQL |
| 阈值/参数 | LLM 猜测或 hard-coded | 从 healthy trace 统计确定 |
| 失败模式 | SQL 编译失败 / 语义错误 | 模板保证编译成功；语义由 data validation 判断 |

---

## 6. 开放问题

1. **Template 库的完备性**：当前的 template 族能覆盖多少种已知 bug？需要对照 B1-B15 逐个验证。
2. **组合 predicate**：有些 invariant 需要 AND/OR 组合多个 atomic check（如"DP 一致 AND 非 frozen AND ZeRO < 3"）。枚举组合会指数增长——需要剪枝策略。
3. **复杂时序关系**：某些 bug 涉及多 step 的复杂时序（如"梯度在连续 K 个 micro_step 累积，第 K+1 步 reset"）。单条 SQL 可能不够表达——需要 window function 或 multi-query pattern。
4. **性能**：2700 个候选 SQL 在 healthy trace 上跑一遍需要多长时间？需要实验验证。
