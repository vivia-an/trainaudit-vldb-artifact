# Runtime Detection：用 verified invariant 检测未知 trace

---

## 1. 运行时检测流程

训练开始后，对一个不知道有没有 silent error 的 trace 做检测：

```
输入：
  - Verified invariant library（来自 Layer 3）
  - 当前训练的 topology config
  - 实时 trace 数据（流式写入 DuckDB）

流程：
  Step A: Topology-aware pruning → 筛出适用的 invariant 子集
  Step B: 层级化执行（coarse → fine drill-down）
  Step C: Violation 分类
  Step D: 诊断报告生成
```

---

## 2. Step A: Topology-Aware Pruning

从 invariant library 中筛出适用于当前配置的子集：

```python
def prune_invariants(library, current_config):
    active = []
    for inv in library:
        # 检查 topology requirements
        if inv.preconditions.topology_requirements:
            if not matches(current_config, inv.preconditions.topology_requirements):
                continue  # 例：dp_size > 1，但当前 dp_size = 1
        
        # 检查 auto-discovered preconditions
        if not matches(current_config, inv.preconditions.auto_discovered):
            continue  # 例：zero_stage < 3，但当前 zero_stage = 3
        
        active.append(inv)
    
    return active
```

预期裁剪率：30-40%（和当前 TrainAudit 类似）。

---

## 3. Step B: 层级化执行

### 3.1 为什么需要层级化

如果 active set 有 150 条 invariant，每条对应一个 SQL 查询，每个 step 都跑 150 个 SQL → 开销过大。

### 3.2 两层执行策略

**Layer-Coarse：聚合检查（每个 step 都跑）**

从 invariant 库中提取一组**聚合 SQL**——不检查每个参数，而是检查整体：

```sql
-- Coarse check: DP 整体一致性
-- 所有参数的 cksum 拼接后再 hash，跨 DP rank 比较
SELECT step,
       COUNT(DISTINCT group_hash) AS distinct_groups
FROM (
    SELECT step, dp_rank,
           MD5(STRING_AGG(param_name || ':' || cksum::VARCHAR, ',' ORDER BY param_name)) AS group_hash
    FROM coredump
    WHERE stage = 'after_optimizer'
      AND requires_grad = true
    GROUP BY step, dp_rank, tp_rank, pp_rank
)
GROUP BY step
HAVING COUNT(DISTINCT group_hash) > 1
```

如果 coarse check 通过 → 该类别下所有 fine-grained invariant 跳过。

**Layer-Fine：精确检查（只在 coarse violation 时触发）**

```sql
-- Fine check: 具体哪个参数、哪个 rank 不一致
SELECT step, param_name, dp_rank, cksum
FROM coredump
WHERE stage = 'after_optimizer'
  AND requires_grad = true
  AND step = {violation_step}
ORDER BY param_name, dp_rank
```

### 3.3 Coarse-Fine 映射

| Coarse Check | 覆盖的 Fine Invariant 类别 |
|-------------|--------------------------|
| DP 整体一致性 | 所有 `dp` 维度的 `cross_rank_equality` invariant |
| TP shard 整体正确性 | 所有 `tp` 维度的 `cross_rank_partitioned` invariant |
| 参数变化检测 | 所有 `stage_changed` / `stage_unchanged` invariant |
| 数值范围检查 | 所有 `bounded_range` / `no_nan_inf` invariant |

预期效果：95% 的时间只跑 4-5 个 coarse SQL；只有异常时才展开 fine-grained。

---

## 4. Step C: Violation 分类

SQL 跑出 violation 后，不是每个都是真 bug：

| 类别 | 描述 | 处理 |
|------|------|------|
| **True violation** | cksum 确实不一致，来自代码 bug | 生成诊断报告 |
| **Step exclusion miss** | 在 step=0 或 warmup 期间的已知例外 | 检查是否在 `step_exclusions` 列表中 |
| **NULL artifact** | 某 rank 在某 step 没有采集到数据 | 检查 COUNT 是否符合预期 |
| **Config edge case** | precondition 没覆盖到的特殊配置组合 | 记录并反馈给 invariant library（用于下一轮 validation） |

分类逻辑：

```python
def classify_violation(violation, invariant, trace_meta):
    # 1. 是否在已知排除 step 中
    if violation.step in invariant.preconditions.step_exclusions:
        return "excluded_step"
    
    # 2. 是否有 NULL 数据
    expected_count = trace_meta.rank_count_for(invariant.rank_dimension)
    if violation.actual_count < expected_count:
        return "null_artifact"
    
    # 3. 是否是新遇到的配置边界
    if not invariant.validation_evidence.covers(trace_meta.config):
        return "untested_config"  # 降低置信度但仍报告
    
    # 4. 真实 violation
    return "true_violation"
```

---

## 5. Step D: 诊断报告

真实 violation 触发结构化诊断报告：

```
╔══════════════════════════════════════════════════════╗
║  VIOLATION DETECTED                                   ║
╠══════════════════════════════════════════════════════╣
║                                                        ║
║  Invariant:  INV-042-E1-cksum                         ║
║  Type:       cross_rank_equality (dp dimension)        ║
║  Semantic:   DP ranks 间 optimizer state 应一致        ║
║              (ZeRO-1/2 不 shard storage)               ║
║                                                        ║
║  Violation Detail:                                     ║
║  ├─ Step: 142                                          ║
║  ├─ Stage: after_optimizer                             ║
║  ├─ Parameter: final_layernorm.bias.exp_avg            ║
║  ├─ Distinct cksum count: 2 (expected: 1)              ║
║  │                                                     ║
║  ├─ Rank breakdown:                                    ║
║  │   dp_rank=0: cksum=0xA3F2...                       ║
║  │   dp_rank=1: cksum=0xA3F2...                       ║
║  │   dp_rank=2: cksum=0x7B1D...  ← DIVERGED           ║
║  │   dp_rank=3: cksum=0xA3F2...                       ║
║  │                                                     ║
║  ├─ First divergence at step: 138                      ║
║  │   (checked via cross-step backtrack)                ║
║  │                                                     ║
║  └─ Likely root cause:                                 ║
║      dp_rank=2 的 optimizer state 从 step 138 开始     ║
║      偏离其他 rank，可能原因：                           ║
║      - Missing all-reduce in optimizer step             ║
║      - Incorrect gradient accumulation reset            ║
║                                                        ║
╚══════════════════════════════════════════════════════╝
```

### 5.1 Backtrack：定位首次偏离

检测到 step=142 的 violation 后，回溯查找首次偏离的 step：

```sql
-- 找到该参数首次出现 DP 不一致的 step
SELECT MIN(step) AS first_divergence
FROM (
    SELECT step, COUNT(DISTINCT cksum) AS dc
    FROM coredump
    WHERE param_name = 'final_layernorm.bias.exp_avg'
      AND stage = 'after_optimizer'
    GROUP BY step, tp_rank, pp_rank
    HAVING COUNT(DISTINCT cksum) > 1
)
```

### 5.2 Cross-Stage 诊断

在 first_divergence step 上，检查不同 stage 的状态：

```sql
-- Step 138 上，diverged rank 在各 stage 的状态
SELECT stage, cksum, grad_cksum
FROM coredump
WHERE step = 138
  AND param_name = 'final_layernorm.bias.exp_avg'
  AND dp_rank = 2
ORDER BY stage
```

通过对比 diverged rank 和 normal rank 在各 stage 的差异，缩小根因范围：
- 如果 `after_backward` 就已经不同 → gradient 计算或 accumulation 出错
- 如果 `after_backward` 相同但 `after_optimizer` 不同 → optimizer step 出错
- 如果 `before_optimizer` 相同但 `after_optimizer` 不同 → all-reduce 或 optimizer 逻辑出错

---

## 6. 开销预估

| 阶段 | 每 step 开销 | 说明 |
|------|-------------|------|
| Trace 采集 | ~3-5% | 和当前 TrainAudit 相同 |
| Coarse SQL 执行 | ~0.5-1% | 4-5 个聚合 SQL，在 DuckDB 上毫秒级 |
| Fine SQL 执行 | ~0-2%（平均） | 仅在 coarse violation 时触发 |
| **总计** | **~5%** | 和当前 TrainAudit 论文声称的开销一致 |

---

## 7. 开放问题

1. **实时性 vs 批量**：每个 step 都检查（实时）还是攒 N 个 step 再检查（批量）？批量能 amortize DuckDB query 开销，但延迟增加。
2. **Coarse check 的粒度**：当前设计是"整个 DP group 的整体 hash"——如果 1000 个参数中只有 1 个偏离，coarse check 一定能检测到吗？→ 能，因为 hash 不同。但 hash 碰撞的概率需要分析。
3. **诊断的自动化程度**：Step D 的 "likely root cause" 目前是模板化的。是否需要 LLM 参与根因分析？→ 可以作为可选功能，但核心检测不依赖 LLM。
