# Trace 数据设计：采什么、怎么采、为什么

---

## 1. 核心原则

> **Trace schema 决定了 invariant 的表达力上限——schema 中不存在的字段/维度，任何方法都无法检查。**

这不是当前 TrainAudit 的新观点（07_全流程思考.md 已阐述），但在新方案中更为关键：因为 predicate 是在 trace 数据上自动枚举的，schema 的丰富度直接决定能发现多少种 predicate。

---

## 2. Trace 的维度空间

Trace 数据是一个高维笛卡尔积上的记录：

```
trace ⊆ step × stage × parameter × rank_tuple × attribute
```

### 2.1 维度详解

#### step（时间维度）
- 每个 training step 的编号
- 用途：检测跨 step 的时序关系（monotonicity、reset pattern、accumulation）
- 关键问题：是否需要 micro-step 级别？
  - 如果框架使用 gradient accumulation，micro_step_id 是区分 "accumulate" 和 "sync" 的关键
  - Figure 1 的 DeepSpeed off-by-one bug 就发生在 micro_step 粒度

#### stage（训练阶段维度）
- 一个 step 内的不同时间点

| stage 名称 | 含义 | 可检查什么 |
|-----------|------|-----------|
| `before_forward` | forward 开始前 | 输入数据一致性、参数初始状态 |
| `after_forward` | forward 结束后 | activation 正确性（如果采集） |
| `after_backward` | backward 结束后 | 梯度是否已计算、gradient accumulation 状态 |
| `before_optimizer` | optimizer step 前 | 梯度是否已同步（all-reduce 是否完成） |
| `after_optimizer` | optimizer step 后 | 参数更新是否正确、DP 一致性 |
| `get_batch` | 数据加载后 | 数据是否正确分发到各 rank |

- 关键问题：stage 粒度决定了能检测的 bug 类型
  - 只有 `after_backward` + `after_optimizer` → 能检测梯度同步和参数更新问题
  - 加上 `before_optimizer` → 能检测 "梯度计算完成但同步前被覆盖" 的时序 bug

#### parameter（模型参数维度）
- 每个 named parameter（weight / bias / optimizer state）
- 需要记录的属性：
  - `param_name`：完整路径名（如 `transformer.layers.0.attention.query.weight`）
  - `param_group`：参数所属的组（用于分组检查）
  - `requires_grad`：是否需要梯度
  - `is_shared`：是否跨层共享（tied weights）

#### rank_tuple（并行拓扑维度）
- 不是一个单一的 rank_id，而是一个元组：

```
rank_tuple = (dp_rank, tp_rank, pp_rank, cp_rank, ep_rank)
```

- 这是 topology-aware 检查的基础
- 关键：必须能从 rank_tuple 推导出任意 rank group 的成员关系
  - `dp_group(rank)` = 所有与 rank 有相同 (tp, pp, cp, ep) 但不同 dp 的 rank 集合
  - `tp_group(rank)` = 所有与 rank 有相同 (dp, pp, cp, ep) 但不同 tp 的 rank 集合
  - 以此类推

#### attribute（观测属性维度）

| 属性 | 类型 | 用途 |
|------|------|------|
| `cksum` | uint64 | 精确一致性检查（hash of tensor） |
| `shape` | tuple | 形状一致性 |
| `dtype` | string | 类型一致性 |
| `norm` | float | L2 norm，数值范围检查 |
| `mean` | float | 均值，数值分布检查 |
| `std` | float | 标准差 |
| `min` / `max` | float | 极值检查 |
| `has_nan` | bool | NaN 检查 |
| `has_inf` | bool | Inf 检查 |
| `grad_cksum` | uint64 | 梯度的精确一致性 |
| `grad_norm` | float | 梯度的 L2 norm |

---

## 3. DuckDB Schema 设计

### 3.1 核心表：`coredump`

```sql
CREATE TABLE coredump (
    -- 时间维度
    step            INTEGER NOT NULL,
    micro_step      INTEGER NOT NULL,     -- gradient accumulation 内的子步
    stage           VARCHAR NOT NULL,      -- before_forward / after_backward / ...
    
    -- 参数维度
    param_name      VARCHAR NOT NULL,
    param_group     VARCHAR,               -- 参数组名
    requires_grad   BOOLEAN,
    is_shared       BOOLEAN DEFAULT FALSE,
    
    -- 拓扑维度
    global_rank     INTEGER NOT NULL,
    dp_rank         INTEGER NOT NULL,
    tp_rank         INTEGER NOT NULL,
    pp_rank         INTEGER NOT NULL,
    cp_rank         INTEGER DEFAULT 0,
    ep_rank         INTEGER DEFAULT 0,
    
    -- 观测属性
    cksum           BIGINT,
    shape           VARCHAR,
    dtype           VARCHAR,
    norm            DOUBLE,
    mean            DOUBLE,
    std             DOUBLE,
    min_val         DOUBLE,
    max_val         DOUBLE,
    has_nan         BOOLEAN DEFAULT FALSE,
    has_inf         BOOLEAN DEFAULT FALSE,
    grad_cksum      BIGINT,
    grad_norm       DOUBLE,
    
    -- 元数据
    timestamp_ns    BIGINT                 -- 采集时刻的 nanosecond timestamp
);
```

### 3.2 配置表：`topology`

```sql
CREATE TABLE topology (
    dp_size     INTEGER NOT NULL,
    tp_size     INTEGER NOT NULL,
    pp_size     INTEGER NOT NULL,
    cp_size     INTEGER DEFAULT 1,
    ep_size     INTEGER DEFAULT 1,
    world_size  INTEGER NOT NULL,
    zero_stage  INTEGER DEFAULT 0,
    
    -- 框架信息
    framework       VARCHAR,               -- megatron / deepspeed / olmo
    framework_version VARCHAR,
    
    -- 训练配置
    gradient_accumulation_steps INTEGER DEFAULT 1,
    use_distributed_optimizer   BOOLEAN DEFAULT FALSE,
    cpu_offload                 BOOLEAN DEFAULT FALSE,
    sequence_parallel           BOOLEAN DEFAULT FALSE
);
```

### 3.3 为什么这样设计

1. **rank_tuple 展开为独立列**（而非 JSON）：SQL 的 GROUP BY / WHERE 直接操作列，性能远优于 JSON extract
2. **micro_step 独立字段**：Figure 1 的 off-by-one bug 发生在 micro_step 粒度，不记录就无法检测
3. **topology 单独表**：predicate enumeration 需要知道当前配置来决定哪些维度的检查有意义（TP=1 时跳过所有 TP 相关 predicate）
4. **`requires_grad` / `is_shared`**：很多 invariant 的 precondition 依赖这些属性（frozen weights 没有 optimizer state）

---

## 4. 采集点设计（Hook 插入位置）

### 4.1 Megatron-LM 的 hook 位置

```
training_step()
├── get_batch()                          ← hook: get_batch
├── forward_step()
│   ├── [before first forward]           ← hook: before_forward
│   └── [after last forward]             ← hook: after_forward
├── backward_step()
│   └── [after backward completes]       ← hook: after_backward
├── [all-reduce / reduce-scatter]        ← （通信操作，不直接 hook）
├── optimizer.step()
│   ├── [before step]                    ← hook: before_optimizer
│   └── [after step]                     ← hook: after_optimizer
└── [logging / checkpointing]
```

### 4.2 每个 hook 采集什么

| Hook | 采集哪些参数 | 采集哪些属性 | 理由 |
|------|-------------|-------------|------|
| `before_forward` | all model params | cksum, shape, dtype | 确认 forward 前参数状态一致 |
| `after_backward` | all model params + grads | cksum, grad_cksum, grad_norm | 梯度计算完成后的状态 |
| `before_optimizer` | all model params + grads | cksum, grad_cksum | 确认 all-reduce 是否已完成 |
| `after_optimizer` | all model params + opt states | cksum, norm, mean, std | 参数更新后的完整快照 |
| `get_batch` | input tensors | cksum, shape | 数据分发正确性 |

### 4.3 采集频率与开销控制

全量采集每个 step 的所有参数 × 所有 stage 开销太大。策略：

1. **阶梯式采集**：
   - 前 N 步（如 100 步）：全量采集——用于建立 healthy baseline
   - 之后：每 K 步采集一次全量，其他步只采集 coarse check（聚合统计量）
   
2. **参数采样**：
   - 不是每个参数都采集——可以按层类型采样（每种 layer type 采集 1-2 个代表）
   - 但 optimizer state 必须全量采集（这是最常出 bug 的地方）

3. **属性分级**：
   - Level 0（必采）：cksum, has_nan, has_inf
   - Level 1（默认采）：norm, grad_norm, shape
   - Level 2（按需采）：mean, std, min, max, quantiles

---

## 5. 和当前 TrainAudit Data Collector 的差异

| 方面 | 当前 TrainAudit | 新方案 |
|------|----------------|--------|
| micro_step | 未明确区分 | 必须记录——micro_step 级 bug 是核心目标 |
| topology 独立表 | 隐式在 invariant precondition 中 | 显式表——predicate enumeration 依赖 |
| 采集频率 | 固定 | 阶梯式——前期密后期稀 |
| 属性丰富度 | cksum + shape + type 为主 | 增加统计量（norm/mean/std）——支撑数值范围类 predicate |

---

## 6. 开放问题

1. **cksum 的 hash 算法选择**：当前用什么？是否需要考虑 floating point 精度问题（bit-exact hash vs. approximate hash）？
2. **跨 step 的 trace 存储量**：如果每步采集 ~1000 个参数 × 5 个 stage × 8 个属性 × 64 个 rank = ~250 万行/step，DuckDB 能否高效处理数千步的数据？
3. **activation tensor 是否需要采集**：当前 schema 只覆盖 model params 和 optimizer states。activation 级别的 silent error 需要额外的采集点。
4. **通信操作的 trace**：是否需要记录 all-reduce / reduce-scatter 的输入输出？这能检测通信库本身的 bug。
