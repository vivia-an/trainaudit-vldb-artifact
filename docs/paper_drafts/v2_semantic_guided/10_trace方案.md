# Trace 方案：基于 128 个真实 Bug 的数据需求分析

> 本方案完全从 bug 数据出发，不是"我们想采什么"，而是"要检测这些 bug 需要什么"。

---

## 1. 核心数据：从 bug 反推

128 个 bug 的 `required_trace_fields` 统计显示，只需要 **27 个字段** 就能覆盖 74% 的 bug（≥50% 字段匹配）。这 27 个字段分为 7 层，按增量覆盖率排列：

| 层级 | 新增字段 | 累计字段数 | 覆盖率（≥50%匹配） | 边际增益 |
|------|---------|-----------|-------------------|---------|
| **Tier 0 基础** | cksum, param_name, dp_rank, tp_rank, pp_rank, step, stage, dtype, shape | 9 | 30% | — |
| **Tier 1 梯度** | grad_norm, grad_cksum | 11 | 41% | +11% |
| **Tier 2 Loss** | loss_value | 13 | 51% | +10% |
| **Tier 3 训练状态** | learning_rate, micro_step_id, iteration | 17 | 59% | +8% |
| **Tier 4 Optimizer** | optimizer_state_cksum | 18 | 61% | +2% |
| **Tier 5 拓扑** | cp_rank, ep_rank, world_size, zero_stage, offload_enabled | 23 | 66% | +5% |
| **Tier 6 额外** | num_tokens, layer_id, has_nan, has_inf | 27 | 74% | +8% |

**结论**：Tier 0-3（17 个字段）覆盖 59%，是性价比最高的组合。全部 27 个字段覆盖 74%。

---

## 2. 具体方案：采什么、在哪采、怎么采

### 2.1 每个 Hook 点采集的字段

#### Hook 1: `after_forward`（覆盖 48 个 bug，最多）

对**每个 named parameter** 采集：

| 字段 | 类型 | 计算方式 | 覆盖的 bug 类型 |
|------|------|---------|----------------|
| cksum | uint64 | `hash(param.data)` | 跨 rank 一致性、跨 step 变化 |
| shape | str | `str(param.shape)` | shape mismatch |
| dtype | str | `str(param.dtype)` | dtype 不匹配 |
| has_nan | bool | `param.isnan().any()` | 数值爆炸 |
| has_inf | bool | `param.isinf().any()` | 数值爆炸 |

对**全局标量** 采集一次：

| 字段 | 类型 | 来源 | 覆盖的 bug 类型 |
|------|------|------|----------------|
| loss_value | float | `loss.item()` | loss 计算错误、异常跳变 |
| num_tokens | int | 当前 batch 的有效 token 数 | loss normalization 错误 |

#### Hook 2: `after_backward`（覆盖 37 个 bug）

对**每个有梯度的 parameter** 采集：

| 字段 | 类型 | 计算方式 | 覆盖的 bug 类型 |
|------|------|---------|----------------|
| grad_cksum | uint64 | `hash(param.grad)` | 梯度同步错误 |
| grad_norm | float | `param.grad.norm().item()` | 梯度爆炸/消失、normalization 错误 |

#### Hook 3: `after_optimizer`（覆盖 19 个 bug）

对**每个 named parameter** 采集：

| 字段 | 类型 | 计算方式 | 覆盖的 bug 类型 |
|------|------|---------|----------------|
| cksum | uint64 | `hash(param.data)` | 参数更新正确性 |
| optimizer_state_cksum | uint64 | `hash(optimizer.state[param]['exp_avg'])` | optimizer state 一致性 |

对**全局标量** 采集一次：

| 字段 | 类型 | 来源 | 覆盖的 bug 类型 |
|------|------|------|----------------|
| learning_rate | float | `optimizer.param_groups[0]['lr']` | LR 调度错误 |

### 2.2 每条记录的公共字段

不管哪个 hook，每条记录都带：

| 字段 | 类型 | 来源 |
|------|------|------|
| step | int | 当前 training step |
| micro_step_id | int | gradient accumulation 内的子步编号 |
| stage | str | hook 名称（after_forward / after_backward / after_optimizer） |
| param_name | str | 参数的完整名称 |
| global_rank | int | 当前进程的全局 rank |
| dp_rank | int | data parallel rank |
| tp_rank | int | tensor parallel rank |
| pp_rank | int | pipeline parallel rank |
| cp_rank | int | context parallel rank |
| ep_rank | int | expert parallel rank |

### 2.3 Topology 表（一次性采集，训练开始时）

| 字段 | 类型 | 说明 |
|------|------|------|
| dp_size | int | DP 组大小 |
| tp_size | int | TP 组大小 |
| pp_size | int | PP 组大小 |
| cp_size | int | CP 组大小 |
| ep_size | int | EP 组大小 |
| world_size | int | 总 GPU 数 |
| zero_stage | int | ZeRO stage (0/1/2/3) |
| gradient_accumulation_steps | int | 梯度累积步数 |
| offload_enabled | bool | CPU offload 是否开启 |
| max_grad_norm | float | 梯度裁剪阈值 |
| framework | str | 框架名+版本 |

---

## 3. DuckDB Schema

```sql
CREATE TABLE coredump (
    -- 时间
    step            INTEGER NOT NULL,
    micro_step      INTEGER NOT NULL,
    stage           VARCHAR NOT NULL,  -- after_forward / after_backward / after_optimizer
    
    -- 参数标识
    param_name      VARCHAR NOT NULL,
    
    -- 拓扑
    global_rank     INTEGER NOT NULL,
    dp_rank         INTEGER NOT NULL,
    tp_rank         INTEGER NOT NULL,
    pp_rank         INTEGER NOT NULL,
    cp_rank         INTEGER DEFAULT 0,
    ep_rank         INTEGER DEFAULT 0,
    
    -- 参数属性（after_forward + after_optimizer 采集）
    cksum           BIGINT,
    shape           VARCHAR,
    dtype           VARCHAR,
    has_nan         BOOLEAN,
    has_inf         BOOLEAN,
    
    -- 梯度属性（after_backward 采集）
    grad_cksum      BIGINT,
    grad_norm       DOUBLE,
    
    -- optimizer 属性（after_optimizer 采集）
    opt_state_cksum BIGINT,
    
    -- 全局标量（每个 stage 一条，param_name = '__global__'）
    loss_value      DOUBLE,
    learning_rate   DOUBLE,
    num_tokens      INTEGER
);

CREATE TABLE topology (
    dp_size         INTEGER NOT NULL,
    tp_size         INTEGER NOT NULL,
    pp_size         INTEGER NOT NULL,
    cp_size         INTEGER DEFAULT 1,
    ep_size         INTEGER DEFAULT 1,
    world_size      INTEGER NOT NULL,
    zero_stage      INTEGER DEFAULT 0,
    grad_accum_steps INTEGER DEFAULT 1,
    offload_enabled BOOLEAN DEFAULT FALSE,
    max_grad_norm   DOUBLE,
    framework       VARCHAR
);
```

---

## 4. 开销估算

假设：1000 个参数，8 个 rank，每 step 3 个 stage

### 每 step 数据量

| 项目 | 计算 | 数据量 |
|------|------|--------|
| after_forward | 1000 params × 8 ranks × 5 fields | 40,000 条 |
| after_backward | 1000 params × 8 ranks × 2 fields | 16,000 条 |
| after_optimizer | 1000 params × 8 ranks × 2 fields | 16,000 条 |
| 全局标量 | 3 stages × 8 ranks × 3 fields | 72 条 |
| **合计** | | **~72,000 条/step** |

每条约 200 bytes → **~14 MB/step**

### 计算开销

| 操作 | 估算 | 占比 |
|------|------|------|
| cksum 计算（hash 1000 个 tensor × 2 次） | ~2-3% | 主要开销 |
| grad_norm 计算（1000 个 tensor） | ~0.5% | 小 |
| DuckDB 写入（72K 行） | ~0.5-1% | 批量写入 |
| **总计** | **~3-5%** | |

### 降低开销的手段

| 手段 | 效果 | 代价 |
|------|------|------|
| 参数采样：只采 10% 参数 | 开销降到 ~0.5% | 可能漏检某些参数的 bug |
| 降低频率：每 K 步采集一次 | 开销降到 ~3%/K | 检测延迟增加 K 步 |
| 去掉 after_forward | 开销减少约 40% | 丢失 48 个 bug 的检测信号 |
| 异步写入 DuckDB | 写入开销几乎为 0 | 需要额外内存 buffer |

**推荐配置**：
- **每步全量采集，只保留最近 2 步数据**（滑动窗口）
- 理由：silent error 是代码 bug（确定性的），不是随机硬件故障。如果 bug 存在，每一步都会触发。检测只需要：
  - 当前步的 cross-rank 比较（1 步）
  - 当前步 vs 上一步的 cross-step 比较（2 步）
- 不需要长期 baseline 或统计阈值
- 存储开销：只保留 2 步 ≈ ~28 MB（可接受）
- 预期计算开销：**~3-5%**（每步都采，但不需要历史积累）

---

## 5. 这个方案覆盖了什么、覆盖不了什么

### 覆盖的 bug 类型（74%，~95 个）

| 类型 | 用哪些字段检测 | 典型 bug |
|------|--------------|---------|
| 跨 rank 参数不一致 | cksum × dp/tp/pp_rank | M-005 router weight 不同步 |
| 梯度同步错误 | grad_cksum × dp_rank | M-001 embedding gradient 错误 |
| 梯度异常 | grad_norm 跨 step 比较 | M-004 gradient explosion |
| Loss 异常 | loss_value 跨 step 比较 | M-016 checkpoint resume loss 跳变 |
| Dtype 不匹配 | dtype × rank | B3 FP16/BF16 mismatch |
| Shape 不匹配 | shape × rank | B9 Q/K/V shape mismatch |
| 参数不更新 | cksum 跨 step 不变 | M-021 expert 参数永不更新 |
| LR 错误 | learning_rate 跨 step | M-008 LR 始终为 0 |
| Optimizer state 不一致 | opt_state_cksum × dp_rank | B10 ZeRO-2 micro_step 错误 |
| 数值爆炸 | has_nan, has_inf | 各种 NaN/Inf bug |

### 可扩展覆盖的 bug 类型（额外 ~28 个，新增 4 类采集点）

| 类型 | 新增采集 | 成本 | 覆盖 bug 数 |
|------|---------|------|------------|
| Attention 逻辑错误 | attention output 的 cksum（在 attention 层加 hook） | 中（每层一次 hash） | ~8 |
| 数据加载错误 | batch index / input tensor hash（在 dataloader 层加 hook） | 极低 | ~5 |
| Checkpoint 损坏 | save/load 时对 state_dict 做 hash | 极低（只在 ckpt 时触发） | ~10 |
| RNG state 不一致 | `torch.cuda.get_rng_state()` 的 hash | 极低 | ~5 |

加上这 4 类后总覆盖率从 74% 提升到约 **90%**。

对应的新增 hook：

| Hook | 触发时机 | 采集内容 |
|------|---------|---------|
| `after_attention` | 每层 attention 计算后 | attention_output_cksum |
| `after_data_load` | 每个 batch 加载后 | batch_index, input_cksum |
| `after_checkpoint_save` | checkpoint 保存后 | state_dict_cksum |
| `after_checkpoint_load` | checkpoint 加载后 | state_dict_cksum |
| （随 after_forward 一起采） | 每步 | rng_state_cksum |

### 真正覆盖不了的 bug 类型（~5 个）

| 类型 | 为什么 | 典型 bug |
|------|--------|---------|
| Sub-percent 数值偏差 | 所有 rank 有相同的错误值，跨 rank 比较无法发现。问题不在采集，在于没有 correctness oracle——无法判断"这个值对不对" | B14, B15 |

这类 bug 必须有 reference implementation 精确对比才能发现（TTrace 的思路），属于方法论边界。

---

## 6. 和当前 TrainAudit Data Collector 的差异

| 方面 | 当前 TrainAudit | 本方案 |
|------|----------------|--------|
| Hook 数量 | 主要 after_backward + after_optimizer | **增加 after_forward**（覆盖最多 bug 的 hook） |
| 梯度信息 | 有 grad_cksum | **增加 grad_norm**（覆盖 29 个 bug） |
| Loss | 未采集 | **增加 loss_value**（覆盖 22 个 bug） |
| LR | 未采集 | **增加 learning_rate**（覆盖 6 个 bug） |
| Micro step | 未明确 | **增加 micro_step_id**（覆盖 6 个 bug） |
| Optimizer state | 未采集 | **增加 opt_state_cksum**（覆盖 7 个 bug） |
| 拓扑 | 有 dp/tp/pp_rank | **增加 cp_rank, ep_rank**（覆盖 CP/EP 相关 bug） |
| 采集频率 | 每步全量 | **阶梯式：前期密后期稀** |
