# Megatron Trace 覆盖率实验记录

> 基于 42 个 Megatron-LM 真实 silent error，逐层增加 trace 侵入性，测量 bug 覆盖率变化。
> 数据来源：`exp/data/megatron_silent_errors.json`
> 代码实现：`exp/sdctrace/auto.py`（L0）、`exp/sdctrace/adapters/megatron_enhanced.py`（增强版）

---

## 1. 实验设计

### 目标
量化"trace 侵入程度"和"bug 覆盖率"之间的关系，找到最优的工程投入点。

### 方法
- 将 trace 方案分为 5 个侵入层级（L0 → L4）
- 对每个层级，统计它能采集的字段集合
- 对 42 个 Megatron bug 的 `required_trace_fields`，计算每个层级的覆盖率
- 覆盖率定义：bug 所需字段 100% 被包含在该层级的可采集字段中

### Bug 数据
42 个 bug 来自 Megatron-LM GitHub issue/PR，涵盖：
- gradient_sync (6)、numerical (9)、communication (5)、sharding (5)
- dtype (4)、control_flow (5)、checkpoint (4)、optimizer_state (3)

---

## 2. 各层级定义与可采集字段

### L0：零侵入（auto_trace，2 行代码）

利用 PyTorch 原生 optimizer hook，不改任何框架代码。

**可采集字段：**
```
cksum, param_name, dp_rank, tp_rank, pp_rank, cp_rank, ep_rank,
shape, dtype, has_nan, has_inf, grad_cksum, grad_norm,
loss_value, learning_rate, iteration, opt_state_cksum, rng_state_cksum
```

**采集方式：**
- optimizer pre-step hook → grad_cksum, grad_norm（梯度还未被 zero_grad）
- optimizer post-step hook → cksum, opt_state_cksum, learning_rate
- tracer.step() → cksum, shape, dtype, has_nan, has_inf, loss_value, rng_state_cksum

**改动量：** 2 行代码
**迁移成本：** 零（任何框架通用）

### L1：+统计量（5 行代码）

在 L0 基础上，对每个参数和梯度额外计算 norm/mean。

**新增字段：**
```
param.norm, param.mean, grad.mean, grad.has_nan, grad.has_inf
```

**采集方式：** 在 L0 的同一个 hook 中多算几个 `.norm().item()` / `.mean().item()`
**开销特点：** GPU 上算标量，不需要 GPU→CPU 全量拷贝，开销反而低于 L0 的 hash

### L2：+Tier A（topology 扩展，零开销）

在 L1 基础上，训练开始时一次性记录完整的 Megatron 配置信息。

**新增字段：**
```
cp_size, tp_size, pp_size, vp_size, vocab_size, num_layers,
total_params, recompute_enabled, seq_len, save_interval
```

**采集方式：** 从 `args` 对象一次性读取，写入 DuckDB
**开销：** 零（只在初始化时执行一次）

### L3：+Tier B/C（标量 + activation hook，~150 行代码）

在 L2 基础上增加：
- 每步额外标量：aux_loss, num_tokens, loss_scale, lr_schedule_params
- per-layer forward hook：attention output cksum, router routing_probs/routing_indices

**新增字段：**
```
aux_loss, num_tokens, loss_scale, micro_step_id,
lr_schedule_params (max_lr/min_lr/initial_lr),
attention_output_cksum, routing_probs, routing_indices,
routing_map_cksum, softmax_scale, position_ids, position_offset, param_value
```

**采集方式：**
- `tracer.step(aux_loss=..., num_tokens=...)` 传入额外标量
- `model.register_forward_hook()` 对 attention/router/RoPE 层自动注册
- 从 optimizer/scheduler 对象读取 loss_scale 和 lr 参数

**改动量：** 3 行用户代码 + ~150 行 `megatron_enhanced.py`
**迁移成本：** 中等（需要知道模型的 layer 命名模式）

### L4：深度侵入（未实现）

需要 hook NCCL 通信层或修改 Megatron 训练循环。

**新增字段：**
```
sync_order, sample_idx, doc_idx
```

**改动量：** 200+ 行，和框架强耦合

---

## 3. 覆盖率实验结果

### 主结果表

| 层级 | 可采集字段数 | 100% 覆盖 bug 数 | 覆盖率 | 增量 | 用户代码改动 |
|------|------------|-----------------|--------|------|------------|
| **L0** | 18 | 20 / 42 | **48%** | — | 2 行 |
| **L1** | 23 | 20 / 42 | **48%** | +0% | 5 行 |
| **L2** | 33 | 30 / 42 | **71%** | +23% | 5 行 |
| **L3** | 47 | 40 / 42 | **95%** | +24% | 3 行 |
| **L4** | 50 | 41 / 42 | **98%** | +3% | 200+ 行 |

注：
- L0→L1 覆盖率不变是因为 norm/mean 不是任何 bug 的"唯一必需字段"，但它们使 bounded_change 类 bug 可被检测（详见下方检测能力分析）
- L2 的大跃升来自 topology 配置信息——很多 bug 的 precondition 判断依赖这些字段

### 按 ≥50% 字段匹配统计

| 层级 | ≥50% 覆盖 | 覆盖率 |
|------|----------|--------|
| L0 | 35 / 42 | 83% |
| L1 | 35 / 42 | 83% |
| L2 | 39 / 42 | 93% |
| L3 | 41 / 42 | 98% |
| L4 | 42 / 42 | 100% |

### 不可覆盖的 bug

| Bug | 缺失字段 | 原因 | 需要的侵入 |
|-----|---------|------|-----------|
| M-019 | sync_order | 需要 hook 通信操作的执行顺序 | L4（NCCL hook） |
| M-027 | sample_idx, doc_idx | 需要 hook dataloader 内部 | L4（数据加载 hook） |

---

## 4. 关键发现

### 发现 1：L2→L3 是 Megatron 的 sweet spot

从 71% 跳到 95%，只需要增加 activation hook 和几个标量——用户代码仍然只有 3 行。

### 发现 2：topology 配置信息的价值被低估

L1→L2 的 +23% 完全来自一次性写入的配置信息（零运行时开销）。很多 bug 的检测不需要新的运行时数据，只需要知道"当前是什么配置"来判断 invariant 是否适用。

### 发现 3：L0 的 hash 方案存在"虚高"覆盖

L0 的 48% 是 100% 字段匹配，看起来不低。但 hash 只能做 binary 判断（相同/不同），对 bounded_change 类 bug（占 42 个中的 ~15 个）实际上无法有效检测——知道"值变了"但不知道"变化是否正常"。加上 norm/mean（L1）后才能真正检测这类 bug。

### 发现 4：最后 5% 的成本极高

从 95% 到 98% 需要 L4 级别的侵入（hook NCCL 或 dataloader），代码量从 ~150 行暴增到 200+ 行，且完全和框架版本耦合。对于论文实验而言，95% 已经是一个很好的结果。

---

## 5. 开销对比

| 层级 | 主要开销 | 估算 |
|------|---------|------|
| L0 | GPU→CPU hash 拷贝 | ~5-10% |
| L1 | L0 + GPU 上算 norm/mean（替代部分 hash 可降低开销） | ~3-5% |
| L2 | L1 + 零（配置一次性写入） | ~3-5% |
| L3 | L2 + activation hash（额外 ~数十个 tensor） | ~4-6% |
| L4 | L3 + 通信 hook | ~6-10% |

**L1 的开销可能低于 L0**：如果用 norm/mean 替代部分参数的 full hash（对 bounded_change 检测效果更好），可以避免大量 GPU→CPU 拷贝。

---

## 6. 实验图设计

### 图 1：侵入层级 vs 覆盖率（柱状图 + 折线）

X 轴：L0, L1, L2, L3, L4
左 Y 轴（柱状）：用户代码改动行数（2, 5, 5, 3, 200+）
右 Y 轴（折线）：bug 覆盖率（48%, 48%, 71%, 95%, 98%）

关键标注：
- L2→L3 区间标注 "sweet spot: +24% coverage, 3 lines of code"
- L3→L4 区间标注 "diminishing returns: +3% coverage, 200+ lines"

### 图 2：bug 类型 × 层级 热力图

```
                    L0    L1    L2    L3    L4
gradient_sync       ●●●   ●●●   ●●●●  ●●●●● ●●●●●●  (6 bugs)
numerical           ●●    ●●●   ●●●●● ●●●●● ●●●●●●● (9 bugs)
communication       ●●    ●●    ●●●   ●●●●  ●●●●●   (5 bugs)
sharding            ●●●   ●●●   ●●●●  ●●●●● ●●●●●   (5 bugs)
dtype               ●●●   ●●●   ●●●●  ●●●●  ●●●●    (4 bugs)
control_flow        ●     ●     ●●●   ●●●●● ●●●●●   (5 bugs)
checkpoint          ●●    ●●    ●●●   ●●●●  ●●●●    (4 bugs)
optimizer_state     ●●    ●●    ●●●   ●●●   ●●●     (3 bugs)
```

### 图 3：已验证的端到端检测

| Bug | 层级 | 复现状态 | 检测结果 |
|-----|------|---------|---------|
| M-005 (router weight 不同步) | L0 | ✅ 已复现 | buggy: detected, fixed: clean |
| ... | ... | 待复现 | ... |

（每复现一个 bug 就补一行）

---

## 7. 代码位置

| 文件 | 层级 | 说明 |
|------|------|------|
| `exp/sdctrace/auto.py` | L0-L1 | 通用版 auto_trace |
| `exp/sdctrace/adapters/megatron_enhanced.py` | L2-L3 | Megatron 增强版 |
| `exp/sdctrace/core/` | 基础 | hash, schema, storage, collector |
| `exp/data/reproductions/M-005/` | 验证 | 第一个端到端复现 |
| `exp/data/megatron_silent_errors.json` | 数据 | 42 个 bug 的完整记录 |
