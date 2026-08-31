# 30. D1 集合整合与扩展实验 Brief（给实验 agent）

> 目标：把当前 D1 17 bug + E2E 26 bug 整合为单一统一集合 D1' = **17 bug**，满足"每类 ≤2 个 / 覆盖 12/13 类 / 三方工具都能跑"。
> 上游：[28_392_extension_brief.md](28_392_extension_brief.md)、[29_staircase_392_brief.md](29_staircase_392_brief.md)
> 完成日期：待定。
> 论文同步集成由我做（不在本 brief 范围）。

---

## 0. TL;DR

把 D1 集合从 17 → 14（削减 numerical/moe 超量）+ 补 3 个新 surrogate（control_flow / communication / offload）= **D1' 17 bug**。在 D1' 上跑三方对比（TrainAudit / TrainCheck / Naïve metric monitoring），交付 paper-ready 数字。

| 操作 | bug | 工作量 |
|---|---|---|
| 削掉 | B13 (numerical), M-014 (numerical), M-NEW-5 (moe) | 0（保留文件，paper 不引用） |
| 保留 | 14 bug（见 §2.2） | 0（已有 surrogate） |
| 新增 surrogate | control_flow / communication / offload 各 1 个 | 3-5 天 |
| 三方跑 | D1' 17 bug × 3 工具 | 1 天 |
| 输出 | 新 detection-results 表 + summary CSV + 报告 | 0.5 天 |
| **总计** | | **5-7 天** |

最终交付的核心数字（预期）：

| 工具 | D1' buggy 检出 | D1' fixed FP |
|---|---|---|
| TrainAudit | ?/17 (期望 ≥16/17 ≈ 94%+) | 期望 ≤1 |
| TrainCheck | ?/17 (期望 8-10/17 ≈ 50-60%) | 期望 0 |
| Naïve Monitoring | ?/17 (期望 0/17) | 期望 0 |

**关键**：3 个新 surrogate 的设计目的是\*\*让 baseline 也能跑\*\*——必须确保 TrainCheck 能学到不变量（健康 trace 可获得）、Naïve metric 能监测（loss/grad_norm 时间序列存在）。

---

## 1. 背景与决策上下文

### 1.1 为什么需要这个实验

当前论文有两套 evaluation 集合：
- **D1 17 bug**（surrogate）：三方对比能跑，但只覆盖 9/13 类，且 numerical (4) / moe (3) 超量
- **E2E 26 bug**（真实复现）：覆盖 11/13 类，但 baseline 在真实 framework 上需要 driver 集成（未做）

用户决策（2026-05-10）：**只用 D1 集合**。要求：
1. 能复现（surrogate 满足）
2. 每类最多 2 个
3. 尽量覆盖所有类型（12/13，loss_computation 作为 §3.3 的 16.8% unobservable gap 边界保留）
4. 边界 case 保留（loss_computation 不补，明确说明）
5. **其他工作（baseline）也能跑** ← 新增诉求，决定走 surrogate 路线

### 1.2 当前 D1 17 bug 类别分布（按检测表 [Rule] 列推断）

| 类别 | 当前 D1 | 删 / 改 |
|---|---|---|
| numerical | 4 (B11, B13, M-014, O-NEW-1) | **削 2 (B13, M-014)** |
| moe | 3 (B8, M-012, M-NEW-5) | **削 1 (M-NEW-5)** |
| dtype | 2 (B3, M-024) | ✓ 保留 |
| gradient_sync | 2 (B1, B2) | ✓ 保留 |
| lr_schedule | 2 (B12, OC-NEW-3) | ✓ 保留 |
| sharding | 1 (M-020) | ✓ 保留 |
| checkpoint | 1 (O-005) | ✓ 保留 |
| data_loading | 1 (O-NEW-9) | ✓ 保留 |
| optimizer_state | 1 (OC-NEW-2) | ✓ 保留 |
| **control_flow** | **0** | **新增 1 surrogate** |
| **communication** | **0** | **新增 1 surrogate** |
| **offload** | **0** | **新增 1 surrogate** |
| loss_computation | 0 | **不补**（保留作 §3.3 16.8% unobservable 边界） |

→ 削减后 14 + 3 新 = **17 bug**，覆盖 **12/13 类**，每类 1-2 个。

---

## 2. Phase 0：明确边界（不需要 agent 做事，是输入）

### 2.1 削减的 3 个 bug

```
benchmark/eval/traincheck_surrogates/B13_buggy.py        # numerical, 与 B11 / O-NEW-1 同质
benchmark/eval/traincheck_surrogates/B13_fixed.py
benchmark/eval/traincheck_surrogates/M-014_buggy.py      # numerical, 与 B11 / O-NEW-1 同质
benchmark/eval/traincheck_surrogates/M-014_fixed.py
benchmark/eval/traincheck_surrogates/M-NEW-5_buggy.py    # moe, 与 B8 / M-012 同质
benchmark/eval/traincheck_surrogates/M-NEW-5_fixed.py
（含 _traincheck_ 变体）
```

⚠️ **不要删除文件**——它们留在 `traincheck_surrogates/` 作为 audit trail。只是论文 D1' 集合不引用。

### 2.2 保留的 14 bug

| ID | Framework | Tier | Rule | Class |
|---|---|---|---|---|
| B1 | Megatron-LM | T1 | replica-cksum-equal (cross-rank) | gradient_sync |
| B2 | Megatron-LM | T1 | grad-replica-cksum-equal (TP frozen-weight) | gradient_sync |
| B3 | DeepSpeed | T1 | comm-dtype-matches-training (BF16/FP16) | dtype |
| B8 | DeepSpeed | T1 | process-group-size-correct (EP) | moe |
| B11 | DeepSpeed | T0 | clip-grad-bounded | numerical |
| B12 | OLMo-core | T0 | initial-lr-present | lr_schedule |
| M-012 | Megatron-LM | T1 | expert-bias-fp32 (dtype demotion) | moe |
| M-020 | Megatron-LM | T1 | layer-count-strict | sharding |
| M-024 | Megatron-LM | T1 | jitter-preserves-dtype | dtype |
| O-005 | OLMo | T0 | checkpoint-preserve-rng | checkpoint |
| O-NEW-1 | OLMo | T0 | norm-output-unit-rms | numerical |
| O-NEW-9 | OLMo | T0 | data-loader-token-id-range | data_loading |
| OC-NEW-2 | OLMo-core | T0 | optim-step-counter-monotonic | optimizer_state |
| OC-NEW-3 | OLMo-core | T1 | sqrt-decay-front-loaded | lr_schedule |

---

## 3. Phase 1：写 3 个新 surrogate

### 3.1 现有 surrogate 格式参考（必读）

每个 D1 bug 有 **3 个文件** 在 [`benchmark/eval/traincheck_surrogates/`](../../benchmark/eval/traincheck_surrogates/):

```
<bug_id>_buggy.py          # ~30-40 行 PyTorch, 简化复现 buggy 行为
<bug_id>_fixed.py          # ~30-40 行 PyTorch, 简化复现 fixed 行为
_traincheck_<bug_id>_buggy.py    # ~80-90 行, 加 TrainCheck trace logging hook
_traincheck_<bug_id>_fixed.py    # 同上
```

**参考样例**：
- 看 `B1_buggy.py` (33 行) 与 `B1_fixed.py` (35 行) 理解最简形式
- 看 `M-012_buggy.py` (含 MoE expert) 理解中等复杂度
- 看 `_traincheck_B1_buggy.py` (87 行) 理解 TrainCheck 适配（多了 \texttt{torch.nn.Module} forward hook、optimizer step hook 来产生 event log）
- 看 [`benchmark/eval/traincheck_surrogates/`](../../benchmark/eval/traincheck_surrogates/) README（如果存在）

### 3.2 三个新 surrogate 规格

#### CF1: control_flow surrogate

**蓝本来源**：392 manifest 里 category=control_flow 的 reproducible bug。强候选：
- `D-001`: micro_step_id off-by-one in DeepSpeed PipelineEngine（§2.2 论文 case study 引用）
- `M-016` 或 OLMo 系 `O-NEW-30`（MoE aux-loss 缺 all-reduce 在 logging path）
- 检索：`grep "category.*control_flow.*reproduction_status.*reproduced" manifest_v2.json`

**核心 bug 语义**（参考 D-001）：
- buggy: `micro_step_id` 计数器在 reset 与 increment 路径不一致，使 grad accumulation 提前 reset
- fixed: 计数器与 grad buffer 步调一致

**Surrogate 设计**（~40 行）：
```python
# CF1_buggy.py - micro_step_id 错配
class FakeAccumulator:
    def __init__(self, accum_steps=4):
        self.micro_step_id = 0
        self.accum_steps = accum_steps
        self.buffer = torch.zeros(8)
    def accumulate(self, grad):
        self.buffer.add_(grad)
        # buggy: increment AFTER reset check
        if self.micro_step_id >= self.accum_steps:
            self.buffer.zero_()  # reset 提前一步触发
        self.micro_step_id += 1
    # ... main loop runs 16 micro-steps, prints final buffer
```

**baseline 适配**:
- TrainCheck 必须能学到"`micro_step_id` 在 reset 时 == accum_steps"这种不变量 → 输出 `micro_step_id` 与 `buffer.norm()` 到 trace
- Naïve 必须能监测某个 metric → 让 buggy 的 grad norm 在累积期内偏离 fixed 但 < 0.5%（这样 Naïve 看不到，TrainAudit 应该能看到）

#### CM1: communication surrogate

**蓝本来源**：392 manifest 里 category=communication 的 reproducible bug。强候选：
- `D-NEW-12` 或类似（wrong process group / collective op mismatch）
- 检索：`grep "category.*communication.*reproduction_status.*reproduced" manifest_v2.json`

**核心 bug 语义**：
- buggy: 在 TP group 应该 all-reduce 时用了 DP group（或反之），导致 cross-rank 数值发散
- fixed: 用对 process group

**Surrogate 设计**（~40 行）：
```python
# CM1_buggy.py - wrong process group
def fake_allreduce(tensor, group_name):
    # buggy: ignores group_name and uses "global" sum
    return tensor * 4  # always 4 ranks, but real intent is 2 (TP=2)

def main():
    # 模拟 TP=2 + DP=2 = 4 ranks
    rank_outputs = [torch.randn(8) for _ in range(4)]
    # buggy: TP allreduce 错误地除以 4 而非 2
    for i, t in enumerate(rank_outputs):
        rank_outputs[i] = fake_allreduce(t, "tp_group_buggy") / 4
    # ...
```

**baseline 适配**:
- TrainCheck: log 每个 collective op 的 `group_size` 参数，让它能学到"TP allreduce 的 size 应该 == 2"
- Naïve: 不该看到——loss 长期累积偏差，单步 grad_norm 不超阈

#### OF1: offload surrogate

**蓝本来源**：392 manifest 里 category=offload 的 reproducible bug（DeepSpeed 独占）。强候选：
- `D-NEW-44`、`D-NEW-54` 等（ZeRO + CPU offload 类）
- 检索：`grep "category.*offload" manifest_v2.json | head -10`

**核心 bug 语义**（参考 DeepSpeed ZeRO offload bug）：
- buggy: CPU offload 后 optimizer state restore 时 dtype 错（fp32 -> fp16），引入数值偏差
- fixed: 显式 cast 回 fp32

**Surrogate 设计**（~40 行）：
```python
# OF1_buggy.py - offload dtype loss
class FakeOffloadOptim:
    def step(self, params, grads):
        # 模拟 offload to CPU
        cpu_state = {p: g.cpu() for p, g in zip(params, grads)}
        # buggy: restore 时 dtype 错
        gpu_state = {p: cpu_state[p].to(p.device).half() for p in params}  # half 错
        # fixed 应该 .to(p.dtype)
        for p, g in gpu_state.items():
            p.data.add_(-1e-3 * g)
```

**baseline 适配**:
- TrainCheck: log `optimizer_state.dtype` 每步，应该不变 → 学到"dtype before/after offload 应该相同"
- Naïve: 不该看到——dtype 错只在长期 loss 缓慢漂移

### 3.3 文件命名 + 目录

```
benchmark/eval/d1_prime/                          # 新目录, 不污染 traincheck_surrogates/
├── CF1_buggy.py
├── CF1_fixed.py
├── _traincheck_CF1_buggy.py
├── _traincheck_CF1_fixed.py
├── CM1_buggy.py
├── CM1_fixed.py
├── _traincheck_CM1_buggy.py
├── _traincheck_CM1_fixed.py
├── OF1_buggy.py
├── OF1_fixed.py
├── _traincheck_OF1_buggy.py
├── _traincheck_OF1_fixed.py
├── README.md                       # 蓝本来源、bug 语义、设计选择
└── source_bug_mapping.json         # CF1 → D-001, CM1 → D-NEW-12, OF1 → D-NEW-44 这种 mapping
```

D1' 17 bug 完整集合 = `traincheck_surrogates/` 中的 14 保留 + `d1_prime/` 中的 3 新增。

### 3.4 Sanity check（写完每个 surrogate 必跑）

```bash
# 1. buggy 与 fixed 必须能跑通
python3 benchmark/eval/d1_prime/CF1_buggy.py    # 不抛异常
python3 benchmark/eval/d1_prime/CF1_fixed.py    # 不抛异常

# 2. buggy 与 fixed 输出必须不同（diff metric 显著）
# 在 surrogate 里加 print(final_metric)，buggy/fixed 应该数值不同

# 3. TrainCheck adapter 必须产生非空 trace
python3 _traincheck_CF1_fixed.py > traces/cf1_fixed.json
# JSON 至少包含 5 条 event 记录
```

---

## 4. Phase 2：D1' 17 bug × 三方工具

### 4.1 跑 TrainAudit

复用论文现有 TrainAudit pipeline（[`benchmark/eval/traincheck_surrogates/`](../../benchmark/eval/traincheck_surrogates/) 应该有 runner 脚本，没有则参考 `benchmark/eval/extension_v3/run_e1_pattern.py` 的结构）。

**输出**：`benchmark/eval/d1_prime/results/trainaudit_d1prime.json`，每 bug 一条记录：
```json
{
  "bug_id": "CF1",
  "buggy_detected": true,
  "fixed_fp": false,
  "trigger_rule": "<rule name>",
  "violations_per_total": "5/120"
}
```

### 4.2 跑 TrainCheck

参考论文 §6 已有的 TrainCheck 跑法。每个 bug：
1. 用 `_traincheck_<bug_id>_fixed.py` 学不变量（healthy trace）
2. 用学到的不变量查 `_traincheck_<bug_id>_buggy.py`
3. 记录 violation count

**输出**：`benchmark/eval/d1_prime/results/traincheck_d1prime.json`

### 4.3 跑 Naïve Monitoring

参考论文 §6 现有 Naïve 实现（loss spike + grad_norm threshold + NaN/Inf 检查）。

**输出**：`benchmark/eval/d1_prime/results/naive_d1prime.json`

### 4.4 Cross-tool agreement

3 个新 surrogate（CF1/CM1/OF1）的预期结果：
- TrainAudit：3/3 buggy detected（如果 surrogate 没设计正确则需要调）
- TrainCheck：1-2/3（取决于 surrogate 是否给到 TrainCheck 学得到的字段）
- Naïve：0/3（按设计应该是亚阈值漂移）

---

## 5. Phase 3：Aggregate + Paper-Ready 表

### 5.1 输出

`benchmark/eval/d1_prime/d1prime_summary.csv`：

| Bug ID | Class | Framework | Tier | TrainAudit Buggy | TrainAudit FP | TrainCheck Buggy | TrainCheck FP | Naive Buggy |
|---|---|---|---|---|---|---|---|---|
| B1 | gradient_sync | Megatron-LM | T1 | ✓ | ✗ | ✗ | ✗ | ✗ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| CF1 | control_flow | (surrogate) | T0/T1 | ? | ? | ? | ? | ? |
| CM1 | communication | (surrogate) | T1 | ? | ? | ? | ? | ? |
| OF1 | offload | (surrogate) | T1 | ? | ? | ? | ? | ? |

### 5.2 Aggregate metrics

`benchmark/eval/d1_prime/d1prime_aggregate.json`:
```json
{
  "n_bugs": 17,
  "trainaudit": {"buggy_detection": "?/17", "fixed_fp": "?/17"},
  "traincheck":  {"buggy_detection": "?/17", "fixed_fp": "?/17"},
  "naive":       {"buggy_detection": "?/17", "fixed_fp": "?/17"},
  "class_coverage": "12/13 (loss_computation 保留作边界)"
}
```

### 5.3 实验报告

`benchmark/eval/d1_prime/d1prime_report.md`，含：
- TL;DR 数字
- Per-bug 检测结果（同 §5.1 表）
- 3 新 surrogate 的设计说明（蓝本来源 / bug 语义 / baseline 适配）
- 失败 case 分析（如果 TrainAudit 不是 17/17，单独说明）

---

## 6. 论文集成（实验完成后我会做，不在本 brief 范围）

预计要改：
- [`main_cn.tex:182`](../../main_cn.tex#L182) abstract：删 "26 bug 端到端复现"，统一为 "17 bug stratified detection"
- [`main_cn.tex:243`](../../main_cn.tex#L243) intro contribution 3：同上
- [`main_cn.tex:619`](../../main_cn.tex#L619) §6.1 workload 段：合并 D1 + E2E 为单一 "D1' 17 bug stratified set"
- [`main_cn.tex:660-685`](../../main_cn.tex#L660) detection-results 表：14 → 17 行（+ CF1/CM1/OF1）
- [`main_cn.tex:703`](../../main_cn.tex#L703) 23/26 检出语句：替换为 D1' 17/17 + boundary 边界（B7/B14/B15 概念删除或独立处理）

---

## 7. 时间预算

| Phase | 工作 | 工作量 |
|---|---|---|
| Phase 1 | 蓝本检索 + 3 surrogate 编写 + sanity check | 3-4 天 |
| Phase 1.5 | TrainCheck-adapted 版本编写 | 0.5-1 天 |
| Phase 2 | 三方运行 D1' 17 bug | 0.5-1 天 |
| Phase 3 | Aggregate + 报告 | 0.5 天 |
| **合计** | | **5-7 天** |

---

## 8. 不要做的事

- ❌ **不要删除** `traincheck_surrogates/` 中的 B13/M-014/M-NEW-5 文件——保留作 audit trail
- ❌ **不要修改** 现有 14 bug 的 surrogate 代码——它们已通过 paper §6 评估
- ❌ **不要重跑** 现有 14 bug 的三方对比——直接复用 paper §6 表 [main_cn.tex:660-685](../../main_cn.tex#L660) 的现有结果（除非发现 stale）
- ❌ **不要补 loss_computation 的 surrogate**——它故意保留作 §3.3 16.8% unobservable 边界
- ❌ **不要在真实 framework 上跑 TrainCheck/Naïve**——这是为什么要 surrogate 的原因
- ❌ **不要改 manifest_v2.json**——D1' 是论文 evaluation set，与 392 调研池不冲突
- ❌ **不要追求 17/17 检出率**——CF1/CM1/OF1 如果有一个不被 TrainAudit 检出，诚实报告（这正是 boundary case 的展现机会）

---

## 9. 一行复算入口

```bash
# Phase 1 sanity
for bug in CF1 CM1 OF1; do
    python3 benchmark/eval/d1_prime/${bug}_buggy.py
    python3 benchmark/eval/d1_prime/${bug}_fixed.py
done

# Phase 2 three-way
python3 benchmark/eval/d1_prime/run_d1prime_threeway.py  # 待写

# Phase 3 aggregate
python3 benchmark/eval/d1_prime/aggregate_d1prime.py     # 待写
```

---

## 10. 决策矩阵（实验完成后看哪一档）

| TrainAudit D1' 检出 | 论文集成方式 |
|---|---|
| 17/17 (100%) | abstract 直接说 "17/17 (100%)"，与原 D1 17/17 对照保住 |
| 16/17 | 1 个 boundary FN，appendix 解释为什么（很可能是 OF1 类似 sub-percent drift） |
| 15/17 或更低 | 重审 surrogate 设计——某个新 surrogate 的 bug 语义可能没设计对，调 prompt 或重写 |

| TrainCheck D1' 检出 | 论文写法 |
|---|---|
| 8-10/17 (与原 10/17 量级一致) | 直接报，"TrainCheck 在更广覆盖类下仍约 50-60%" |
| <8/17 | 解释新 surrogate 上 TrainCheck 失效的原因（缺少 trace schema 字段） |
| 11+/17 | 不太可能，但若发生说明 TrainCheck 在某些新类上反而更强，appendix 单独讨论 |

---

## 11. 边界 FN（B7/B14/B15）的命运

当前论文 §6 line 703 提"3 个 boundary FN（B7/B14/B15）"——这些是 E2E 26 bug 集合的 ID，**不在 D1 17 bug 表里**。本实验整合后 E2E 26 bug 概念删除。处理选项（**用户决定，agent 不需做**）：

- 选项 A：完全删除 B7/B14/B15 references，论文不再提 "boundary FN"
- 选项 B：把 B7/B14/B15 + loss_computation 合并为 "Boundary Case Set"，appendix 单独展示
- 选项 C：用 §6.1 line 621 的 unobservable boundary 描述统一覆盖（推荐）

→ 选 C 时，B7/B14/B15 的具体 ID 论文删除，统一指向 "16.8% runtime-unobservable bugs (sub-percent drift / spatial dilution / dataset shuffle below noise floor)"，与 §3.3 staircase 的 16.8% 数字闭环。
