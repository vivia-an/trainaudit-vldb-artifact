# 29. Staircase 392 全集实验 Brief（给实验 agent）

> 目标：在 392 全集上算 invariant 三层 staircase（schema_only / schema_topo / schema_topo_precond / none），替换论文 §3.3 [main_cn.tex:381](../../main_cn.tex#L381) 当前的 128 子集数字（20% / 51% / 74%）。
> 上游：[27_master_summary.md](27_master_summary.md) §6（v1→v2 prompt 演化）、[28_392_extension_brief.md](28_392_extension_brief.md)（参考体例）、[28_extension_summary_report.md](../../benchmark/eval/extension_v3/extension_summary_report.md)（E1/E2/E3 报告）。
> 完成日期：待定。

---

## 0. TL;DR

实验交付**两组数字**：392 全集上 4 个 staircase layer 的分布 + 三个论文锚点（schema-only / schema+topo / schema+topo+precond）的累积覆盖率。

| 数字 | 论文 [:381](../../main_cn.tex#L381) 当前 (128 子集) | 392 实测目标 | 处理 |
|---|---|---|---|
| `pi_schema` 充分（cumulative） | **20%** | ?% | 直接换 |
| `pi_schema + pi_topo` 充分 | **51%** | ?% | 直接换 |
| `pi_schema + pi_topo + pi_precond` 充分 | **74%** | ?% | 直接换 |
| `none` 不可在三层内检出 | **26%** | ?% | 直接换 |

⚠️ **风险预警**：30-bug v2 pilot 在 392 池上的初步分布是 **schema_only=10% / schema_topo=0% / schema_topo_precond=73% / none=17%**（[27_master_summary.md](27_master_summary.md) §6.3）。如果全集仍是 `schema_topo ≈ 0%`，论文 §3.3 的核心论点（"加入 $\pi_{\text{topo}}$ 把覆盖率从 20% 提升到 51%，是 +31pp 的关键证据"）会被**反转**。实验 agent 必须诚实报告塌方分布，不要为了配合论文而调 prompt。

---

## 1. 背景与决策上下文

### 1.1 这个实验是什么

论文 §3.3 三层验证的核心数字 **20% → 51% → 74%** 来自 128 hand-curated bugs 上的 v1 prompt 标注（详见 [27_master_summary.md](27_master_summary.md) §6）。E1/E2/E3 三个扩展实验**都不测这个**——它们分别测 8 模式覆盖、trace tier、hook coverage，与 staircase 是不同分析。

我们需要单独跑 staircase——它依赖 `minimum_sufficient_layer` 字段（v2 schema 的 12 字段之一）的全集标注。

### 1.2 v1 → v2 已修复的 bug（**不要再回到 v1**）

- v1 prompt 把 `pi_schema_applicable` / `pi_topo_applicable` / `pi_precond_applicable` 三个 bool 定义为"可同时为 true"，与论文 staircase 的"互斥充分层"语义不一致 → staircase 不可算
- v2 用单字段 `minimum_sufficient_layer ∈ {schema_only, schema_topo, schema_topo_precond, none}` 替代，对应论文三个数字
- v2 prompt 完整定义在 [`benchmark/eval/annotate_prompt.md`](../../benchmark/eval/annotate_prompt.md)

### 1.3 30-bug pilot 已有数据（**复用，不重跑**）

[`benchmark/eval/annotations/`](../../benchmark/eval/annotations/) 下已有 30 个 v2 标注（pilot 产出），分布：

| Layer | Pilot Count | Pilot % | 128 池 ref |
|---|---|---|---|
| `schema_only` | 3 | 10% | 20% |
| `schema_topo` | 0 | **0%** ⚠️ | 31% (cumulative gap) |
| `schema_topo_precond` | 22 | 73% | 23% (cumulative gap) |
| `none` | 5 | 17% | 26% |

→ 全集（剩余 362 个）跑完后，要看：
1. `schema_topo` 比例是否仍接近 0%（如是 → 论文论点要重写）
2. 三个累积数字（schema cumulative / +topo cumulative / +precond cumulative）的最终值

---

## 2. 输入数据

### 2.1 主输入

| 文件 | 路径 | 内容 |
|---|---|---|
| 392 manifest | [`benchmark/eval/manifest_v2.json`](../../benchmark/eval/manifest_v2.json) | per-bug id / framework / category 等 |
| 128 池原始字段 | [`exp/data/*_silent_errors.json`](../../exp/data/) | 含 invariant / detection_signal 等深字段 |
| 295 池 per-bug | [`benchmark/bugs/<id>/config.json`](../../benchmark/bugs/) | root_cause / invariant / detection_method 等 |
| **v2 prompt** | [`benchmark/eval/annotate_prompt.md`](../../benchmark/eval/annotate_prompt.md) | ⭐ stable prefix，**直接复用** |
| **30 已标注** | [`benchmark/eval/annotations/`](../../benchmark/eval/annotations/) | ⭐ 复用，不重跑 |

### 2.2 待标注的 362 个 bug

```bash
# 计算待标注列表
python3 -c "
import json
from pathlib import Path
manifest = json.loads(Path('benchmark/eval/manifest_v2.json').read_text())
done = {p.stem for p in Path('benchmark/eval/annotations/').glob('*.json')}
todo = [b['id'] for b in manifest['bugs'] if b['id'] not in done]
print(f'TODO: {len(todo)}')  # 期望 ~362
Path('benchmark/eval/staircase_v3/todo.json').parent.mkdir(parents=True, exist_ok=True)
Path('benchmark/eval/staircase_v3/todo.json').write_text(json.dumps(todo, indent=2))
"
```

---

## 3. 实验规格

### 3.1 任务定义

对每个待标注 bug，调用 v2 prompt 输出 12 字段 JSON。**关注的核心字段是 `minimum_sufficient_layer`**——其他 11 字段与 E1/E2/E3 已有结果的对照可作 sanity check。

### 3.2 输出 schema

每 bug 一个 JSON：[`benchmark/eval/staircase_v3/annotations/<bug_id>.json`](../../benchmark/eval/staircase_v3/)

**不要写到 `benchmark/eval/annotations/`**——保持 30-bug pilot 目录纯净；新结果在独立目录。最后 aggregate 时把两边合并。

### 3.3 Aggregate 输出

`benchmark/eval/staircase_v3/staircase_392.json`:

```json
{
  "metadata": {
    "n_bugs": 392,
    "n_pilot_reused": 30,
    "n_new_annotated": 362,
    "method": "v2 prompt minimum_sufficient_layer field on 392 unified pool",
    "llm_model": "Claude Opus 4.7 1M (Temperature=0)",
    "date": "2026-05-DD"
  },
  "per_bug": {
    "<bug_id>": {
      "minimum_sufficient_layer": "schema_only|schema_topo|schema_topo_precond|none",
      "category": "<13 类>",
      "framework": "...",
      "rationale": "<30-300 chars>"
    },
    ...
  },
  "aggregate": {
    "by_layer": {
      "schema_only": {"count": ?, "pct": ?},
      "schema_topo": {"count": ?, "pct": ?},
      "schema_topo_precond": {"count": ?, "pct": ?},
      "none": {"count": ?, "pct": ?}
    },
    "cumulative_for_paper": {
      "pi_schema_sufficient":          {"pct": ?, "comment": "= schema_only / 392"},
      "pi_schema_topo_sufficient":     {"pct": ?, "comment": "= (schema_only + schema_topo) / 392"},
      "pi_three_layer_sufficient":     {"pct": ?, "comment": "= (schema_only + schema_topo + schema_topo_precond) / 392"},
      "unobservable":                  {"pct": ?, "comment": "= none / 392"}
    },
    "by_framework": {
      "megatron-lm": {"schema_only": ?, "schema_topo": ?, ...},
      ...
    },
    "by_source_pool": {
      "128_only": {"schema_only": ?, ...},
      "both": {...},
      "295_only": {...}
    }
  }
}
```

### 3.4 Paper-ready CSV

`benchmark/eval/staircase_v3/staircase_summary.csv`：

| Layer | 128 池 baseline | 392 实测 | Δpp | 状态 |
|---|---|---|---|---|
| `pi_schema` only sufficient | 20% | ?% | ? | ?? |
| `+pi_topo` sufficient | +31% (cum 51%) | ?% | ? | ?? |
| `+pi_precond` sufficient | +23% (cum 74%) | ?% | ? | ?? |
| unobservable | 26% | ?% | ? | ?? |

---

## 4. 通用方法学

### 4.1 复用 v2 prompt 与已有标注

- **Prompt**：直接用 [`benchmark/eval/annotate_prompt.md`](../../benchmark/eval/annotate_prompt.md)，不要改 SYSTEM / 表 A-G / 反例段。如必须微调，记录 diff 在 `staircase_v3/prompt_diff.md`。
- **30-bug pilot**：照原样复用，不重跑。aggregate 时把 30 个文件 `cat` 进 staircase_392.json 即可。

### 4.2 模型与并发

- Claude Opus 4.7 1M，Temperature=0，prompt caching = stable prefix on prompt 文件。
- 362 calls × 1 prompt ≈ 8-10M token，估计 \$15-25。
- 可分 4-5 个 batch 并发 subagent（每 batch ~80-90 bug），与 28-brief 同样的方式。

### 4.3 Sanity check

跑全量前先做：

1. **5-bug spot check 在 30-bug pilot 上**：用现在的 prompt 重跑 30 个 pilot 中的 5 个，对比已有 annotation。一致率应 ≥80%。否则 prompt 漂移，要排查。
2. **128 子集 sanity（可选）**：随机抽 20 个有完整 v1 annotation 且属于 128 子集的 bug，跑 v2 prompt。聚合 staircase 数字应再现论文的 20/51/74%（误差 ±5pp）。如果不能再现，说明 v2 prompt 与论文 v1 数字定义有偏差，要先解决再跑全量。

### 4.4 失败处理

- **pilot 5-bug 一致率 <80%**：报告差异，调 prompt 或汇报方法学问题，不要硬跑全量
- **sanity 128 子集与 20/51/74% 偏差 >5pp**：诚实报告，appendix 加一段"v2 schema 与原 v1 staircase 数字的口径差异"
- **全集 `schema_topo ≈ 0%`**（pilot 已显示）：报告事实，**不调 prompt**——这可能是真实分布（NEW 池里所有跨 rank 比对的 bug 同时也需要 module attr 守护）。让 paper 决定怎么写。

---

## 5. 输出物清单

```
benchmark/eval/staircase_v3/
├── todo.json                          # 362 bug ID list
├── prompt_diff.md                     # 如果调了 prompt（理想是空文件）
├── batch_input/
│   ├── batch1_input.json              # 90 bug 的 stripped input
│   ├── batch2_input.json
│   ├── batch3_input.json
│   └── batch4_input.json
├── annotations/
│   ├── <bug_id>.json × 362            # 新标注
├── pilot_sanity_recheck.json          # §4.3 step 1 的 5-bug spot check 结果
├── 128_subset_sanity.json             # §4.3 step 2 的 20-bug sanity（可选）
├── staircase_392.json                 # ⭐ aggregate（含 pilot 30 + new 362）
├── staircase_summary.csv              # ⭐ paper-ready
└── staircase_summary_report.md        # ⭐ 实验总结报告
```

---

## 6. 论文集成（实验完成后我会改）

### 6.1 §3.3 [main_cn.tex:381](../../main_cn.tex#L381) 的处理

当前：
```
在 128-bug 深字段标注子集上（详见附录~\ref{app:methodology}）：仅 \pi_{schema} 达到 20%；
加入 \pi_{topo} 达到 51%；三层全部到位达到 74%。剩余 26% ...
```

按报告交付的数字，规则：

| 全集 schema_topo 占比 | 处理方式 |
|---|---|
| ≥ 5%（与 128 池 31pp 跳变在同量级） | 全替换为 392 数字，§3.3 论证不变 |
| 1--5%（明显衰减但非 0） | 全替换为 392 数字，§3.3 文字加一句解释 |
| **≈ 0%（pilot 倾向）** | **保留 128 数字 + appendix 加专节解释为什么扩展不可比**：v2 schema 的"互斥充分层"在 NEW 池里偏向最深层 |
| 不能 sanity 出 v1 的 20/51/74%（口径差异） | appendix 详述 v1 vs v2 schema 差异，主文用 v2 数字（392），脚注引论文原 128 数字 |

### 6.2 同时要更新的位置

- [main_cn.tex:381](../../main_cn.tex#L381) 主文 §3.3 三个数字
- 26% gap 的提法也要校准（→ 392 池 unobservable %）
- Abstract / intro / conclusion 不直接引这些数字，**不需要改**
- 如果 §3.3 论点反转，需重写 §3.3 段落（约半小时工作）

---

## 7. 时间预算

| 阶段 | 工作量 |
|---|---|
| Phase 0：环境核查 + todo list 生成 | 0.1 天 |
| §4.3 step 1：5-bug pilot recheck | 0.2 天 |
| §4.3 step 2：20-bug 128-subset sanity（可选） | 0.3 天 |
| 362 bug 全量标注（4 batches 并发） | 0.5 天 |
| Aggregate + summary report | 0.4 天 |
| **合计** | **1.0--1.5 天**（含可选 sanity） |

Token 预算 ~10M，成本 \$15-25。

---

## 8. 不要做的事

- ❌ **不要回到 v1 的 3 bool schema** —— 互斥语义已修复，回退会让 staircase 不可算
- ❌ **不要为了配合论文 20/51/74% 调 prompt** —— pilot 显示塌方就是塌方，必须诚实
- ❌ **不要重跑 30 个 pilot annotation** —— 复用现有
- ❌ **不要改 `benchmark/eval/annotate_prompt.md`** —— 它是其他下游任务的 stable prefix
- ❌ **不要改 `manifest_v2.json`** —— 标注写到独立目录
- ❌ **不要扩展到 v2 之外的 12 字段分析**（pattern / hook / tier 等）—— 这些已由 E1/E2/E3 完成

---

## 9. 一行复算入口

```bash
# Step 1: 准备 todo + batch input
python3 benchmark/eval/staircase_v3/build_todo.py     # 待写

# Step 2: pilot recheck (5 bugs)
python3 benchmark/eval/staircase_v3/pilot_recheck.py  # 待写

# Step 3: 全量标注（subagent × 4 batches）
# (subagent 调用，由 agent 自行编排)

# Step 4: aggregate + summary
python3 benchmark/eval/staircase_v3/aggregate.py      # 待写
```

预期最终产出：`staircase_summary.csv` 含 4 个数字（schema_only / schema_topo / schema_topo_precond / none 占比）+ 3 个论文锚点（cumulative 数字）+ 与 128 池对照 Δpp。

---

## 10. 决策矩阵（实验完成后我看哪一档）

| schema_topo 占比 | none 占比 | 论文处理 |
|---|---|---|
| 25-35% | 20-30% | ✅ 直接替换 §3.3 三个数字（论证强度等价） |
| 5-25% | 任意 | 🟡 替换数字 + §3.3 加一句"中间层贡献减弱" |
| < 5% | < 30% | ⚠️ 保留 128 数字，appendix 加专节"v2 在更大池上的塌方分布" |
| 任意 | > 30% | ⚠️ unobservable 显著上升，§3.3 需重写"runtime 边界" |
