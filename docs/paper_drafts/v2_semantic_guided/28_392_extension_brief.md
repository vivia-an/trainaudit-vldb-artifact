# 28. 392 全集扩展实验 Brief（给实验 agent）

> 目标：把目前仅在 **128 hand-curated subset** 上算过的三项分析扩展到 **392 unified pool**，给论文 §3.3 / §4 提供 392 池版本的数字。
> 上游：[27_master_summary.md](27_master_summary.md)、[taxonomy_methodology.md](taxonomy_methodology.md)。

---

## 0. TL;DR — 这个实验要交付什么

三组数字，每组都给 **128 子集 ↔ 392 全集** 的对比表，外加 per-bug label JSON：

| 实验 | 128 池现有数字（论文 §） | 392 池目标数字 | 输出文件 |
|---|---|---|---|
| **E1: 8 pattern coverage** | 84/128 = 66%（§4.1, [main_cn.tex:423](../../main_cn.tex#L423)；附录表 [:949](../../main_cn.tex#L949)） | `pattern_hit_count / 392` per pattern + 总覆盖 | `pattern_coverage_392.json` |
| **E2: Trace schema tier coverage** | Tier 0/1/2-3/5 → 30/51/61/74%（§4.x, [main_cn.tex:535](../../main_cn.tex#L535)） | 各 tier 累积覆盖 / 392 | `trace_tier_392.json` |
| **E3: After-forward hook coverage** | 48/128 = 37.5%（§4.x, [main_cn.tex:557](../../main_cn.tex#L557)） | 5 个 lifecycle hook 各自可观测 bug 数 / 392 | `hook_coverage_392.json` |

每个实验都必须输出：
1. **per-bug label**（JSON）
2. **aggregate metrics**（CSV，paper-ready）
3. **vs 128 子集对比**（差异 >5pp 必须给出原因分析）

---

## 1. 背景与决策上下文

### 1.1 为什么要做

论文当前 §3.3 / §4 多处用的是 **128 子集** 数字（staircase 20/51/74%、8 pattern 66%、trace tier 30/51/61/74%、after-forward hook 48/128）。我们已经把 §3.1 taxonomy 表换成 392 池数字（[main_cn.tex:319-343](../../main_cn.tex#L319)），现在要决定上面这些 128 数字要不要也换。

**用户决策**（2026-05-10）：
- **Staircase 不扩**——已有 30-bug v2 pilot 显示塌方（10/0/73/17 vs 128 池 20/31/23/26），文档 §6.3 已确认。
- **E1 / E2 / E3 直接跑 392 全集**，不做 30-bug spot check。如果与 128 子集偏差 >5pp，论文用双数字 + honest reporting；偏差 <5pp 直接换。

### 1.2 与已做工作的关系

- **不重做** taxonomy 13 类归类（已完成，见 `manifest_v2.json` 与 `category_alias_map.json`）。
- **不重做** silent-vs-loud 评级（已完成，见 `silent_evidence_392.json`）。
- **不重做** IRR（已完成，见 `irr_50_results.json`）。
- 本实验 **只补三项 fine-grained 字段标注**，复用现有 392 manifest。

---

## 2. 输入数据

### 2.1 主输入

| 文件 | 路径 | 内容 |
|---|---|---|
| 392 unified manifest | `benchmark/eval/manifest_v2.json` | 每 bug 的 id / framework / source pool / category（13 类）/ root_cause / invariant / detection_method 等 |
| Per-bug config | `benchmark/bugs/<id>/config.json` | 详细字段（295 池来源）：root_cause、trigger_conditions、detect.py、reproduce.sh 等 |
| 128 池字段参考 | 见 `manifest_v2.json` 中带 `source_pool: 128` 或 `both` 的 entry | 含深字段（invariant / detection_signal / required_trace_fields / check_stage / parallel_dimension / invariant_type） |

### 2.2 128 池基线数字（用作对照与 sanity check）

| Pattern ID | Pattern 名（参考 [main_cn.tex:949](../../main_cn.tex#L949) 表） | 128 池命中数 |
|---|---|---|
| P1 | Dtype Preservation | 12 |
| P2 | … | （从论文表读完整 12 个） |
| ... | | |
| **合计（去重）** | | **84/128 = 66%** |

**Trace tier 128 池基线**（从 [main_cn.tex:535](../../main_cn.tex#L535) 与 [figures/tier_coverage.pdf](../../figures/tier_coverage.pdf)）：

| Tier | 字段集合 | 128 池累积覆盖 |
|---|---|---|
| 0 | param checksum, name, rank, step, stage, dtype, shape | 30% |
| 1 | + 梯度统计 | (推) ~40% |
| 2 | + loss 值 | 51% |
| 3 | + 控制变量 | (推) ~58% |
| 4 | + optimizer-state checksum | 61% |
| 5--6 | full 27 字段 schema | 74% |

> ⚠️ Tier 1/3 的具体数字 master_summary 没明说，需要从 `recompute_v2/` 或原始 128 池标注里抠出来作为基线。

**Hook coverage 128 池基线**（论文 [main_cn.tex:557](../../main_cn.tex#L557)）：

| Hook | 128 池可观测 bug 数 |
|---|---|
| before-forward | ? |
| **after-forward** | **48/128 = 37.5%**（论文重点） |
| main-grad-in-backward | ? |
| after-backward | ? |
| before-optimizer | ? |

> ⚠️ 其他 4 个 hook 的具体数字也要在实验中复算 128 子集数字以建立对照基线。

---

## 3. 三项实验的具体规格

### 3.1 E1: 8 Pattern Coverage

**问题定义**：给定一个 bug，它能被 8 个 pattern（P1--P8，参考论文附录表 [:949](../../main_cn.tex#L949)）中哪些 pattern 实例化检出？

**Pattern 列表**（从 [main_cn.tex:949-964](../../main_cn.tex#L949-L964) 读取完整定义；至少包含名称、Type A/B/C、规则形式 r 与 scope σ）。每个 pattern 是一个 `<r, σ, m, ρ>` 4 元组。

**输出 schema** (`pattern_coverage_392.json`)：

```json
{
  "metadata": {
    "n_bugs": 392,
    "patterns": ["P1", "P2", ..., "P8"],
    "method": "LLM-assisted classification with rule-based verification",
    "llm_model": "<model used>",
    "date": "2026-05-DD"
  },
  "per_bug": {
    "M-001": {
      "category": "moe",
      "patterns_hit": ["P3"],
      "rationale": "router weight 跨 TP rank 不一致 → P3 Cross-Rank Replication"
    },
    ...
  },
  "aggregate": {
    "P1": {"hit_count": ?, "framework_breakdown": {...}},
    ...,
    "any_pattern": {"hit_count": ?, "pct": ?},
    "no_pattern": {"hit_count": ?, "pct": ?}
  }
}
```

**Method**：
1. 对每个 bug，把 `root_cause` + `invariant` + `detection_method` 字段喂给 LLM，prompt 输出 "this bug 触发哪些 pattern"
2. 同一 bug 可命中多个 pattern（OR），可能 0 个（no pattern hit → 该 bug 不在 8 模式覆盖内）
3. 验证：**全部 32 个 cross-pool 重叠 bug 用作 sanity check**——它们在 128 池有人工 pattern 标注，LLM 标注必须与之一致率 ≥75%；否则需要 prompt 调整后重跑。

**对照与 honest reporting**：
- 输出表 `pattern_coverage_summary.csv`：列 = (P1..P8, total)，行 = (128 子集, 392 全集, Δpp)
- 若 392 总覆盖与 128 子集差 >5pp，appendix 写 "extension reveals X% on full pool vs Y% on hand-curated subset, difference attributable to ..."

---

### 3.2 E2: Trace Schema Tier Coverage

**问题定义**：给定一个 bug，要观测到它需要 trace schema 至少包含到哪个 tier 的字段？

**Tier 定义**（论文 [main_cn.tex:535](../../main_cn.tex#L535) 与 figures/tier_coverage.pdf）：
- Tier 0: param checksum, name, rank, step, stage, dtype, shape
- Tier 1: + 梯度统计（grad norm, grad mean, grad std）
- Tier 2: + loss 值
- Tier 3: + 控制变量（learning rate, optimizer config flags）
- Tier 4: + optimizer-state checksum
- Tier 5--6: 完整 27 字段 schema（含 module attribute、function-call event、checkpoint metadata）

**输出 schema** (`trace_tier_392.json`)：

```json
{
  "metadata": {...},
  "per_bug": {
    "M-001": {
      "min_tier_required": 0,
      "fields_required": ["param_checksum_per_rank", "step"],
      "rationale": "需要 cross-rank checksum 比较 → tier-0 字段足够"
    },
    ...
  },
  "aggregate": {
    "tier_0": {"cumulative_hit": ?, "pct": ?},
    "tier_1": {"cumulative_hit": ?, "pct": ?},
    ...,
    "tier_5_6": {"cumulative_hit": ?, "pct": ?},
    "unobservable": {"hit_count": ?, "pct": ?}
  }
}
```

**Method**：
1. LLM 读 bug 的 `root_cause` + `invariant` + `detect.py`（如果有），输出"检出该 bug 所需的最少字段集合"
2. 把字段集合映射到 tier（取最大 tier 即为 min_tier_required）
3. 累积覆盖：tier-N 累积 = sum(min_tier_required ≤ N)
4. 部分 bug 可能 unobservable（任何 tier 都不够）→ 计入 26% 极限的对照

**对照基线**：
- 128 子集累积应再现 30/-/51/-/61/74%，作为 LLM 标注质量的 sanity check
- 32 个 cross-pool 重叠 bug 一致性 ≥80%

---

### 3.3 E3: Hook Coverage

**问题定义**：每个 bug 在 5 个 lifecycle hook 中的哪一个（或几个）可观测？

**5 个 hook**（论文 [main_cn.tex:557](../../main_cn.tex#L557)）：
1. `before-forward`
2. `after-forward` ← 论文重点（128 池 48/128）
3. `main-grad-in-backward`
4. `after-backward`
5. `before-optimizer`

**输出 schema** (`hook_coverage_392.json`)：

```json
{
  "metadata": {...},
  "per_bug": {
    "M-001": {
      "hooks_observable": ["after-forward", "after-backward"],
      "earliest_observable": "after-forward",
      "rationale": "router weight 偏离在 forward 计算时已显现"
    },
    ...
  },
  "aggregate": {
    "before-forward": {"observable_count": ?, "pct": ?},
    "after-forward":  {"observable_count": ?, "pct": ?},
    "main-grad-in-backward": {...},
    "after-backward": {...},
    "before-optimizer": {...},
    "earliest_distribution": {
      "before-forward": ?, "after-forward": ?, ...
    }
  }
}
```

**Method**：
1. LLM 读 bug 的 `root_cause` + `invariant` + `check_stage`（128 池有），输出"在哪个 hook 第一次可观测到"
2. 注意区分 "earliest observable hook"（最早暴露的位置）vs "all observable hooks"（任何能检测到的位置）—— 论文要的是 earliest（48/128 是 after-forward 的 earliest 计数）
3. 部分 bug 可能在多个 hook 同时可观测（例如 checkpoint bug 在 save 和 load 都可观测）

**对照基线**：
- 128 子集 after-forward earliest 应再现 48/128 = 37.5%
- 32 个 cross-pool 重叠 bug 一致性 ≥80%

---

## 4. 通用方法学要求

### 4.1 LLM prompt 设计

- **三个实验各写一份独立 prompt**，存放 `benchmark/eval/extension_v3/prompts/{e1_pattern,e2_tier,e3_hook}.md`
- prompt 必须输出 **structured JSON**（不要自由文本）
- prompt 必须包含：
  - 任务定义（用本 brief 的 §3 文字）
  - 输出 schema 示例
  - 至少 3 个 few-shot 例子（从 128 子集抽，含正反例）
  - "如果不确定，输出 `uncertain` 并解释" 的 escape hatch

### 4.2 LLM 模型选择

建议 GPT-4.1 或 Claude Opus 4.x。**不要** 用 GPT-4o-mini 等小模型——precision 不够。每 bug 一次调用，全集约 392 calls × 3 实验 = 1176 calls，估计 token 用量 30--40M，成本 \$50--80。

### 4.3 验证步骤（每实验）

1. **128 子集 sanity check**：在 128 子集上跑完整流程，对比已知数字（66% / 30-51-61-74% / 48/128），偏差 >5pp 调 prompt 后重跑
2. **Cross-pool 32 bug 一致性**：与 128 池原标注对比，一致率 ≥75%（pattern）/ 80%（tier、hook）
3. **手工抽样**：每实验从 295-only 子集随机抽 10 bug，作者手工核验 LLM label，记录 disagreement
4. **Confidence interval**：用 bootstrap 给出 95% CI

### 4.4 失败处理

- 若 sanity check 偏差 >10pp 且 prompt 调整无效 → **不强行扩展**，论文继续用 128 子集 + appendix 写"extension attempted but ..." honest reporting
- 若个别 bug 字段太稀疏（L4 中部分 case）LLM 无法判断 → 标 `uncertain`，aggregate 时单独列一行 "unable to classify"

---

## 5. 输出物清单

```
benchmark/eval/extension_v3/
├── prompts/
│   ├── e1_pattern.md
│   ├── e2_tier.md
│   └── e3_hook.md
├── pattern_coverage_392.json          # E1 per-bug + aggregate
├── trace_tier_392.json                # E2
├── hook_coverage_392.json             # E3
├── pattern_coverage_summary.csv       # paper-ready table（128 vs 392 vs Δ）
├── trace_tier_summary.csv             # 同上
├── hook_coverage_summary.csv          # 同上
├── extension_summary_report.md        # 总结：三项数字 + sanity check 通过情况 + 重大偏差
└── disagreement_set/                  # 32 cross-pool 一致性 audit
    ├── e1_disagreements.json
    ├── e2_disagreements.json
    └── e3_disagreements.json
```

---

## 6. 论文集成 hook（实验完成后我会改）

数字交付后，我会按以下规则更新 `main_cn.tex`：

| 偏差 | 处理 |
|---|---|
| ±5pp 内 | §4 直接换 392 数字，加一句"This rate holds across the 128-bug hand-curated subset and the 392-bug full pool, see App.~\ref{app:methodology-extension}" |
| 5--10pp | 双报数字："on the 128-bug hand-curated subset where deep field annotations are available, X%; on the full 392-bug pool, Y%" |
| >10pp | 保留 128 数字，appendix 新增 §A.7 "Extension to 392-bug Pool" 报告差异 + 原因分析 |

新增的 appendix subsection 模板（数据出来后再写）：

```
\subsection{扩展到 392 全集的字段分析}
\label{app:method-extension}

为验证 §\ref{subsec:invariant_miner} 的 8 模式覆盖率与 §\ref{subsec:data-collector}
的 trace schema tier coverage 是否在更大 pool 上稳定, 我们将 128-bug
hand-curated subset 上的字段标注扩展到 392 全集 ...
```

---

## 7. 复算入口（实验 agent 跑完后填写）

```bash
# 端到端复算（实验 agent 完成后填实际命令）
python3 benchmark/eval/extension_v3/run_e1_pattern.py
python3 benchmark/eval/extension_v3/run_e2_tier.py
python3 benchmark/eval/extension_v3/run_e3_hook.py
python3 benchmark/eval/extension_v3/aggregate_summary.py
```

预期产出：3 张 summary CSV + 一份 `extension_summary_report.md` 给我做论文集成。

---

## 8. 时间预算

| 阶段 | 工作量 |
|---|---|
| Prompt 设计 + few-shot 例子整理（三个实验） | 0.5 天 |
| 128 子集 sanity check + prompt 迭代（三个实验） | 1 天 |
| 392 全集 LLM 标注（三个实验，1176 calls） | 0.5 天（API 并行） |
| Cross-pool 32 bug 一致性 + 手工抽样 audit | 1 天 |
| Aggregate + summary report 撰写 | 0.5 天 |
| **合计** | **3.5 天** |

---

## 9. 不要做的事

- ❌ 不要重新跑 silent-vs-loud 评级（已完成）
- ❌ 不要重新跑 13 类 IRR（已完成）
- ❌ 不要扩 staircase 数字（已确认塌方，论文用 128 子集 + 显式说明）
- ❌ 不要新增 fine-grained 12 字段标注（α 方案已搁置，[27_master_summary.md](27_master_summary.md) §6 文档为证）
- ❌ 不要改 `manifest_v2.json` 主文件 —— 三项分析的 label 写到 `extension_v3/` 独立目录
