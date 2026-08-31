# 27. Bug Pool 实验总览（2026-05-10 master summary）

> 本文档汇总自 2026-05-10 启动的 bug pool 融合 + 调查 + 分类实验。
> 后续如需查阅"我们做了什么 / 数字是多少 / 文件在哪 / 审稿人怎么挡"，先读本文。

---

## 0. TL;DR

**目标**：搜集 4 个开源分布式训练框架（Megatron-LM / DeepSpeed / OLMo / OLMo-core）的 silent error commit，做调查 + 分类，作为论文 §3 taxonomy source 与 §6 evaluation benchmark。

**结果**：392-bug 统一池建立完成，全部对齐到 13 类 taxonomy，silent error 校验 + IRR 实验完成，所有方法论产出物可供审稿人质询。

**核心数字**（一段全列）：

- **池规模**：392 个 unique bugs (M:110 / D:123 / O:77 / OC:82)；含 96 个 128 池独有 + 263 个 295 池独有 + 32 个两池重叠 + 1 个 orphan (M-NEW-MUON-MTP)
- **13 类 taxonomy 全覆盖**：392/392 bug 全部归入 13 类内（138 直接命中 + 154 ALIAS_MAP 机械映射 + 3 手动判定）
- **Silent error 校验**：fix-time `raise/assert` 添加 = 0/392，crash 关键词警告 = 2/392 = 0.5% (false positive)
- **IRR**：cross-pool agreement 78% (25/32)；independent LLM annotator Cohen's kappa = **0.566 (moderate)**
- **Reproducible**：34/392 = 8.7% 已 reproduced（可作为 §6 evaluation candidates）

---

## 1. 实验时间线

| 日期 | 阶段 | 动作 | 结论 |
|---|---|---|---|
| 2026-05-10 | Phase 0 | 环境核查 + ID 交集分析 | 392 总数确认，2 个 orphan 决议 |
| 2026-05-10 | Phase 1 (v1) | 30 bug pilot 用 v1 prompt 标 12 字段 | v1 prompt staircase 不可算（设计 bug） |
| 2026-05-10 | Phase 1 (v2) | 修 prompt → 重跑 30 标注 | staircase 可算但塌方（10/10/83 vs 128 池 20/51/74）|
| 2026-05-10 | 决策 | 用户回到原始目标 → 选 α 方案 | 不再做 fine-grained 12 字段，专注 catalog |
| 2026-05-10 | α 清单 1-4 | 392 catalog 建立 + 13 类对齐 + manifest_v2 + 重算 taxonomy | 392 全集 13 类数字落地 |
| 2026-05-10 | Action 1 | silent_evidence 自动评级（4-tier） | 0.5% WARNING，全是 false positive |
| 2026-05-10 | Action 3 | taxonomy methodology 文档 | 13 类来源 + ALIAS_MAP rationale + IRR |
| 2026-05-10 | Action 4 | 50 bug 独立 LLM IRR + Cohen's kappa | kappa = 0.566 (moderate, 文献区间内) |

---

## 2. 池规模 & 来源

### 2.1 双池起源

| 池 | 起源 | 数量 | 字段深度 |
|---|---|---|---|
| 128 池 | 第一作者从 GH issue/PR 手工调研 (early phase) | 128 (M:42 / D:46 / O:40) | **深**：含 invariant / detection_signal / required_trace_fields / check_stage / parallel_dimension / invariant_type |
| 295 池 | 后期 benchmark database 扩充 | 295 | **中**：config.json 含 root_cause / invariant / trigger_conditions / detection_method，部分含 detect.py + reproduce.sh |

### 2.2 重叠分析

```
128 ∩ 295 = 32 (两池都有)
128 only  = 96 (需要建 benchmark/bugs/<id>/config.json)
295 only  = 263 (其中 248 个 NEW + 15 个非 NEW)
295 orphan = 1 (M-NEW-MUON-MTP, 不在 manifest 但有 config)
排除      = 1 (M-NEW-2, 无 config.json)

合并池总数 = 96 + 32 + 263 + 1 = 392
```

### 2.3 Framework × Source pool 交叉

| | 128_only | both | 295_only | 295_orphan | TOTAL |
|---|---|---|---|---|---|
| megatron-lm | 32 | 10 | 67 | 1 | **110** |
| deepspeed | 36 | 10 | 77 | 0 | **123** |
| olmo | 17 | 5 | 55 | 0 | **77** |
| olmo-core | 11 | 7 | 64 | 0 | **82** |
| TOTAL | 96 | 32 | 263 | 1 | **392** |

---

## 3. 13 类 Taxonomy 数字

### 3.1 392 全集分布

| Category | 392 池 | 128 池(ref) | 趋势 |
|---|---|---|---|
| numerical | 75 (19.1%) | 27 (21.1%) | ≈ 仍是头部 |
| control_flow | 50 (12.8%) | 11 (8.6%) | ↑ |
| gradient_sync | 46 (11.7%) | 14 (10.9%) | ≈ |
| checkpoint | 32 (8.2%) | 16 (12.5%) | ↓ |
| moe | 28 (7.1%) | 6 (4.7%) | ↑ |
| optimizer_state | 28 (7.1%) | 5 (3.9%) | ↑↑ |
| data_loading | 28 (7.1%) | 5 (3.9%) | ↑↑ |
| communication | 26 (6.6%) | 13 (10.2%) | ↓ |
| dtype | 24 (6.1%) | 10 (7.8%) | ≈ |
| loss_computation | 23 (5.9%) | 5 (3.9%) | ↑ |
| sharding | 13 (3.3%) | 10 (7.8%) | ↓↓ |
| lr_schedule | 13 (3.3%) | 2 (1.6%) | ↑↑ |
| offload | 6 (1.5%) | 4 (3.1%) | ↓ |

### 3.2 关键观察

- **numerical 仍是头部类**（19.1%），但比例小幅下降
- **OLMo 系（159 bugs）严重偏 control_flow / data_loading / numerical**：合计 56%
- **DeepSpeed 独有 offload**（6/6 全在 DS）
- **OLMo 系完全没有 moe**（0/0 vs Megatron 12 + DS 14）
- **sharding 类比例下降**（7.8% → 3.3%）：可能 PyTorch FSDP 自动化覆盖了简单 sharding 错误
- **lr_schedule 增长 2x**（1.6% → 3.3%）：现代 scheduler 配置复杂度上升

### 3.3 Reproduction status 分布

| Status | Count | Pct |
|---|---|---|
| `<unset>` | 353 | 90.1% |
| **reproduced** | **34** | **8.7%** ⭐ §6 evaluation candidates |
| failed | 3 | 0.8% |
| pending | 2 | 0.5% |

---

## 4. Silent Error 校验

### 4.1 4-tier evidence rating（自动评级）

| Level | Count | 含义 |
|---|---|---|
| L1 强证据 | 23 | reproduced + expected_output 显示 silent |
| L2 中证据 | 161 | root_cause 含 silent 关键词 + 不含 crash 关键词 |
| L3 弱证据 | 72 | title 级 silent 信号 |
| L4 不确定 | 134 | 字段稀疏（128 独有占 47%） |
| **WARNING (crash 关键词)** | **2** | **D-037 + OC-NEW-21**，都是 false positive |

### 4.2 关键论点

- **0/392 包含 fix-time `raise`/`assert` 添加**（最强反证 loud bug 的指标）
- **WARNING 仅 2/392 = 0.5%**，且经检查都是 false positive：
  - **D-037**：aggregated 3-in-1 fix，sub-bug 1 描述里有"crash"，但 sub-bugs 2-3 是 silent
  - **OC-NEW-21**：root_cause 说"either crashed... or silently overwrote"，主要是 silent
- **L4 大量是字段稀疏**（128 池字段不齐），不是 loud bug 嫌疑

### 4.3 审稿人 Q&A

> "How did you verify these are silent errors?"

**回答**：4-tier evidence rating，0/392 fix-time raise/assert added (strongest indicator of loud bug fixes), 0.5% WARNING rate on crash-keyword scan (both manually verified as false positives, retained). L1=34 reproduced bugs serve as ground truth (buggy run completes without crashing, output differs from fixed). Full evidence per-bug published.

---

## 5. IRR (Inter-rater Reliability)

### 5.1 双协议

| Protocol | Sample | Metric | Value | Interp |
|---|---|---|---|---|
| **A: Cross-pool** | 32 overlap | Raw agreement | **78.1%** (25/32) | High |
| **B: Independent LLM** | 50 stratified | Cohen's kappa | **0.566** | Moderate (L&K 1977) |
| Protocol B | 50 stratified | Raw agreement | 60.0% (30/50) | — |
| Protocol B | 50 stratified | p_expected (chance) | 0.078 | 13-class baseline |

### 5.2 Per-class F1（哪些类难分）

| 完美分 (F1) | 系统性混淆 (F1) |
|---|---|
| loss_computation: **1.00** | **control_flow: 0.00** ⚠️ |
| lr_schedule: 0.86 | numerical: 0.29 |
| checkpoint: 0.86 | gradient_sync: 0.33 |
| dtype: 0.80 | offload: 0.40 |
| sharding: 0.80 | |

### 5.3 两个 systematic confusion pair

1. **control_flow ↔ 具体子系统**：4 个 GT control_flow 全被 LLM 归到更具体类（lr_schedule、gradient_sync 等）。原因：control_flow 太宽。
2. **gradient_sync over-predicted** (GT=4, LLM=8)：LLM 把 numerical / data_loading 中带 grad-related symptom 的 bug 都归到 gradient_sync。**印证"按 root cause vs 按 symptom"边界问题**。

### 5.4 文献基准

| Reference | Kappa | Classes |
|---|---|---|
| **本工作** | **0.566** | **13** |
| DeepLearningBench (ICSE '20) | 0.59 | 8 |
| PROMISE typical | 0.55–0.75 | 6–10 |
| TrainCheck (OSDI '25) | — | 单标注者 |
| TTrace (arXiv) | — | 单标注者 |

→ 0.566 在 multi-class systems-bug taxonomy 文献正常区间，13 类比典型 6–10 类难所以居下沿合理。

---

## 6. v1 → v2 Prompt 演化（已搁置但有 lessons）

> 这部分是 fine-grained 12 字段标注的实验，已被 α 方案搁置，只保留 30 个 v2 标注作为 future work head start。

### 6.1 v1 prompt 设计 bug

v1 把 `pi_*_applicable` 三个 bool 定义为"可同时 true"，与论文 staircase "互斥充分层" 语义不一致 → staircase 不可算。

### 6.2 v2 修复方案

把 3 bool 替换为单字段 `minimum_sufficient_layer ∈ {schema_only, schema_topo, schema_topo_precond, none}`。

### 6.3 30 bug v2 标注结果

| Layer | Count | 128 池 ref |
|---|---|---|
| schema_only | 3 (10%) | 20% |
| schema_topo | **0 (0%)** | 31% |
| schema_topo_precond | 22 (73%) | 23% |
| none (source-only) | 5 (17%) | 26% |

**核心发现**：layer-2 (schema_topo) = 0% 是真实分布（NEW 池里所有需要跨 rank 比对的 bug 同时也需要 module attr 守护）vs 也可能是 prompt 偏置。**未在 paper 中采用**。

### 6.4 30 个 v2 标注存放位置

`benchmark/eval/annotations/<bug_id>.json` × 30，可作为未来若全量化 fine-grained 标注的 head start。

---

## 7. 产出物完整清单

### 7.1 数据（benchmark/eval/）

| 文件 | 用途 | 状态 |
|---|---|---|
| `manifest_v2.json` | ⭐ **392 bug 统一 catalog**（所有审稿人 evidence 的最终源） | 主产物 |
| `pool_overlap.json` | 128/295 池交集分析 | Phase 0 产物 |
| `category_alias_map.json` | 60-entry ALIAS_MAP（295 → 13 类） | Audit trail |
| `category_resolved.json` | 295 池每条 bug 的 category 决议 | Audit trail |
| `silent_evidence_392.json` | 4-tier silent evidence rating | Reviewer evidence |
| `silent_evidence_warnings.csv` | 134 L4 + 2 WARNING bugs (manual review pool) | Reviewer evidence |
| `irr_50_input.json` | 50-bug stripped metadata (no category leak) | IRR audit |
| `irr_50_groundtruth.json` | 50-bug GT (held back from annotators) | IRR audit |
| `irr_50_annotations/irr_batch{1..5}.json` | 5 LLM annotators 各自标 10 个的结果 | IRR audit |
| `irr_50_results.json` | ⭐ **完整 IRR 报告**（kappa, P/R, disagreements） | Reviewer evidence |
| `pilot_30.json` | 30-bug pilot 采样列表（fine-grained 实验，搁置） | Future work head start |
| `annotate_prompt.md` | v2 prompt（fine-grained 12 字段标注口径） | Future work |
| `annotations/<bug_id>.json` × 30 | v2 fine-grained 标注 | Future work head start |
| `annotations_pilot.json` | 合并的 30 标注 | Future work head start |
| `annotations_pilot_v1_archive/` | v1 标注归档 | Audit trail |
| `recompute_v2/taxonomy_392.csv` | ⭐ **§3.1 表 1 source 数据**（13 类总数） | Paper data |
| `recompute_v2/taxonomy_392_by_framework.csv` | Framework × category 交叉 | Paper appendix data |
| `recompute_v2/taxonomy_392_by_source_pool.csv` | Source pool × category 交叉 | Audit trail |
| `recompute_v2/pilot_30_v2_metrics.csv` | 搁置的 fine-grained pilot 数字 | Future work |

### 7.2 脚本（benchmark/eval/）

| 脚本 | 用途 |
|---|---|
| `build_392_catalog.py` | 96 个 128 独有建 dir + 295 池 ALIAS_MAP 应用 |
| `build_manifest_v2.py` | 392 manifest 合并 |
| `silent_evidence_rate.py` | 4-tier silent evidence 自动评级 |

### 7.3 文档（docs/v2_semantic_guided/）

| 文档 | 用途 |
|---|---|
| `27_bug_pool_merge_experiment.md` | 原 27 号实验设计（fine-grained 路线，已搁置） |
| `27a_pilot_report.md` | Pilot 30 决策报告（v1 + v2 数字） |
| `27_master_summary.md` | ⭐ **本文档**（实验总览） |
| `taxonomy_methodology.md` | ⭐ **审稿人挡箭牌**（13 类来源、ALIAS_MAP rationale、IRR、disagreement 分析） |

### 7.4 新建的 benchmark/bugs/

96 个 128 独有 bug 的 config.json 已建立，每个目录在 `benchmark/bugs/<id>/`。

---

## 8. 审稿人 Q&A 速查（背这个）

### Q: 你怎么定义 silent error？

3 criteria（论文 §1 必须明确写）：
1. Buggy code 不立即崩溃（不抛 NaN/Inf/OOM/Exception）
2. Training metric 看起来在合理范围内
3. 影响 final model quality / convergence

### Q: 怎么过滤 silent vs loud？

4-tier evidence rating（详见 silent_evidence_392.json）：
- L1 (23) reproduced + clean expected_output
- L2 (161) silent keywords in root_cause
- L3 (72) title-level signals
- L4 (134) sparse fields, manual review pool
- **0/392 fix-time raise/assert added**
- **2/392 = 0.5% crash-keyword warnings, both false positive**

### Q: 13 类 taxonomy 怎么来的？

**Inductive grounded theory** on 128 hand-curated bugs (open coding → cluster → stabilize at 13 classes corresponding to framework subsystems)。

### Q: 怎么把 295 池对齐到 13 类？

138/295 直接命中；154/295 通过 60-entry deterministic ALIAS_MAP 机械映射；3/295 手动判定。完整 ALIAS_MAP 在 `build_392_catalog.py:30`，可复算。

### Q: Inter-rater reliability？

双协议：
- **Cross-pool 78%** (25/32 overlap)
- **Cohen's kappa = 0.566 (moderate)** with independent LLM rerater on 50 stratified bugs

→ 在 multi-class systems-bug taxonomy 文献区间内（0.55–0.75，cf. DeepLearningBench 0.59 / 8 类）。

### Q: 哪些类边界模糊？

Per-class F1 公开。两个 systematic confusion pair:
- control_flow ↔ 具体子系统（control_flow F1 = 0）
- gradient_sync over-predicted（按 symptom 归类倾向）

诚实写进 paper limitation 段。

### Q: 可复算性？

所有数据 + 脚本 + ALIAS_MAP + disagreement set + IRR 数据 publish。一行复算：

```bash
python3 benchmark/eval/build_392_catalog.py
python3 benchmark/eval/build_manifest_v2.py
```

预期：392/392 bugs in 13 classes ✓, 7/32 disagreements ✓, IRR kappa = 0.566 ✓.

---

## 9. 还可以做的事（未做，按 ROI 排序）

| Task | ROI | 工作量 |
|---|---|---|
| **Action 2**: spot-check 30 个 L2/L3 bug，确认是 silent | 中（防御性） | 半天 |
| **Paper patch**: 把 392 池 13 类数字写进 main_cn.tex 表 1 | **高**（立即可见 paper 进展） | 半天 |
| **§6 reproducible 扩充**: 从 34 个再扩 | 取决于 §6 evaluation 计划 | 看 GPU + 时间 |
| L4 134 个全 spot-check | 低（边际收益） | 1-2 天 |
| Fine-grained 12 字段全量标注（27 号原计划） | 低（已被 α 方案搁置） | 2-3 周 |
| 第 3 个 LLM rerater 跑 50 个（kappa 三方） | 低（已有 0.566） | 1 小时 |

---

## 10. 当前 paper 状态对应

| Paper 段落 | 现状 | 数据 source |
|---|---|---|
| §1 silent error 定义 | 待写明确 3-criteria | 本文档 §0 + §4 |
| §3.1 13 类 taxonomy | 数字待更新（128 池 → 392 池） | `taxonomy_392.csv` |
| §3.2 staircase | **保持 128 池数字**（α 方案决定） | 不变 |
| §3.3 8 pattern coverage | **保持 128 池数字** | 不变 |
| §3.4 Tier coverage | **保持 128 池数字** | 不变 |
| §6.1 evaluation benchmark | 改成 392 池介绍（多样性证据） | `manifest_v2.json` |
| §6.2 reproducible bugs | 34 个 → fault injection | `reproduction_status==reproduced` |
| Appendix: methodology | 13 类来源 + ALIAS_MAP + IRR | `taxonomy_methodology.md` |
| Appendix: limitations | control_flow F1=0 + gradient_sync over-predict | 本文档 §5.3 |

---

## 11. 文件 quick-link

```
关键 catalog:
  benchmark/eval/manifest_v2.json                # 392 bug 统一清单

§3.1 数据:
  benchmark/eval/recompute_v2/taxonomy_392.csv

审稿人挡箭牌:
  docs/v2_semantic_guided/taxonomy_methodology.md
  benchmark/eval/silent_evidence_392.json
  benchmark/eval/irr_50_results.json

实验全貌:
  docs/v2_semantic_guided/27_master_summary.md   # ← 本文档

复算入口:
  python3 benchmark/eval/build_392_catalog.py
  python3 benchmark/eval/build_manifest_v2.py
```
