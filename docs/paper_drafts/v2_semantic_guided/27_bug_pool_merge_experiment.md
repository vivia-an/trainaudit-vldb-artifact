# 27. Bug Pool Merge Experiment：把 128 池融合进 295 池，重算所有 §3/§4 数字

> **Status**：待执行（2026-05-10 立项）
> **Owner**：实验 agent（独立执行，含决策点）
> **Estimated effort**：4 周（含 pilot → 全量 → 重算 → paper patch）
> **Deadline**：1 个月内

---

## 0. TL;DR

论文 [main_cn.tex:315](../../main_cn.tex#L315) 当前写的"调研 295 → 定位 128 → 采样 23"是**事后包装**，不是真实历史。真实情况是两次独立调研：

- **128 池**：早期人工精细调研（[exp/data/{megatron,deepspeed,olmo}_silent_errors.json](../../exp/data/)，[09_silent_error_benchmark.md](09_silent_error_benchmark.md)），用作 §3 分类与三层框架素材
- **295 池**：后期为 evaluation 扩充的 benchmark database（[benchmark/eval/manifest.json](../../benchmark/eval/manifest.json)），用作 driver pool
- **两池 ID 重叠仅 32 个**

本任务把两池**真正融合**成一个 391-bug 统一池（295 + 96 个 128 独有），并对 295 池里 248 个 NEW bug 补做 128 池那种逐 bug 的精细标注，让 §3/§4 所有数字（13 类 taxonomy、20%/51%/74% staircase、8 模式 66% coverage、Tier 0-6、`after-forward` hook 48 个）都基于统一的标注口径重算。

**用户已授权**：可接受重算后数字变化（如 staircase 从 20%/51%/74% 变成例如 18%/47%/70%）；不接受的红线是数字塌陷到例如 12%/35%/55%（pilot 阶段会判定）。

---

## 1. 数据资产清单（输入）

### 1.1 128 池（早期人工调研，结构化字段已齐）
- `exp/data/megatron_silent_errors.json` — 42 条
- `exp/data/deepspeed_silent_errors.json` — 46 条
- `exp/data/olmo_silent_errors.json` — 40 条（含 OLMo + PyTorch FSDP，未拆分 olmo-core）
- 每条字段（已经有的）：
  ```
  id, repo, issue_or_pr, url, title, description,
  category, parallel_dimension, detection_signal,
  required_trace_fields, check_stage, invariant_type, severity
  ```

### 1.2 295 池（后期 benchmark database，元数据为主）
- `benchmark/bugs/<BUG_ID>/config.json` × 295 个
- `benchmark/eval/manifest.json`（聚合视图）
- `benchmark/eval/manifest_summary.md`（统计概览）
- 每条字段（已经有的）：
  ```
  bug_id, framework, category, buggy_commit, fixed_commit,
  expected_output, detection_method, root_cause, gpu_needed,
  trigger_conditions, invariant, severity, title, issue_url,
  has_detect_py, has_reproduce_sh, has_trainaudit_driver,
  reproduction_status
  ```
- **缺失字段（需补标）**：`parallel_dimension`、`invariant_type`、`required_trace_fields`、`check_stage`、`pi_schema_applicable`、`pi_topo_applicable`、`pi_precond_applicable`、`pattern_id`、`tier_field`、`hook_point`

### 1.3 标注口径锚点
- `docs/v2_semantic_guided/09_silent_error_benchmark.md`（128 池的分类与统计来源）
- `docs/v2_semantic_guided/26_task_baseline_3way.md`（baseline 评估口径）
- `docs/v2_semantic_guided/22_paper_evidence_index.md`（paper 数字索引）
- 论文里 8 个模式的定义：[main_cn.tex:949-964](../../main_cn.tex#L949) 附录 pattern catalog
- 论文里 6 层 schema Tier 定义：[main_cn.tex:535](../../main_cn.tex#L535) 附近

### 1.4 Paper 待重写段落（验收时同步对照）
- [main_cn.tex:315](../../main_cn.tex#L315) — 295/128 调研段
- [main_cn.tex:341](../../main_cn.tex#L341) — 表 1 13 类 taxonomy
- [main_cn.tex:381](../../main_cn.tex#L381) — staircase 20%/51%/74%
- [main_cn.tex:419-423](../../main_cn.tex#L419) — 8 模式 66% coverage
- [main_cn.tex:535](../../main_cn.tex#L535) — Tier 0-6 schema 30%/51%/61%/74%
- [main_cn.tex:557](../../main_cn.tex#L557) — `after-forward` hook 覆盖 48 个
- [main_cn.tex:616](../../main_cn.tex#L616) — workload 段
- [main_cn.tex:949-964](../../main_cn.tex#L949) — 附录 pattern catalog
- 同步：所有 main_cn.tex 改动用 `sync-cn` skill 同步到 main.tex

---

## 2. 交付物（输出）

按交付顺序：

| # | 交付物 | 路径 | 用途 |
|---|---|---|---|
| D1 | 标注 prompt template | `benchmark/eval/annotate_prompt.md` | LLM 标注的 fixed prompt，保证一致性 |
| D2 | Pilot 30 bug 标注结果 | `benchmark/eval/annotations_pilot.json` | Phase 1 输出 |
| D3 | Pilot 决策报告 | `docs/v2_semantic_guided/27a_pilot_report.md` | Phase 1 后 GO/NO-GO 决策 |
| D4 | 全量统一标注 manifest | `benchmark/eval/manifest_v2.json` | 391 bug 全字段统一标注 |
| D5 | 重算结果 CSV | `benchmark/eval/recompute_v2/*.csv` | 各项数字（13 类、staircase、8 模式、Tier、hook）的可复算 source-of-truth |
| D6 | 重算总结报告 | `docs/v2_semantic_guided/27b_recompute_report.md` | 新旧数字对照表 + 解释 |
| D7 | Paper patch | `docs/v2_semantic_guided/27c_paper_patches.md` | 给 main_cn.tex 各段的具体修订建议（含中英文） |

---

## 3. 执行步骤

### Phase 0：环境准备（半天）

```bash
cd /volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025
mkdir -p benchmark/eval/recompute_v2
mkdir -p benchmark/eval/annotations
```

确认依赖：
```bash
python3 -c "import json, pandas; print('OK')"
```

### Phase 1：Pilot（2-3 天）

#### Step 1.1：分层采样 30 个 NEW bug
按 framework 比例采样，从 295 池里只挑 NEW bug（即 ID 含 `NEW` 的，共 248 个）：

- D-NEW-* 共 72 → 抽 10 个
- M-NEW-* 共 61 → 抽 8 个
- O-NEW-* 共 52 → 抽 7 个
- OC-NEW-* 共 63 → 抽 5 个

采样原则：在每个 framework 内按 `category` 字段做 stratified sampling，覆盖 ≥ 5 类。优先选 `reproduction_status == 'reproduced'` 的（如有），再选 `<unset>`。

把采样结果保存为 `benchmark/eval/pilot_30.json`（含 bug_id 列表）。

#### Step 1.2：撰写标注 prompt template (D1)
文件：`benchmark/eval/annotate_prompt.md`

内容必须包含：
1. **背景**：一段话告诉 LLM 这是 silent error 标注任务
2. **输入格式**：bug 的 config.json 全文 + GitHub issue 全文（如可拉到）+ fix commit diff（如可拉到）
3. **输出 JSON schema**（见本文档 §4 字段定义）
4. **13 类 taxonomy 的 1 句话定义**（直接 copy 09 文档 §2 表）
5. **8 个模式的判定 checklist**（见本文档 §4.4）
6. **6 层 Tier 的字段归属**（见本文档 §4.5）
7. **三层属性 (pi_schema/topo/precond) 的判定规则**（见本文档 §4.6）
8. **常见误判反例**：列 5-10 个 LLM 容易标错的边界情况

#### Step 1.3：跑 LLM 标注
- 对 pilot_30 中每个 bug，构造完整 prompt → 调用 Claude（用 prompt caching，prompt template 是 stable prefix）
- 输出统一 JSON 到 `benchmark/eval/annotations_pilot.json`

#### Step 1.4：人工抽检（**要找用户参与**）
- 30 个全量人工 review（pilot 阶段必须 100% 覆盖）
- 修订标注口径（如发现系统性错误）
- 把修订过的 prompt 保存为 v2，进入 Phase 2

#### Step 1.5：合并 pilot 标注 + 重算
把 30 个新标注 merge 进原 128 池，**临时**得到 158-bug 池，重算：
- 13 类 taxonomy 分布
- staircase 三层覆盖率
- 8 模式 coverage
- Tier 0-6 覆盖率
- `after-forward` hook 覆盖数

输出：`benchmark/eval/recompute_v2/pilot_158_metrics.csv`

#### Step 1.6：决策点 (D3)
基于 pilot 数据线性外推到 391-bug 全集，写决策报告 `27a_pilot_report.md`：

| 数字 | 原 (128) | Pilot 158 实测 | 全集 391 外推 | 红线 | 决策 |
|---|---|---|---|---|---|
| schema staircase | 20% | ? | ? | ≥ 15% | GO/NO-GO |
| topo staircase | 51% | ? | ? | ≥ 40% | GO/NO-GO |
| 三层 staircase | 74% | ? | ? | ≥ 60% | GO/NO-GO |
| 8 模式 coverage | 66% | ? | ? | ≥ 55% | GO/NO-GO |

**GO 条件**：所有 4 个指标都不跌破红线。
**NO-GO 应对**：退回方案 A（只融合元数据，§3 数字保持不变，明说"§3 标注基于 128-bug 子集，§6 evaluation 基于 391-bug 池"）。

### Phase 2：全量标注（5-7 天）

如果 Phase 1 GO：

#### Step 2.1：补 96 个 128 独有 bug 的 config.json
128 池里有 96 个 ID 没进 benchmark/bugs/。脚本翻译：
- 输入：`exp/data/{megatron,deepspeed,olmo}_silent_errors.json` 中那 96 条
- 输出：`benchmark/bugs/<id>/config.json`（用 SCHEMA.md 字段，从 128 池字段直接映射 + LLM 补全 missing field）
- 注意：OLMo+FSDP 池里有些 bug 实际是 olmo-core，按 fix commit 的 repo 拆分

#### Step 2.2：跑剩余 218 个 NEW bug 标注
- 用 v2 prompt
- 输出到 `benchmark/eval/annotations/<bug_id>.json`
- **抽检 30%**（约 70 个），重点查：
  - `invariant_type`（容易混 bounded_change vs cross_rank_equality）
  - `pattern_id`（容易把 P3 cross-rank 误标到 numerical bug）
  - `pi_topo_applicable`（容易在 topology-independent bug 上误标 yes）

#### Step 2.3：字段对齐 + 合并
- 原 128 池字段名与新标注字段名做 1-1 映射（写一个 `align_schema.py`）
- 合并所有 391 条到 `benchmark/eval/manifest_v2.json` (D4)
- Schema 校验：所有条目必须有完整的 12 个标注字段（见 §4.1），缺失的标 `null` 并记入失败清单

### Phase 3：重算所有数字（4-5 天）

#### Step 3.1：写 6 个独立的 recompute 脚本
每个脚本读 `manifest_v2.json`，输出一个 CSV：

1. `recompute_taxonomy.py` → `recompute_v2/taxonomy.csv`
   - 13 类 × (Pool count, Smp count) — 注意 Smp 列要明确口径，是 §6 主表 23 还是别的
2. `recompute_staircase.py` → `recompute_v2/staircase.csv`
   - 3 行：schema-only / +topo / +precond，每行一个覆盖率数字
3. `recompute_pattern_coverage.py` → `recompute_v2/pattern_coverage.csv`
   - 9 行（8 个模式 + 合计），每行：pattern_id, type, count, percent
4. `recompute_tier_schema.py` → `recompute_v2/tier_schema.csv`
   - 7 行（Tier 0-6），每行：fields, cumulative_coverage_percent
5. `recompute_hook_coverage.py` → `recompute_v2/hook_coverage.csv`
   - 5 行（before-fwd / after-fwd / main-grad-bwd / after-bwd / before-opt）+ aux taps
6. `recompute_framework_breakdown.py` → `recompute_v2/framework_breakdown.csv`
   - 4 行（Megatron / DeepSpeed / OLMo / OLMo-core），每行总数 + 13 类细分

#### Step 3.2：写新旧对照报告 (D6)
文件：`docs/v2_semantic_guided/27b_recompute_report.md`

格式：每个数字一节，含：
```
## staircase 三层覆盖率
- 原数字 (128 池)：20% / 51% / 74%
- 新数字 (391 池)：X% / Y% / Z%
- 变化原因：[1-2 句话]
- 对论文叙事的影响：[是否需要重写定性表述？]
```

### Phase 4：Paper Patch (D7)（3-5 天）

文件：`docs/v2_semantic_guided/27c_paper_patches.md`

每个待改段落给一个 patch block：
```markdown
### Patch 1: §3 调研段 (main_cn.tex:315)

**原文**：
> 我们调研了来自 Megatron-LM（60）、DeepSpeed（75）、OLMo（70）和 OLMo-core（60）的 295 个静默错误案例...

**改写**：
> 我们建立了一个跨 4 个框架（Megatron-LM、DeepSpeed、OLMo、OLMo-core）共 391 个静默错误案例的 benchmark database...
```

每段中文 patch 后给一段对应英文 patch（同 main.tex 段落）。

最后用 `sync-cn` skill 把 main_cn.tex 改动同步到 main.tex 验证。

---

## 4. 标注口径定义（核心：所有 LLM 标注都要按这套口径）

### 4.1 必填字段 schema (12 个)

```json
{
  "bug_id": "M-NEW-10",
  "framework": "megatron-lm",
  "category": "<13 类之一，见 §4.2>",
  "parallel_dimension": "<DP|TP|PP|EP|CP|SP|FSDP|none|combo>",
  "invariant_type": "<bounded_change|cross_rank_equality|value_equality|dtype_consistency|numerical_consistency|value_range|implementation_equivalence|monotonic|completeness|other>",
  "required_trace_fields": ["<field1>", "<field2>", ...],
  "check_stage": "<before_forward|after_forward|main_grad_in_backward|after_backward|before_optimizer|checkpoint_save|checkpoint_load|all_reduce|build|other>",
  "pi_schema_applicable": true | false,
  "pi_topo_applicable": true | false,
  "pi_precond_applicable": true | false,
  "pattern_id": "<P1|P2|P3|P4|P5|P6|P7|P8|none>",
  "tier_field": "<Tier 0-6, 表示发现这个 bug 最少需要哪一层 schema>",
  "rationale": "<1-2 句话解释为什么这么标，便于人工抽检>"
}
```

### 4.2 13 类 category 定义（同 [09 文档](09_silent_error_benchmark.md) §2）

| 类别 | 定义（1 句话） |
|---|---|
| numerical | 数值计算错误（loss scaling 不当、normalization 漏除、累加溢出等），值"看起来合理但不正确" |
| checkpoint | save/load 时状态丢失或被错误覆盖（含 optimizer state、RNG 状态、step counter） |
| gradient_sync | 梯度的跨 rank 同步/归约错误（accumulation count、reduction op、dtype 不匹配） |
| communication | 通信操作参数错误（wrong group、错的 collective、size 对不上） |
| control_flow | 控制流错误（counter 漏 inc/重复 inc、条件分支走错、初始化顺序倒置） |
| sharding | 参数切分错误（切错维度、切片重叠、cross-shard 边界处理错） |
| dtype | 数据类型错误（float32 期望但得到 bfloat16、scale factor 用错精度） |
| moe | MoE 专用错误（router 初始化、expert 不均、EP group 配置） |
| optimizer_state | optimizer state 错误（momentum/Adam state 跨 rank 不一致、reset 逻辑错） |
| loss_computation | loss 计算逻辑错误（label mask 漏、loss reduction 模式错） |
| data_loading | 数据加载/采样错误（shard 重复、shuffling seed 错、epoch boundary 错） |
| offload | CPU offload 相关同步错误 |
| lr_schedule | learning rate 调度错误（warmup 步数算错、resume 时 step 重置） |

如果 295 池里某 bug 不属于上述任一类，**不要新增类别**，归入最接近的一类，并在 `rationale` 里说明。如果归并困难超过 ~5%，回到本文档加新类。

### 4.3 invariant_type 定义（同 09 文档 §4）

按运行时可检查的不变量类型分：
- `bounded_change`：值的变化在合理范围内（最大类别）
- `cross_rank_equality`：跨 rank 的值应该一致
- `value_equality`：某个值应该等于特定值
- `dtype_consistency`：数据类型应该一致
- `numerical_consistency`：数值精度一致性
- `value_range`：值在特定范围内
- `implementation_equivalence`：两种实现应产生相同结果
- `monotonic`：值应单调（如 step 计数）
- `completeness`：某个集合应完整（如所有 expert 都被访问）
- `other`：上述以外（在 rationale 里说明）

### 4.4 8 模式判定 checklist（核心：决定 pattern_id）

> 详见 [main_cn.tex:949-964](../../main_cn.tex#L949)。每个模式给一个判定问题：

| Pattern ID | Type | 判定问题（yes 才能标） |
|---|---|---|
| P1 | A (trace SQL) | bug 是否表现为"某 trace 字段在某条件下应等于某确定值"？例如 step counter 应单调 +1 |
| P2 | A | bug 是否表现为"两个 trace 字段之间应满足某代数关系"？例如 `grad_norm == sum(grad_chunks)` |
| P3 | A | bug 是否表现为"DP/TP 中本应相同的张量在 rank 间不一致"？（cross-rank replication） |
| P4 | B (runtime hook) | bug 是否需要在某 hook 点检查"参数 dtype / shape 与配置预期一致"？ |
| P5 | B | bug 是否需要在某 hook 点检查"梯度满足 norm/scale 边界"？ |
| P6 | B | bug 是否需要在某 hook 点检查"通信 group 配置与 topology 一致"？ |
| P7 | B | bug 是否需要在某 hook 点检查"模块状态计数器（counter / flag）符合预期"？ |
| P8 | C (静态断言) | bug 能否在模型构建时通过对 `model.named_modules()` 的静态断言发现？ |

每个 bug 最多归入 1 个 pattern。如果跨 2 个，选 detection 阶段更早的那个。如果 8 个都不匹配，标 `pattern_id: "none"`。

### 4.5 6 层 Tier 定义（决定 tier_field）

> 详见 [main_cn.tex:535](../../main_cn.tex#L535)。

| Tier | 字段 | 累计覆盖率（128 池上） |
|---|---|---|
| 0 | parameter checksum, name, rank coords, step, stage, dtype, shape | 30% |
| 1 | + gradient stats (norm, mean, max) | 41% |
| 2 | + loss value | 51% |
| 3 | + control variables (lr, accumulation count) | 56% |
| 4 | + optimizer state checksum | 61% |
| 5 | + 5 个补充字段 (略，见原表) | 68% |
| 6 | + 6 个补充字段（共 27 字段） | 74% |

`tier_field` 取值：bug 能被检出所需的**最低** Tier。例如 dtype mismatch bug → Tier 0（dtype 字段在 Tier 0），grad accumulation count error → Tier 3（accumulation count 在 Tier 3）。

### 4.6 三层属性 (pi_schema / topo / precond) 判定规则

- `pi_schema_applicable: true` ⟺ 存在某个 trace 字段，单看其值就能判定 bug（不需要 cross-rank 比对、不需要 phase 守护）
- `pi_topo_applicable: true` ⟺ 检测需要把"哪些 rank 的字段应该相等"考虑进去（依赖 DP/TP/PP/EP topology）
- `pi_precond_applicable: true` ⟺ 检测只在某个特定阶段 / 模块属性下才有效（例如"仅当 use_distributed_optimizer=True 时检查"）

三个 bool **互不排斥**，可以同时 true。一个 bug 至少要标 1 个 true（否则它不可检测）。

---

## 5. 关键工程约束

### 5.1 Bug ID 命名规则
- 96 个 128 独有 bug 直接保留 128 池里的原 ID（M-001..M-042 等），不重新编号
- 如有 ID 冲突（理论上不应有，因为重叠的 32 个本来就是同 ID），以 benchmark/bugs/<id>/config.json 已有内容为准，但**字段标注以 128 池为准**（更精细）

### 5.2 LLM 标注的可复算性
- 所有标注必须保留 prompt + response 原文到 `benchmark/eval/annotations/<bug_id>.{prompt,response}.txt`
- prompt 用 prompt caching（template 是 stable prefix，per-bug 内容是 dynamic suffix）
- 用 Claude Opus 4.7 1M 模型（与本论文项目一致）

### 5.3 数据 versioning
- 不要覆盖原 128 池的 `exp/data/*.json` 和原 295 池的 `benchmark/bugs/<id>/config.json`
- 所有合并产物放新文件 `benchmark/eval/manifest_v2.json`
- 如果改 config.json 字段，必须在 git 单独 commit（`feat(benchmark): add 12-field annotation to <id>`）

### 5.4 不要做的事
- ❌ 不要改 `exp/data/*.json` 的原数据（128 池是 paper 现有数字的 source of truth，要保留以备审稿人质疑）
- ❌ 不要因为重算后数字降低而手工调整某个 bug 的标注（标注口径一旦定下，所有标注必须按口径机械执行）
- ❌ 不要新增 13 类以外的类别（如确实必要，回报告，不要私自决定）
- ❌ 不要跑任何需要 GPU 的实验（这次纯标注，不复现 bug；reproduced 状态用 manifest 现有的）
- ❌ 不要动 paper 的 main.tex / main_cn.tex（patch 只输出到 27c_paper_patches.md，由用户最终审核后人工应用）

---

## 6. 验收 Checklist

### Phase 1 验收
- [ ] `benchmark/eval/pilot_30.json` 存在，含 30 个 bug_id
- [ ] `benchmark/eval/annotate_prompt.md` 存在，包含 §4 全部口径
- [ ] `benchmark/eval/annotations_pilot.json` 存在，30 条 schema 完整
- [ ] `27a_pilot_report.md` 含 GO/NO-GO 决策与新数字 vs 红线表
- [ ] 用户对 pilot 标注做了人工抽检并签字（在 27a 报告末尾）

### Phase 2 验收
- [ ] 96 个新 config.json 已落地到 `benchmark/bugs/`
- [ ] `benchmark/eval/manifest_v2.json` 含 391 条，schema 完整
- [ ] 失败/缺失字段清单 < 5%（写入 27b 报告）
- [ ] 30% 抽检结果记录在 27b 报告

### Phase 3 验收
- [ ] 6 个 recompute CSV 全部生成
- [ ] `27b_recompute_report.md` 含全部 5 个数字（13 类 / staircase / 8 模式 / Tier / hook）的新旧对照

### Phase 4 验收
- [ ] `27c_paper_patches.md` 覆盖本文档 §1.4 列出的全部 8 段 paper 待改位置
- [ ] 每段 patch 含中英文双版本

---

## 7. 中期汇报模板

每完成一个 Phase，agent 用以下模板向用户汇报，等待 ack 再进下一个 Phase：

```markdown
# Phase X 完成汇报

## 关键数字
[新数字表]

## 风险点
[本 phase 发现的口径模糊 / 标注异常 / 工程问题]

## 下一 Phase 启动建议
[GO / NO-GO，以及如果 GO 的话需要用户先确认的事项]

## 产出物
[文件路径列表]
```

---

## 8. 附录：常见误判反例（标注 prompt 必须包含）

> 以下是 128 池标注时遇到的边界情况，agent 必须在 prompt 中给 LLM 列出来。

1. **dtype bug ≠ pi_schema 单独可检**：bf16 模型里某个 buffer 误存为 fp32，单看 dtype 字段似乎能检出，但实际需要"该 buffer 在 init 阶段应为 bf16"这个 precondition，所以 `pi_schema=true, pi_precond=true`
2. **跨 rank 等价 ≠ pi_topo always-on**：DP-replicated 参数应跨 DP rank 相等，但 TP-sharded 参数不应跨 TP rank 相等。所以 cross-rank-equality bug 要标 `pi_topo=true, pi_precond=true`（precond 是"该参数的 tensor_model_parallel attribute 为 False"）
3. **gradient_sync vs communication 的边界**：bug 在 gradient all-reduce 阶段触发但根因在 group 配置 → 标 `category: communication`；根因在 reduction op 选错 → 标 `category: gradient_sync`
4. **pattern_id 的 P3 vs P4**：如果 bug 的检测可以转化为"trace 数据库上对某字段做 GROUP BY rank HAVING COUNT(DISTINCT value) > 1"，标 P3（Type-A）；如果需要在运行时 hook 里读 module attribute 才能判定，标 P4（Type-B）
5. **Tier 归属看"最简单字段"，不是"最完整字段"**：如果 bug 既能用 dtype 检（Tier 0）又能用 optimizer state checksum 检（Tier 4），标 Tier 0
6. **MoE bug 与 EP topology**：MoE bug 默认 `parallel_dimension: EP`，但如果 fix 在 single-GPU 也能复现（bug 与 EP 无关），标 `none`
7. **checkpoint bug 的 check_stage**：如果 bug 在 load 时显现但根因在 save 时漏存 → check_stage 标 `checkpoint_load`（即"运行时第一次能观测到"的阶段）
8. **OLMo vs OLMo-core 的拆分**：128 池里的 O-XXX 如果 fix commit 在 `allenai/OLMo` repo → framework=olmo；在 `allenai/OLMo-core` repo → framework=olmo-core。不能简单按编号区间分。

---

## 9. 启动指令

实验 agent 启动时按以下顺序执行：

1. 读本文档全文（含 §4 口径定义）
2. 检查 §1 列出的所有输入文件存在
3. 进入 Phase 0 → Phase 1
4. Phase 1 完成后停下，按 §7 汇报模板向用户汇报，**等用户 ack 才能进 Phase 2**
5. Phase 2/3/4 同样在每个 Phase 末尾汇报，等 ack

如果中途遇到口径模糊（例如某 bug 找不到合适的 13 类归属），**不要私自决定**，停下来记录具体 bug ID + 困难，列入汇报。
