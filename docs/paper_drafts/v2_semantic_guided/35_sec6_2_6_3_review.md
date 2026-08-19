# 35. §6.2 / §6.3 实验写法 Review

> 日期：2026-05-17  
> 对象：`main_cn.tex` §6.2「检测有效性」与 §6.3「诊断准确率」  
> 核心结论：这两节目前的问题主要不是语言，而是实验口径。主实验把真实 replay bug、真实语义 surrogate、synthetic/minimal surrogate、boundary case 混在一个 22-case 表里统计；§6.3 又把“规则名有语义”写成“诊断准确率”，但没有逐 case 的 ground truth 对照表。这个形态会被系统/数据库会议 reviewer 追问。

## 1. 最大问题：主检出率混入 surrogate

当前 §6.2 把 `SDC-22` 写成主检测集合，并报告 `TrainAudit 20/22`、`TrainCheck 5/22`、`Naive 0/22`。但现有审计文件已经把这个口径废弃：

- `docs/v2_semantic_guided/34_real_sdc22_experiment_requirements.md` 明确要求：主文 Table 6 / 主检出率必须 real-only。
- `benchmark/eval/real_sdc/table6_current_audit.csv` 显示：
  - `ID1/CC1/PE1/AV1/TA1/SC1/CW1/LN1` 是 synthetic surrogate，应移到附录 sanity check。
  - `CF1/CM1/OF1` 是 real blueprint surrogate，其中 `M-010`、`D-029` 可作为真实替换候选，`O-014/CM1` 已降级。
  - `DL2/O-022` 没有 commit pair，已降级。
- `benchmark/eval/real_sdc/SMOKE_REPORT.md` 当前可稳定写入主文的 real replay 结果是 `N_real=9`，另有 `B3/B8` 因环境不兼容 pending。

这和 OSDI/VLDB 常见写法的差异在于：他们通常会把 primary benchmark 和 micro/synthetic study 分开。主表中的每一行要能追溯到 issue/PR/commit/log；synthetic 或 unit-test style case 可以作为 ablation/sanity，不进入主 detection rate。

**建议改法：**

1. 主文 §6.2 改成 `Real-SDC-N`，只统计真实上游 bug / commit pair。
2. 表格第一列加 provenance：`case_id / issue-or-commit`，或在附录加 provenance table。
3. `ID1--LN1` 这 8 个 case 不再放主表；放到附录“Pattern sanity: P9--P16 surrogate check”。
4. 不再固定写 `22`、`20/22`、`13/13`。按最终真实集合实际规模写，例如：

```text
我们在 Real-SDC-9 上评估当前可重放的真实静默错误；另报告 1 个真实 boundary case 和 8 个附录 surrogate sanity case。主检出率只在 Real-SDC-9 上计算。
```

如果后续补齐 `B3/B8` 专用 venv 或 E2 的真实 P9/P11/P13/P15/P16 候选，则把 `N` 机械更新，而不是为了 22 行凑数。

## 2. Baseline 对比现在容易被认为“不公平或不可复算”

§6.2 的 narrative 说是 same-harness controlled comparison，但 reviewer 会继续问：

- 每个 case 的 buggy / reference fixed / held-out fixed 是否分开？
- TrainCheck 的 `FAIL` 是否和 `MISS` 分开统计？
- TrainCheck / Naive 是否也跑在 real-only 主集合上，还是只跑在 surrogate harness 上？
- `0 FP` 是一次 fixed rerun，还是多 seed / long clean audit？

当前仓库里 `benchmark/eval/real_sdc/real_sdc_same_harness.csv` 主要是 TrainAudit 的 real smoke 结果，还没有 real-only 三方完整矩阵。因此主文如果继续用三方表，应明确它是 `D2-sanity same-harness`，不能作为 real-only 主结果。

**建议改法：**

主文 §6.2 分成两个表或两个段落：

- **Main real replay result**：Real-SDC-N，只算真实 bug。若目前只有 TrainAudit 跑通，就只报告 TrainAudit，并把 B3/B8 标成 `env-pending, excluded from numerator`。
- **Controlled surrogate sanity**：D2/SDC-22 或 P9--P16 surrogate，用于说明方法覆盖 pattern 与 baseline failure mode，但标题和正文不能叫“bug 主集合”，也不能进入 abstract/main claim。

更好的最终形态是补跑：

```text
benchmark/eval/real_sdc/real_sdc_same_harness.csv
```

其中每个工具都有：

```text
phase ∈ {reference_fixed, buggy, heldout_fixed}
verdict ∈ {DETECTED, MISS, CLEAN, FAIL, NOT_RUN}
```

然后主表直接由 CSV 生成。

## 3. §6.3 不是“诊断准确率”，而是“诊断可用性”的定性论证

§6.3 目前的强 claim 是：规则名就是诊断特征、层次结构提供两级粒度、RCA agent 能追根因。但它没有量化实验：

- 没有 `ground_truth_category` vs `predicted_parent_category`。
- 没有 `expected_leaf_rule` vs `predicted_leaf_rule`。
- 没有 `suspect_object` / `first_bad_step` 的正确性统计。
- 没有 blind review 或 debug usefulness 标注。

因此标题叫「诊断准确率」会非常危险。OSDI/VLDB reviewer 很可能会说：这是 qualitative case study，不是 accuracy experiment。

**两种改法二选一：**

### 方案 A：补量化诊断表，保留“准确率”

补一个 `diagnosis_accuracy.csv`，字段至少包括：

```text
case_id, ground_truth_category, expected_leaf_rule,
predicted_parent_category, predicted_leaf_rule,
suspect_object, first_bad_step,
correct_parent, correct_leaf, useful_for_debug, reviewer
```

主文 §6.3 报：

```text
parent-category accuracy = x/N
leaf-rule accuracy = y/N
debug-useful = z/N
```

两个 boundary FN 不进入诊断分母，单独列 `no_alarm_by_design`。

### 方案 B：不补实验，降调标题和 RQ

把标题改为「诊断信息质量」或「诊断可用性」。正文写成：

```text
本节不是报告一个独立分类器的准确率，而是审视 TrainAudit 告警携带的调试信息：规则名、父类 taxonomy、嫌疑对象、首次违规 step，以及 RCA probe 如何压缩搜索空间。
```

然后保留 1--2 个 case study，但删除“20 个都准确”这类未被表格支撑的语句。

## 4. 建议的 §6.2 结构

推荐按系统论文更熟悉的模板重排：

1. **Question**：能否在真实 replay bug 上检测语义静默错误？
2. **Dataset**：Real-SDC-N 的入选标准、provenance、排除规则。
3. **Protocol**：buggy run / fixed run / held-out fixed run，TrainCheck 的 reference fixed 单独说明。
4. **Metrics**：detection、fixed FP、FAIL 单独列。
5. **Main Result**：主表只放真实 case。
6. **Baseline Failure Analysis**：解释 baseline 漏检，但不要用 surrogate 主导主 claim。
7. **Boundary**：把 LC1/O-003 写成 trace-observability boundary，而不是规则失败。
8. **New Bug Case Study**：Megatron issue #4641 作为 external validation，不计入 rate。

可替换的主结果句式：

```text
在当前 smoke-passed 的 Real-SDC-9 上，TrainAudit 检出 9/9 个真实可重放静默错误，并在对应 fixed runs 上产生 0/9 false positives。B3 和 B8 的真实 bug recipe 已审计通过，但当前共享环境与对应 DeepSpeed 版本不兼容，因此作为 env-pending 行报告，不计入分母。LC1/O-003 是真实 boundary case；其亚百分比 drift 低于当前在线 trace schema 的稳定可观测阈值，按设计不计入 detector-coverable 分母。
```

如果不想在主文暴露 `smoke-passed` 这个工程词，可以改成：

```text
在本文当前可重放的 Real-SDC-9 集合上……
```

## 5. 建议的 §6.3 结构

如果不补 `diagnosis_accuracy.csv`，建议这样写：

1. **Question**：告警是否携带足够调试信息？
2. **Method**：对 §6.2 检出的 real cases 审查 alert artifact。
3. **Artifact fields**：rule、parent category、suspect object、first bad step、context events。
4. **Result**：用表格展示 4--6 个代表行，而不是声称“准确率”。
5. **Case Study**：保留 H5，但写明是 end-to-end diagnosis demonstration，不进入检测率。
6. **Comparison**：TrainCheck 若未检测，则无诊断；若检测，只报告 relation-template violation，需要人工映射。

推荐标题：

```tex
\subsection{诊断信息质量}
```

推荐开头：

```text
本节不把诊断建模为一个事后分类任务，而是评估告警本身是否携带调试所需的信息。我们检查 §6.2 中 TrainAudit 检出的真实 replay cases 的告警 artifact：触发规则、taxonomy 父类、嫌疑对象、首次违规 step 与上下文事件。若没有独立 blind review，本节只报告诊断可用性，不报告 accuracy。
```

## 6. 需要同步改的地方

如果按上面的 real-only 口径改正文，还需要同步：

- Abstract / Introduction 中所有 `20/22`、`25/27`、`13/13` 之类主 claim。
- §6 开头的 Workload 段落：不要说主 workload 是 22 bug，改为 Real-SDC-N + appendix surrogate sanity。
- §6 的 Metrics 段落：如果 §6.3 降调，就把「诊断准确率」改成「诊断信息质量」。
- Appendix：新增 provenance table，并把 `ID1--LN1` 移到 sanity/microbenchmark。
- Figure/Table caption：避免用 “bug” 指 surrogate；用 “case” 或 “surrogate case”。

## 7. 一句话判断

你感觉“不像 OSDI/VLDB”的地方主要是：当前文字先给强结果，再解释集合构造；而高质量系统论文通常先把 benchmark provenance 和 protocol 钉死，再报结果。把主结果收敛到 real-only、把 surrogate 移到附录、把 §6.3 从“准确率”改成可量化表或降调为 qualitative diagnosis utility，这两节就会稳很多。
