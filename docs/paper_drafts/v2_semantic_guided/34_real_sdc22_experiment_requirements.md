# 34. Real-only SDC Evaluation Requirements

> 当前日期：2026-05-17。  
> 目的：把主文 Table 6 / SDC-22 的实验集合重新收敛到“真实静默错误”底线。  
> 核心原则：主实验中的每一行必须能追溯到真实上游 bug / issue / PR / commit。Synthetic 或 minimal surrogate 只能进入附录的 sanity check，不能计入主表检出率。

## 0. 结论先行

当前 Table 6 中的 `CF1/CM1/OF1` 与 `ID1--LN1` 不能默认作为主实验 bug 计数。

- `CF1/CM1/OF1` 是从真实 bug 语义抽象出的 surrogate。它们可以尝试被真实 blueprint bug 替换：`CF1 -> M-010`，`CM1 -> O-014`，`OF1 -> D-029`。
- `ID1--LN1` 是 P9--P16 的 minimal surrogate。它们可以证明 pattern/unit-test 能跑通，但不是真实上游静默错误复现，不能放进主表当 bug。
- 如果某个 pattern 暂时找不到真实可复现 case，就不要硬补。正文可以写“该 pattern 来自 392 真实 bug 的归纳，但未进入端到端 replay 集合”。

主表的目标不是固定 22 行，而是一个可审计的真实集合：`N_real` 个真实静默错误 + 可选 `N_boundary` 个真实边界案例。最终数字按实验结果更新，不手工凑数。

## 1. 入选标准

一个 case 只有同时满足下面条件，才能进入主文 Table 6 和主检出率统计。

1. **真实来源**：必须有真实上游来源之一：issue URL、PR URL、fix commit、buggy/fixed commit pair，或项目历史中可审计的 bug record。
2. **静默错误**：buggy run 在目标触发条件下不以 crash/assert/NaN explosion 作为主要表象；训练或训练相关逻辑能够继续执行，但语义结果错误、状态错误或轨迹偏离。
3. **可复现证据**：至少有一个 runnable script 或 driver，能在 buggy 版本触发 violation，并在 fixed 版本或修复配置上保持 silent/clean。
4. **可对照**：优先要求 `buggy_commit` 与 `fixed_commit`。如果没有 commit pair，必须说明 ground-truth 对照是什么，例如修复配置、upstream PR diff、或手写 patch。
5. **可记录**：必须保存 stdout/stderr、环境信息、命令、seed、GPU 数、commit hash、verdict CSV。
6. **不靠自造语义**：不能为了覆盖某个 pattern 人为构造一个没有真实 bug 来源的 minimal example 并计入主表。

允许进入附录但不能进入主检出率统计的 case：

- minimal surrogate / unit-test style case；
- 只验证 rule implementation 的 inline check；
- 无真实来源链接的 toy example；
- 真实 bug 的抽象再现，但没有跑真实 bug 或没有明确 blueprint 映射。

## 2. 需要补的实验任务

### E0. 审计当前 Table 6 每一行

输出文件：`benchmark/eval/real_sdc/table6_current_audit.csv`

字段：

```text
case_id, keep_in_main, source_kind, original_bug_id, framework, issue_url,
buggy_commit, fixed_commit, is_real_silent_error, is_surrogate,
is_boundary, current_table_row, required_action, notes
```

`source_kind` 只允许：

- `real_commit_pair`
- `real_issue_or_pr`
- `real_boundary`
- `surrogate_from_real_blueprint`
- `synthetic_surrogate`
- `unknown`

当前判断建议：

| 当前 ID | 初步判断 | 动作 |
|---|---|---|
| `B1/B3/B8/B11/B12` | 真实 bug 候选 | 保留，但补齐 issue/commit/log |
| `M-020/O-005/O-NEW-9/OC-NEW-2` | 真实 bug 候选 | 保留，但补齐 issue/commit/log |
| `CF1` | surrogate from `M-010` | 尝试用真实 `M-010` 替换；否则移到附录 |
| `CM1` | surrogate from `O-014` | 尝试用真实 `O-014` 替换；否则移到附录 |
| `OF1` | surrogate from `D-029` | 尝试用真实 `D-029` 替换；否则移到附录 |
| `ID1/CC1/PE1/AV1/TA1/SC1/CW1/LN1` | synthetic P9--P16 surrogate | 从主表移除；只可作为附录 sanity check |
| `LC1 (O-003)` | 真实边界候选 | 可保留为 boundary，但不算 runtime-observable detected bug |
| `DL2 (O-022)` | 真实边界候选 | 可保留为 boundary，但不算 runtime-observable detected bug |

验收：

- 每行都有 `keep_in_main=true/false`。
- `keep_in_main=true` 的行不能是 `synthetic_surrogate`。
- 该 CSV 能解释为什么 Table 6 最终不是原来的 22 行。

### E1. 用真实 blueprint bug 替换 CF1/CM1/OF1

优先尝试三条替换：

| 原 surrogate | 真实 blueprint | 真实语义 | 目标 |
|---|---|---|---|
| `CF1` | `M-010` | MoE aux loss / invocation count / recompute control-flow | 跑真实 bug 或真实 commit-pair |
| `CM1` | `O-014` | metric/reduction communication inconsistency | 跑真实 bug 或真实 issue reproduction |
| `OF1` | `D-029` | offload restore / dtype or device semantic drift | 跑真实 bug 或真实 commit-pair |

输出文件：

- `benchmark/eval/real_sdc/blueprint_replacements.csv`
- 原始日志：`benchmark/eval/real_sdc/logs/blueprint/<bug_id>/`

字段：

```text
replacement_for, real_bug_id, framework, issue_url, buggy_commit,
fixed_commit, runnable, trainaudit_verdict, traincheck_verdict,
naive_verdict, fixed_verdict, blocker, log_path
```

验收：

- 能跑通则替换主表中的 surrogate ID。
- 跑不通也要记录 blocker，例如 `ENV_BLOCKED`、`OLD_COMMIT_DEP_INCOMPATIBLE`、`NO_FIXED_COMMIT`、`NO_METRIC_STREAM`。
- 跑不通时不要保留 surrogate 在主表里。

### E2. 为 P9--P16 寻找真实 bug；找不到就移除

这里不是要求每个 pattern 都必须有端到端真实复现。要求是：只要进入主表，就必须是真实 bug。

输出文件：`benchmark/eval/real_sdc/p9_p16_real_candidate_search.csv`

字段：

```text
pattern_id, pattern_name, candidate_bug_id, framework, issue_url,
buggy_commit, fixed_commit, evidence_in_392_pool, runnable,
selected_for_main, reason_if_not_selected, notes
```

候选来源：

- `benchmark/eval/manifest_v2.json`
- `benchmark/eval/silent_evidence_392.json`
- `benchmark/eval/category_resolved.json`
- `benchmark/bugs/<bug_id>/config.json`

验收：

- 每个 P9--P16 pattern 至少有一次检索记录。
- 如果没有真实可复现候选，写 `selected_for_main=false` 与原因。
- 不允许用 `ID1--LN1` 直接顶替真实 bug。

### E3. 真实主集合 same-harness 重跑

在 E0--E2 后形成新的真实主集合，例如 `Real-SDC-N`。

输出文件：

- `benchmark/eval/real_sdc/real_sdc_manifest.json`
- `benchmark/eval/real_sdc/real_sdc_same_harness.csv`
- 日志目录：`benchmark/eval/real_sdc/logs/same_harness/<case_id>/`

Manifest 字段：

```text
case_id, original_bug_id, framework, repo, issue_url, buggy_commit,
fixed_commit, source_kind, category, pattern_id, tier, trigger_condition,
buggy_script, fixed_script, trainaudit_rule, selection_reason,
boundary_type, evidence_paths
```

Result CSV 字段：

```text
case_id, original_bug_id, tool, phase, verdict, detected,
violation_count, total_checks, fail_kind, seed, gpu_count,
command, log_path
```

`phase` 只允许：

- `buggy`
- `reference_fixed`
- `heldout_fixed`

`verdict` 只允许：

- `DETECTED`
- `MISS`
- `CLEAN`
- `FAIL`
- `NOT_RUN`

验收：

- 主表所有 `keep_in_main=true` 的真实 case 都有 buggy 与 heldout fixed 结果。
- TrainCheck 的 `reference_fixed` 与 `heldout_fixed` 分开，不能用同一次 fixed run 同时做推断和 FP。
- `FAIL` 不能计入 detected，也不能混成 `MISS`。

### E4. 附录 provenance table

主表只放短 ID / commit。长链接放附录。

输出建议：在论文附录新增一个表，或生成 `benchmark/eval/real_sdc/provenance_table.tex` 后 include。

附录字段：

```text
case_id, original_bug_id, framework, issue_or_pr, buggy_commit,
fixed_commit, source_file_or_script, evidence_log
```

主表建议第一列改成：

- 真实 bug：`B11 / 005afe12`
- 无短 commit 时：`M-020 / 99f999a`
- 边界 case：`O-003 / boundary`

不要再把 `ID1--LN1` 放在主表第一列。

## 3. 最终论文数字怎么写

实验完成后，用真实集合的实际规模写论文，不固定 22。

推荐表达：

```text
We evaluate on Real-SDC-N, a set of N replayable real silent-error cases
plus M real boundary cases, each linked to an upstream issue/PR/commit.
```

中文正文：

```text
我们在 Real-SDC-N 上评估，该集合包含 N 个可重放的真实静默错误
以及 M 个真实边界案例；每个案例均可追溯到上游 issue/PR/commit。
```

如果 P9--P16 没有全部真实 replay，不要写“端到端覆盖 16/16 patterns”。可以写：

```text
The pattern catalog is derived from the 392-case survey. The end-to-end
replay set covers the subset of patterns for which we have replayable
real bugs; the remaining patterns are reported as survey-supported but
not counted in the main replay metric.
```

中文：

```text
模式目录来自 392 个真实案例的归纳；端到端 replay 集合只统计其中
具有真实可复现 bug 的模式，其余模式作为 survey-supported pattern
报告，不计入主检出率。
```

## 4. 不要做的事

- 不要为了让 Table 6 看起来完整而保留 `ID1--LN1`。
- 不要把 synthetic surrogate 写成 bug。
- 不要把 “pattern 能被规则检测” 等价成 “真实 bug 已复现”。
- 不要强行维持 `22 bug`、`20/22`、`13/13`，除非真实集合实验真的支持。
- 不要删除不利结果。真实 bug 跑不通、baseline 崩溃、无 metric stream 都要记录。

## 5. 推荐执行顺序

1. 先做 E0，把当前 Table 6 每行定性。
2. 做 E1，优先把 `CF1/CM1/OF1` 换成 `M-010/O-014/D-029`。
3. 做 E2，从 392 pool 中找 P9--P16 是否有真实可 replay case。
4. 形成 `real_sdc_manifest.json`，确定最终 `N_real + N_boundary`。
5. 跑 E3 same-harness。
6. 生成 E4 附录 provenance table。
7. 最后再改 `main_cn.tex` / `main.tex` 的 Table 6 和 summary 数字。

