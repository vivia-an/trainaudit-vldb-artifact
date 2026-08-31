# 33. Sec 6.1 实验设置补强清单

> **2026-05-17 更正：Table 6 / 主检出率必须采用 real-only 口径。**
> 本文档早期版本允许 `real_semantics_surrogate` 进入 SDC-22 主集合，并把 P9--P16 minimal surrogate 作为主表补充 case。这个口径现在废弃。
> 主文 Table 6 中的每一行必须是真实上游静默错误或真实边界案例；synthetic/minimal surrogate 只能进入附录 sanity check，不能计入主检出率。
> 新的执行要求见 [`34_real_sdc22_experiment_requirements.md`](34_real_sdc22_experiment_requirements.md)。如果本文档后续段落与 34 号文档冲突，以 34 号文档为准。

> 目的：把 `main_cn.tex` §6.1（实现与评估开头：RQ / 硬件 / workload / baseline / metric）里读起来别扭的地方，拆成可以交给实验同学执行的任务。
> 当前日期：2026-05-17。
> 总原则：不要手工改数字。每个结论都要落到 CSV + 原始 log + 复算命令。

## 0. 为什么 6.1 现在别扭

§6.1 现在试图同时承担五件事：说明硬件充分性、说明 SDC-22 的代表性、说明 baseline 公平性、说明 fixed false positive 测量、说明开销目标。问题不是文字，而是有些实验还没有形成可复算证据：

1. **“所有实验都在 8×H200 上完成”容易被质疑。** 当前 overhead 证据主要是 CPU toy / microbenchmark，D2-new 里也有 inline surrogate 检查；如果不补 GPU wall-clock，RQ3 的 `<5%` 会显得悬空。
2. **“8 张 GPU 足够”是论断，不是实验。** 需要至少 1/2/4/8 GPU 的 topology sensitivity，或者把这句话降调成“本研究只验证 invariant 结构，不声称多节点吞吐”。
3. **“workload 取自 392-bug 池”与 Table 6 的 case ID 混合会让人困惑。** 表里同时有真实 bug ID、surrogate case ID、boundary case ID。需要一个 SDC-22 provenance manifest，明确每行来自哪里、为什么入选。
4. **“same harness baseline”需要更硬的证据包。** TrainCheck 结果现在分散在 batch txt / d2 report 中；`baseline_traincheck.py` 里 fixed FP 还有 legacy placeholder 说明。需要统一 CSV，把 reference fixed、buggy check、held-out fixed check 分开。
5. **“20 个运行时可观测 bug”里包含 surrogate。** 这可以成立，但必须写成“case”，并提供每个 surrogate 对应的真实语义来源或 pattern 目的。
6. **RQ2 诊断准确率目前更像定性叙述。** 需要逐 case 的 diagnosis label vs ground truth label 表，否则“诊断准确率”这个 metric 会被审稿人追问。
7. **泛化/跨版本段落的数字需要和 6.1 对齐。** §6.5 说 12 real bugs、31 个月、288/295 driver、104 tests；这些最好有一个 matrix 和日志索引，否则 6.1 的评估范围会显得飘。

## 1. P0 必做实验

### E0. SDC-22 provenance manifest

**要解决的别扭点：** Table 6 的 ID 看起来没有规律；“bug”与“case”混用。

**要做：**
- 新建 `benchmark/eval/sec6_1/sdc22_manifest.json`。
- 每个 case 一行，字段至少包括：
  `case_id, source_kind, original_bug_id, framework, category, pattern_id, tier, role_in_sdc22, buggy_script, fixed_script, trainaudit_rule, traincheck_source, naive_source, selection_reason, evidence_path`。
- `source_kind` 只允许：`real_commit`, `real_semantics_surrogate`, `boundary_real`。
- 对 CF1/CM1/OF1、ID1--LN1，写清楚 surrogate 是补哪个 taxonomy/pattern，以及对应真实 bug 或 pattern 来源。

**验收：**
- 能从 manifest 机械生成 Table 6 的 22 行顺序。
- 每个 case 都能追到脚本和结果文件。

### E1. SDC-22 三方同 harness 完整重跑

**要解决的别扭点：** 当前同 harness 数字散在多个文件里，fixed FP 证据不够集中。

**要做：**
- 对 SDC-22 的 22 个 case，分别跑：
  - TrainAudit buggy detection
  - TrainAudit held-out fixed FP
  - TrainCheck reference fixed infer
  - TrainCheck buggy check
  - TrainCheck held-out fixed check
  - Naive buggy check
  - Naive held-out fixed check
- 输出统一 CSV：`benchmark/eval/sec6_1/sdc22_same_harness.csv`。
- 原始日志放：`benchmark/eval/sec6_1/logs/sdc22_same_harness/<case_id>/`。

**建议命令入口：**
- TrainCheck surrogate：`benchmark/eval/traincheck_surrogates/run_one.sh <case_id>`，但需要确认它真的跑 held-out fixed，而不是只解析 buggy check。
- Naive：`python benchmark/eval/baseline_naive.py --mode synthetic ...`，需要扩到当前 SDC-22 manifest。
- TrainAudit：复用现有 synthetic / inline / driver 路径，但输出必须归一到同一个 CSV。

**验收：**
- 22 个 case 没有缺行。
- `phase` 明确区分 `reference_fixed`, `buggy`, `heldout_fixed`。
- `FAIL` 与 `MISS` 分开：工具崩溃写 `FAIL`，检测能力漏检写 `CLEAN/MISS`。
- Table 6 的 20/22、5/22、0/22 可由 CSV 一条命令复算。

### E2. 真实 GPU overhead / memory / trace-volume 测量

**要解决的别扭点：** §6.1 metric 写 `<5%`，§6.4 目前主要是 CPU 估计；审稿人会要求真实 H200 wall-clock。

**要做：**
- 至少在 Megatron-LM 和 DeepSpeed 上各跑一个干净训练 workload。
- 每个 workload 跑三种配置：
  1. baseline no TrainAudit
  2. TrainAudit sync
  3. TrainAudit async + production sampling 配置
- 每种配置至少 200 steps，最好 3 seeds / repeats。
- 记录：
  `mean_step_ms, p50_step_ms, p95_step_ms, total_wall_s, gpu_mem_peak_mb, trace_mb_per_step, events_per_step, rules_eval_per_step, violations, overhead_pct`。
- 输出：`benchmark/eval/sec6_1/gpu_overhead_h200.csv`。

**验收：**
- 如果 production 配置 overhead `<5%`，正文可以保留 RQ3 目标。
- 如果 `5%--15%`，正文改成“low double-digit overhead / engineering optimization ongoing”。
- 如果 `>15%`，§6.1 和 §6.4 必须降调，不要写 production-ready。

### E3. Held-out fixed 多次重跑 FP audit

**要解决的别扭点：** 0/22 FP 很强，但单次 fixed rerun 对随机训练不够稳。

**要做：**
- 对 SDC-22 的 22 个 fixed case 做至少 3 次 held-out rerun。
- 对真实框架 case，尽量换 seed / dataloader order；对 surrogate，至少换 torch seed。
- 输出：`benchmark/eval/sec6_1/sdc22_fixed_fp_reruns.csv`。

**验收：**
- 每行包含 `case_id, tool, seed, verdict, violation_count, log_path`。
- 目标：TrainAudit 0 FP；TrainCheck / Naive 也记录，但不强行要求。

### E4. 真实 commit subset 上的 baseline sanity

**要解决的别扭点：** “每个案例来自真实 issue/fix 语义”容易被理解成所有 baseline 都跑了真实原始框架 commit。当前主要是 same-harness / surrogate。

**要做：**
- 对 Table 6 里的真实 ID 子集优先跑原始 framework commit：`B1, B3, B8, B11, B12, M-020, O-005, O-NEW-9, OC-NEW-2, O-003, O-022`。
- 每个 case 尝试 TrainAudit / TrainCheck / Naive 三方；跑不通必须写 `BLOCKED_ENV` 或 `NO_METRIC_STREAM`，不能省略。
- 输出：`benchmark/eval/sec6_1/real_commit_baseline_matrix.csv`。

**验收：**
- 至少给出“哪些能直接跑，哪些为什么不能”的完整矩阵。
- 如果 TrainCheck/Naive 在真实 commit 大面积不可运行，正文要明确：“Table 6 是 controlled same-harness comparison；real-commit all-baseline evaluation is engineering-prohibitive”。

## 2. P1 强烈建议实验

### E5. Topology / rank-scale sensitivity

**要解决的别扭点：** “8 张 GPU 足够”缺少实验支撑。

**要做：**
- 选 4 个代表 case：cross-rank replication、process group、optimizer state、Source-B invocation frequency。
- 跑 1/2/4/8 GPU 配置；能做多节点则加 16 GPU / 2-node。
- 记录检测结果、first violation step、overhead、trace volume。
- 输出：`benchmark/eval/sec6_1/topology_scale.csv`。

**验收：**
- 检测 verdict 不随 rank 数异常变化。
- 如果多节点没跑，正文不要暗示已经验证多节点 production。

### E6. Diagnosis accuracy table

**要解决的别扭点：** RQ2 写“诊断准确率”，但现在没有量化表。

**要做：**
- 对 20 个 detected SDC-22 case，收集 TrainAudit 输出的：
  `predicted_parent_category, predicted_leaf_rule, suspect_object, first_bad_step`。
- 与 manifest 里的 `ground_truth_category, expected_rule` 对比。
- 由非实现同学 blind review 一遍，标 `correct_parent`, `correct_leaf`, `useful_for_debug`。
- 输出：`benchmark/eval/sec6_1/diagnosis_accuracy.csv`。

**验收：**
- 能报告 parent accuracy、leaf-rule accuracy、debug usefulness。
- 两个 boundary case 不计入诊断率，但要列为 `no_alarm_by_design`。

### E7. Online detection latency vs TrainCheck offline latency

**要解决的别扭点：** 文中强调 TrainAudit 在线、TrainCheck 离线，但缺少统一计时。

**要做：**
- 对 5 个代表 case 记录：
  `bug_injected_step, first_violation_step, trainaudit_alert_wall_s, traincheck_collect_s, traincheck_infer_s, traincheck_check_s`。
- 输出：`benchmark/eval/sec6_1/detection_latency.csv`。

**验收：**
- TrainAudit 的 first violation step 应接近 bug 出现 step。
- TrainCheck 的总耗时要按 collect/infer/check 分解，避免只写定性“数小时”。

### E8. Cross-framework / cross-version evidence matrix

**要解决的别扭点：** §6.5 的 12 real bugs、31 个月、adapter LoC、288/295 driver 成功率需要证据索引。

**要做：**
- 输出：
  - `benchmark/eval/sec6_1/cross_version_matrix.csv`
  - `benchmark/eval/sec6_1/adapter_loc.csv`
  - `benchmark/eval/sec6_1/driver_pool_success.csv`
- 每个 matrix row 包含 `framework, bug_id, commit, commit_date, rule, tier, buggy_verdict, fixed_verdict, log_path`。

**验收：**
- §6.5 两个表能从 CSV 复算。
- FSDP 如果只有 adapter LoC、没有 bug replay，要明确标 `adapter_only`。

### E9. Clean HEAD scan across frameworks

**要解决的别扭点：** §6.1 提到 Megatron-LM HEAD 上发现 1 个新 bug；如果只扫一个框架，容易像 cherry-pick。

**要做：**
- 对 Megatron-LM、DeepSpeed、OLMo、OLMo-core HEAD 各跑一次 clean scan。
- 输出：`benchmark/eval/sec6_1/head_clean_scan.csv`。

**验收：**
- 每个框架都有 `commit, config, steps, verdict, violations, issue/pr link if any`。
- 如果只有 Megatron-LM 有 candidate，也可以成立，但要显示其他框架扫过且无告警。

## 3. P2 可选但能显著增强说服力

### E10. P9--P16 从 inline check 升级到 online verifier

**要解决的别扭点：** D2-new 的 P9--P16 目前是 inline rule check；如果正文说“同一 online verifier / SQL engine”会显得未完全闭环。

**要做：**
- 把 P9--P16 至少 3 个代表模式接入正式 verifier 路径。
- 选择 ID1、TA1、SC1 或 LN1，覆盖 init snapshot / aliasing / checkpoint / loss normalization。
- 输出：`benchmark/eval/sec6_1/p9_p16_online_verifier.csv`。

**验收：**
- 不再只依赖 `benchmark/eval/d2_extension/trainaudit_inline_d2.py`。
- 能证明 Source-C / Source-B / state rows 进入统一 trace 后仍可检测。

### E11. 34-fault injection regression

**要解决的别扭点：** §6.2 提到 34-fault 注入基准，但它不在 §6.1 的 workload 定义里。

**要做：**
- 重跑 34-fault 注入，输出可复算 aggregate。
- 输出：`benchmark/eval/sec6_1/fault_injection_34.csv`。

**验收：**
- severe/medium 与 subthreshold boundary 分开统计。
- 如果正文继续引用“31 个可观测 severe/medium 全检出”，必须能从 CSV 复算。

## 4. 建议目录结构

```text
benchmark/eval/sec6_1/
  sdc22_manifest.json
  sdc22_same_harness.csv
  sdc22_fixed_fp_reruns.csv
  gpu_overhead_h200.csv
  real_commit_baseline_matrix.csv
  topology_scale.csv
  diagnosis_accuracy.csv
  detection_latency.csv
  cross_version_matrix.csv
  adapter_loc.csv
  driver_pool_success.csv
  head_clean_scan.csv
  logs/
    <experiment>/<case_id>/<timestamp>.log
```

## 5. 对实验同学的执行顺序

1. **先做 E0 + E1。** 这是 Table 6 和 §6.1 workload 公平性的地基。
2. **再做 E2 + E3。** 这是 RQ3 和 0 FP 的地基。
3. **并行做 E4。** 跑不通也有价值，因为它能解释为什么论文用 same-harness。
4. **然后做 E6。** RQ2 没量化表会比较危险。
5. **时间允许再做 E5 / E8 / E9。** 这些主要支撑“泛化”和“8 GPU 足够”的叙事。
6. **P2 只在正文坚持写统一 online verifier 覆盖 P9--P16 时必须做。**

## 6. 写回论文时的决策规则

- 如果 E2 没有真实 GPU `<5%`，§6.1 的 efficiency metric 不要写成已经达成，只写目标或设计预算。
- 如果 E4 真实 commit baseline 跑不全，Table 6 caption 必须明确是 `controlled same-harness comparison`。
- 如果 E0 显示 22 行里 surrogate 占比较高，表头用 `Case ID`，不要用 `Bug ID`。
- 如果 E6 没做完，RQ2 不要写“诊断准确率”，改成“诊断信息质量 / qualitative diagnosis utility”。
- 如果 E5 没做多节点，§6.1 不要写 production multi-node，只写 single-node H200。

## 7. 不要做的事

- 不要把 27-bug D2 extension 数字和当前正文 SDC-22 的 22-bug 数字混在同一个表里。
- 不要把 surrogate 说成真实 framework commit bug；可以说“real-semantics surrogate”。
- 不要把工具运行失败记成漏检；`FAIL` 和 `MISS` 要分开。
- 不要只保存最终 CSV；必须保留每个 case 的原始 stdout/stderr。
- 不要因为某个实验数字不好看就手工改表。数字不好看时，改 claim 的强度。
