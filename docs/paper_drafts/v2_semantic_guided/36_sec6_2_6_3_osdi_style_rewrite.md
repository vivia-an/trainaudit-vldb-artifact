# 36. §6.2 / §6.3 为什么“不像实验”以及怎么按 OSDI/VLDB 风格重写

> 日期：2026-05-17  
> 对象：`main_cn.tex` §6.2「检测有效性」与 §6.3「诊断准确率」  
> 这份文档专门回应一个更根本的问题：当前小节不是“实验写法”，而是“结论写法”。它没有先把实验对象、变量、流程、指标、成功判据讲清楚，所以读者会感觉结果是靠嘴说出来的。

## 1. 你感觉不对的地方是什么

当前 §6.2 / §6.3 的主要问题不是数字大小，而是缺少实验骨架。系统论文里的实验通常长这样：

```text
Question -> Setup -> Method -> Metric -> Result -> Analysis -> Takeaway
```

但现在更像：

```text
Claim -> Table -> Explanation -> More Claim
```

这会造成三个直接问题：

1. **不知道实验要验证什么。**  
   §6.2 开头说“检测有效性”，但没有把实验假设说清楚，例如“是否能在 silent error 的 first bad step 附近触发语义不变量，且 fixed run 不触发”。

2. **不知道实验怎么做。**  
   表前虽然有“评估流程”，但还不够实验化。读者不知道每个 case 跑几次、输入是什么、输出是什么、怎样判定 detected/missed/FP/fail、TrainCheck 的 trace 怎么产生、Naive 的窗口和阈值为何合理。

3. **不知道结论从哪个测量值推出来。**  
   比如 §6.3 说“规则名即诊断特征”，这不是一个自然的实验结果。要变成实验，需要定义诊断任务、ground truth、判分标准，例如 parent category 是否正确、leaf rule 是否正确、suspect object 是否可用。

一句话：现在的问题是“没有实验协议，只有实验结论”。

## 2. OSDI/VLDB 风格通常会怎样写

不是说必须模仿固定模板，但系统论文的 evaluation 小节通常会把每个实验写成一个小闭环：

```text
Goal.
We evaluate whether X can do Y under condition Z.

Setup.
We use dataset D, workload W, hardware H, tool versions V, and baselines B.

Procedure.
For each case, we run A/B/C. We collect signals S. We repeat R times. We separate training/reference/test traces.

Metrics.
We count detection iff ...; false positive iff ...; diagnosis is correct iff ...

Results.
Table N shows ...

Why.
The misses occur because ...

Takeaway.
X works for Y, but not for Z.
```

你现在的 §6.2/§6.3 少的是 `Setup / Procedure / Metrics`，所以读起来像作者在解释自己的系统为什么好，而不是用实验去验证它。

## 3. §6.2 应该验证什么

§6.2 不应该泛泛叫“检测有效性”，而应该拆成一个明确的实验问题：

> **E1: Can TrainAudit detect replayable silent errors before outcome-level metrics expose them, while staying silent on fixed runs?**

中文可以写：

> 本实验验证：当一个真实静默错误被重放时，TrainAudit 是否能在运行时 trace 中触发对应语义不变量；当同一 case 的 fixed 版本或修复配置运行时，TrainAudit 是否保持静默。

这个实验的单位应该是 **case-run pair**，而不是笼统的 “bug”：

- `buggy run`：应该触发；
- `reference fixed run`：给 TrainCheck 学 invariant，如果使用 TrainCheck；
- `held-out fixed run`：测 FP；
- 可选 `repeat/seed`：测稳定性。

### §6.2 必须补的实验定义

建议正文或表注里明确：

```text
Detected: 工具在 buggy run 结束前至少产生一个与该 case ground-truth fault pattern 匹配的告警。
Miss: 工具完成 buggy run 但没有产生匹配告警。
False positive: 工具在 held-out fixed run 上产生任何告警。
Fail: 工具或 harness 崩溃；Fail 不计为 detected，也不和 Miss 混合。
Boundary: 当前 trace schema 下不存在稳定 runtime signal 的真实 case；单独报告，不进入 detector-coverable 分母。
```

### §6.2 推荐结构

```tex
\subsection{检测有效性}

\paragraph{实验目的。}
本实验验证 TrainAudit 是否能在真实静默错误的 buggy run 中触发语义不变量，并在对应 fixed run 中保持静默。我们同时比较两个 baseline：TrainCheck 和 Naive Monitoring。

\paragraph{实验对象。}
我们使用 Real-SDC-N / D2-controlled set。每个 case 包含：来源、buggy 版本、fixed 版本、触发配置、ground-truth fault pattern、期望触发规则。

\paragraph{实验流程。}
对每个 case，我们执行三个 phase：reference-fixed、buggy、held-out-fixed。TrainCheck 只使用 reference-fixed 学习 invariant；所有工具在 buggy 上测 detection，在 held-out-fixed 上测 FP。

\paragraph{指标。}
定义 Detected / Miss / FP / Fail / Boundary。报告 detection rate、FP rate、tool failure count 和 first violation step。

\paragraph{结果。}
表格给出 aggregate，另一个表给出 per-case verdict。

\paragraph{分析。}
解释每类 baseline miss 是因为缺 topology、precondition、schema，还是因为工具失败。
```

注意这里最关键的不是表，而是表前的协议。协议扎实，表才像实验。

## 4. §6.3 应该验证什么

当前 §6.3 的问题更严重：它标题叫“诊断准确率”，但里面没有诊断实验。

如果要叫“诊断准确率”，实验问题应是：

> **E2: Given a detected silent error, can TrainAudit localize it to the correct semantic category, rule-level invariant, suspect object, and first bad step?**

也就是说，§6.3 的输入不是“所有 bug”，而是：

```text
Input = §6.2 中已经 detected 的 buggy runs
Output = diagnosis artifact
Ground truth = case manifest 中人工标注的 root cause/category/rule/object
```

### §6.3 必须补的指标

至少要有这几个：

```text
Parent accuracy:
  predicted semantic category == ground-truth category

Leaf-rule accuracy:
  triggered rule == expected rule / acceptable equivalent rule

Object localization:
  suspect parameter/module/rank 是否命中 ground-truth object

First-step accuracy:
  first_bad_step 是否等于或接近人工标注的 first divergence step

Debug usefulness:
  非实现者 blind review，判断报告是否足以把工程师引到正确代码区域
```

如果没有这些，就不要叫 accuracy。应该改成：

```tex
\subsection{诊断信息质量}
```

或者：

```tex
\subsection{告警可解释性与根因定位案例}
```

### §6.3 推荐结构：如果补量化实验

```tex
\subsection{诊断准确率}

\paragraph{实验目的。}
检测告警只有在能缩小调试范围时才有工程价值。本实验评估 TrainAudit 在已检出 case 上输出的诊断 artifact 是否命中 ground-truth fault category、leaf invariant、suspect object 和 first bad step。

\paragraph{实验对象。}
我们使用 §6.2 中 TrainAudit 检出的 N 个 buggy runs。两个 boundary FN 不触发规则，因此不进入诊断分母。

\paragraph{Ground truth。}
每个 case 的 ground truth 来自上游 fix diff、issue/PR 描述和人工审计 manifest。我们记录 parent category、expected rule、root-cause object 和 first divergence step。

\paragraph{判分标准。}
Parent / leaf / object / first-step 分别判分。若两个规则是同一 fault pattern 的等价表达，则计为 equivalent leaf hit，并在表注说明。

\paragraph{结果。}
表格报告 parent accuracy、leaf-rule accuracy、object localization、debug-useful rate。

\paragraph{案例分析。}
选择 1--2 个多规则触发 case，展示 C1 expander 和 C2 RCA 如何进一步缩小搜索空间。
```

### §6.3 推荐结构：如果不补量化实验

```tex
\subsection{诊断信息质量}

\paragraph{目的。}
本节不报告独立诊断准确率，而是检查 TrainAudit 的告警是否携带调试所需的信息。

\paragraph{方法。}
我们审查 §6.2 中 detected cases 的告警 artifact，包括触发规则、taxonomy 父类、嫌疑对象、首次违规 step 和上下文事件。

\paragraph{结果。}
用一个小表展示 4--6 个代表 case 的 artifact。

\paragraph{案例研究。}
展示复杂 case 的 RCA 链。
```

这样写就诚实很多，也更像系统论文。

## 5. 当前正文中最像“靠嘴说”的句子

下面这些句子不是不能写，但必须有实验定义支撑：

1. **“规则名即为诊断特征，无需额外解释。”**  
   这更像设计哲学，不是实验结果。应改成：在 N 个 detected cases 中，触发规则的 leaf label 与人工 ground truth 一致 x/N。

2. **“层次结构提供两级诊断粒度。”**  
   这是系统设计，不是实验。应给 parent category accuracy 和 leaf-rule accuracy。

3. **“TrainCheck 的 schema 刚性使其诊断失败。”**  
   如果 TrainCheck 未检测，当然没有诊断；但这属于 detection failure，不应直接写成 diagnosis accuracy 0，除非明确定义“未检测则诊断失败”。

4. **“Naive 和 TrainCheck 都能报告 NaN，但无法穿透到 zero_grad 语义假设。”**  
   这是 case study，可以写，但不能替代整体诊断实验。

## 6. 一版更像实验的 §6.2 开头草稿

```tex
\paragraph{实验目的。}
本实验回答一个具体问题：给定一个可重放的静默错误，\textsc{TrainAudit} 是否能在 buggy run 的运行时 trace 中触发与该错误语义相匹配的不变量，并在对应 fixed run 上保持静默？这区别于只观察 loss/NaN/gradient norm 的 outcome-level monitoring，也区别于从 clean trace 中学习行为模板的 TrainCheck。

\paragraph{实验对象与输入。}
每个 case 由四部分组成：buggy 版本或配置、fixed 版本或修复配置、触发该错误的最小训练 workload，以及人工审计得到的 ground-truth fault pattern。对每个 case，我们生成三个运行：reference-fixed run、buggy run 和 held-out fixed run。reference-fixed run 只用于 TrainCheck 的 invariant inference；所有方法都在 buggy run 上测检出，在 held-out fixed run 上测误报。

\paragraph{判定规则。}
若工具在 buggy run 结束前产生至少一个与该 case ground-truth fault pattern 匹配的告警，则记为 detected；若运行完成但没有匹配告警，则记为 miss；若工具或 harness 崩溃，则记为 fail，fail 不计入 detected；若工具在 held-out fixed run 上产生任何告警，则记为 false positive。当前 trace schema 下无稳定运行时信号的 case 单独标为 boundary，不进入 detector-coverable 分母。
```

这三段补上后，后面的表才有“实验”的感觉。

## 7. 一版更像实验的 §6.3 开头草稿

如果补量化表：

```tex
\paragraph{实验目的。}
检测到错误并不等于能帮助工程师修复错误。本实验评估 \textsc{TrainAudit} 的告警是否把错误定位到正确的语义类别、叶级不变量、嫌疑对象以及首次违规 step。输入为 \S\ref{subsec:detection} 中 \textsc{TrainAudit} 已检出的 buggy runs；未触发告警的 boundary cases 不进入诊断分母。

\paragraph{Ground truth 与判分。}
每个 case 的 ground truth 由上游 issue/PR、fix diff 和人工审计 manifest 给出，包括 parent category、expected leaf rule、root-cause object 和 first divergence step。我们报告四个指标：parent-category accuracy、leaf-rule accuracy、object-localization accuracy 和 debug-useful rate。debug-useful 由未参与实现的审阅者根据告警是否足以把工程师引导到正确代码区域进行标注。
```

如果不补量化表：

```tex
\subsection{诊断信息质量}

\paragraph{目的。}
本节不把诊断作为独立分类器来评估，而是检查 \textsc{TrainAudit} 的告警 artifact 是否携带调试所需的信息：触发规则、语义父类、嫌疑对象、首次违规 step 和上下文事件。我们在 \S\ref{subsec:detection} 中已检出的 cases 上审查这些 artifact，并通过两个代表性案例展示它们如何压缩工程师的搜索空间。
```

## 8. 最重要的修改建议

我建议你先不要急着润色句子，而是先做三件事：

1. **在 §6.2 表前补完整 protocol。**  
   让读者知道每个数字如何产生。

2. **把 §6.3 的标题先改掉，除非你能补 diagnosis accuracy CSV。**  
   没有 ground truth 和判分规则，不要叫准确率。

3. **每个 finding 都改成“测量结果 + 解释”，不要直接写设计结论。**  
   比如“规则名有语义”应该变成“在 N 个 detected cases 中，leaf-rule label 与 ground truth 一致 x/N，因此规则名可作为诊断信号”。

## 9. 最短的判断

你说“实验全是靠嘴说出来的”是准确的。现在 6.2/6.3 缺少实验协议和判分标准，所以读者会觉得作者在解释系统，而不是展示实验。按 OSDI/VLDB 风格，先把 `what is measured`、`how it is measured`、`what counts as success` 写清楚，再给表和结论，整节会立刻稳很多。
