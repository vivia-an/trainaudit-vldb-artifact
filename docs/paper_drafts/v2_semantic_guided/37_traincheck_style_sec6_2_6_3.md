# 37. 参考 TrainCheck 后的 §6.2 / §6.3 短写法

> 日期：2026-05-17  
> 参照对象：TrainCheck OSDI'25 §5 Evaluation，尤其 §5.1 Silent Error Detection、§5.2 New Silent Errors、§5.3 False Positive。  
> 目标：不要把正文写成实验 runbook；只补足必要的实验设计，使结果不像“靠嘴说出来”。

## 1. TrainCheck 的 evaluation 是怎么写的

TrainCheck 的结构很克制：

1. **Evaluation 开头列问题 + 硬件。**  
   它先说评估回答五个问题：能否检测真实 silent errors、检测多快、能否帮助诊断、是否准确、开销多少。然后用一小段交代硬件和软件栈。

2. **Detection 小节先给实验对象。**  
   它不是直接上结果，而是先说：收集并复现 20 个真实 silent errors，其中 6 个来自 prior study，14 个来自 GitHub/StackOverflow/social media，覆盖不同 root cause。

3. **Baseline 写成短列表，但参数写清。**  
   Spike detector、Trend detector、Anomaly detector、PyTea/NeuRI 各用一句话定义，并给出关键阈值或默认参数，避免“随便设 baseline”。

4. **Methodology 是一段，不繁琐。**  
   每个 error 准备 reproduction script，运行后产生两类输出：runtime trace 给 invariant/checker，loss/accuracy/gradient norm 给 signal detectors。然后跑 detectors 收集结果。

5. **判定标准用一句话钉住。**  
   它强调只统计 true detections；为了避免奖励乱报警的 detector，还跑 fixed versions，看 detector 是否也在 error-free traces 上报警。

6. **Diagnosis 不单独吹成大实验。**  
   它说 diagnosis 不是 primary goal，只分析 violation reports 是否能帮助 debugging，然后给 10 exact / 8 close-to-root-cause 这种非常直接的统计。

这个风格对你最有用：**短，但每个结果都有 setup 和判定标准。**

## 2. 你的 §6.2 应该借 TrainCheck 的哪几件事

你不用写很长，只需要在表前补四个短段：

```tex
\paragraph{实验对象。}
...

\paragraph{基线。}
...

\paragraph{方法。}
...

\paragraph{判定标准。}
...
```

注意它们每段 3--5 行即可。不要变成 runbook。

### 推荐替换草稿：§6.2 表前

```tex
\paragraph{实验对象。}
我们使用 Real-SDC-N 作为主检测集合。每个 case 都对应一个真实上游 issue、PR 或 commit pair，并包含一个可重放的 buggy run、一个 fixed run，以及人工审计得到的 ground-truth fault pattern。Real-SDC-N 覆盖 Megatron-LM、DeepSpeed、OLMo 和 OLMo-core 中的分布式状态同步、通信 dtype、checkpoint、optimizer state、data loading 等静默错误。当前 trace schema 下没有稳定运行时信号的真实案例单独标为 boundary，不进入 detector-coverable 分母。

\paragraph{基线。}
我们比较两个基线。Naïve Monitoring 代表常见训练监控，检查 loss/gradient 的 NaN/Inf、loss spike 和 gradient-norm spike。TrainCheck~\cite{jiang2025traincheck} 是最接近的系统，它从 clean trace 中推断 API/tensor 行为不变量；我们按其工作流为每个 case 提供 clean reference trace，并在 held-out fixed run 上单独测量误报。

\paragraph{方法。}
对每个 case，我们运行三种 trace：reference-fixed run、buggy run 和 held-out fixed run。reference-fixed run 仅用于 TrainCheck 推断不变量；所有方法都在 buggy run 上测检出，在 held-out fixed run 上测误报。\textsc{TrainAudit} 的规则库在本实验前由 pattern catalog 和目标框架源码生成，不使用该集合的 buggy/fixed traces 进行调参或筛选。

\paragraph{判定标准。}
若工具在 buggy run 中产生与该 case ground-truth fault pattern 匹配的告警，则记为 detected；运行完成但无匹配告警记为 miss；工具或 harness 崩溃记为 fail，fail 不计入 detected；若工具在 held-out fixed run 上产生任何告警，则记为 false positive。我们同时报告 detector-coverable cases 和 boundary cases，以避免把当前不可观测信号误写成规则缺陷。
```

这四段加起来并不长，但实验味道会立刻出来。

## 3. §6.2 结果段也要像 TrainCheck 一样“结果 + 失败解释”

TrainCheck 的 detection 结果不是只说“18/20”，而是立刻解释：

- 所有 detected cases 在 root cause 触发后一轮内发现；
- 两个 missed cases 为什么超出当前观察面；
- signal baselines 为什么只检测到少数极端 case。

你的对应写法可以是：

```tex
\paragraph{结果。}
表~\ref{tab:summary_3way} 和表~\ref{tab:detection-results} 给出结果。\textsc{TrainAudit} 在 detector-coverable cases 上触发了匹配的语义不变量，并在 held-out fixed runs 上保持静默。相比之下，TrainCheck 只在故障直接表现为其 API/tensor 行为模板变化时触发；Naïve Monitoring 只观察 outcome-level 标量，因此漏掉 loss 和 gradient norm 仍处于正常范围内的语义错误。

\paragraph{漏检与边界。}
未检出的 boundary cases 不包含稳定的在线 trace 信号：一个表现为亚百分比数值漂移，另一个表现为统计噪声地板以下的数据顺序偏差。这些案例需要扩展 observation surface 或离线 reference oracle，而不是简单增加规则。
```

## 4. §6.3 应该借 TrainCheck 的“降调”写法

TrainCheck 没把 diagnosis 写成一个很重的实验。它明确说 diagnosis 不是 primary goal，然后报告 violation reports 对 debugging 的帮助程度。

你的 §6.3 如果暂时没有 diagnosis CSV，也应该这样降调：

```tex
\subsection{诊断信息质量}
\label{subsec:diagnosis}

\paragraph{目的。}
诊断不是一个独立分类器任务；我们评估的是告警 artifact 是否能把工程师引到正确的调试区域。对 \S\ref{subsec:detection} 中被 \textsc{TrainAudit} 检出的 cases，我们检查每个告警包含的触发规则、taxonomy 父类、嫌疑对象、首次违规 step 和上下文 trace。

\paragraph{判定标准。}
若告警直接命中 ground-truth root-cause object 或对应源码路径，我们记为 exact；若告警定位到正确的语义子系统、参数组、rank 或 hookpoint，但仍需要人工沿调用链追一步，我们记为 near-root-cause。未检测的 boundary cases 不进入诊断分母。

\paragraph{结果。}
在 detected cases 中，\textsc{TrainAudit} 的告警均提供了 leaf-level rule label 和 taxonomy 父类；这些字段足以把错误分派到对应的训练子系统。复杂传播型错误需要进一步使用 C1 expander 和 C2 RCA agent；下面两个 case study 展示多条告警如何共同收敛到根因。
```

如果你能补 `diagnosis_accuracy.csv`，就把第三段改成具体数字：

```tex
在 N 个 detected cases 中，parent category 命中 X/N，leaf rule 命中 Y/N；其中 A 个 exact，B 个 near-root-cause。
```

但如果没有这个表，千万不要叫“诊断准确率”。

## 5. 一个更短的最终结构

我建议 6.2 / 6.3 最终只保留下面的骨架：

```text
6.2 检测有效性
  实验对象         1 段
  基线             1 段
  方法与判定标准   1 段
  表 1 aggregate
  表 2 per-case
  结果             1 段
  漏检/边界        1 段
  新 bug           1 段 + 图

6.3 诊断信息质量
  目的与判定       1 段
  结果             1 段或小表
  case study       1--2 个
  与 TrainCheck 对比 1 段
```

这就够了。不要再加“发现 1/2/3/4”那种列表式论证，容易像 proposal 或技术报告。系统论文更偏爱短 protocol + 表 + focused analysis。

## 6. 最关键的风格差异

TrainCheck 给你的启发不是“写更多实验设置”，而是：

```text
每个实验小节只回答一个问题；
表前用一小段说明数据怎么来、工具怎么跑、怎么算成功；
表后只解释主要结果和失败边界。
```

你现在的 6.2/6.3 要改的就是这个顺序：先 measurement contract，再 result，再 interpretation。
