# v3 · 中文 Introduction：从 main.tex 直译

> 基于 main.tex 当前 Introduction 的忠实中文翻译，作为后续优化的基底。

---

## 中文 Intro 正文 v3

训练一个大语言模型需要花费数千万美元，并在数千张 GPU 上持续运行数周。其底层软件栈——涵盖用户代码、编译器优化、通信库以及 Megatron-LM 等分布式训练框架 \cite{narayanan2021efficient}——复杂程度极高。这种复杂性使训练任务容易遭受 silent errors \cite{silent_bug_2024,study_guan_2023}——一类本质上难以检测和诊断的故障。

与立即终止执行的崩溃或内存溢出错误不同，silent errors 在正常执行过程中完全不被察觉。它们既不触发异常，也不表现在标准训练指标中——loss 曲线、gradient norm 或 NaN/Inf 检查均无异常。相反，它们使运行中的实现悄悄偏离其预期的算法行为，可能导致模型质量下降或收敛变慢，且难以归因到具体原因。Silent errors 并非罕见的边界情况：它们以多种形式出现在多个生产级框架中——包括遗漏 all-reduce 调用、错误的计数器重置、错误的 gradient normalization、以及通信组配置错误。尽管表现形式多样，这些 bug 共享一个共同的根本原因：每一个都违反了某个 semantic invariant——分布式执行状态必须满足的条件，以忠实实现预期的优化算法。例如数据并行 rank 间的参数一致性，以及正确的梯度累积时序。

[Figure 1] 展示了一个来自 DeepSpeed ZeRO-2 optimizer 的具体示例。`micro_step_id` 计数器中的一个 off-by-one 错误导致 gradient buffer 在本应累积的时刻被重置（"Copy"）——这一违规不触发任何运行时异常，且 loss 曲线在很多 iteration 内几乎不变，却悄悄丢弃了部分预期的梯度更新。

在实践中，silent errors 主要通过高层训练指标来检测——loss 曲线、gradient norm 和 NaN/Inf 检查。这些指标存在两个根本性的局限。第一，它们是滞后的：当异常在 loss 中显现时，训练任务可能已经浪费了数小时甚至数天的集群时间。第二，它们是无信息量的：一个 loss spike 表明出了问题，但不告诉你在哪里或为什么。更根本地，这些局限指向一个共同瓶颈：检测器缺少 correctness oracle——不是缺观测数据，而是缺少判断"在当前并行配置和训练阶段下，观测到的状态关系是否正确"的依据。训练过程可以记录丰富的运行时数据，但这些观测只描述"发生了什么"，不说明"什么应该发生"。

现有工作从不同来源寻找这种依据，但都存在关键盲区。TrainCheck \cite{jiang2025traincheck} 从 healthy executions 的 API 调用 trace 中推断行为模式，在 API 级故障上实现了较强覆盖，但行为模板在结构上无法捕获 topology-aware semantic invariants——这些条件编码在框架设计意图中而非 trace 中。TTrace \cite{ttrace2025} 通过将中间 tensor 与 trusted reference implementation 对齐比较获得更强的语义信号，但要求分布式执行能够与 reference 对齐，且只能 offline 运行。TrainVerify \cite{trainverify2025} 通过证明 distributed execution plan 与 logical specification 等价提供形式化保证，但其范围是 plan-level verification，不覆盖运行时状态转移。

然而绝大多数真实 silent error 恰恰发生在运行时状态更新和通信关系中，这些关系是否正确取决于 topology、rank group 和 training phase——现有方法要么无法编码这类关系，要么无法在训练中持续检查。\textsc{TrainAudit} 正是针对这一互补且尚未被充分解决的 semantic logic faults 类别。

这类 semantic invariants 被显式编码在训练框架的源码和文档中——它们规定了参数如何跨设备同步、梯度如何累积、optimizer states 如何演化——但在数百个源文件和多种并行配置下大规模提取它们，过去没有可行手段。TrainAudit 解决这一问题的核心思路是：将 LLM 作为候选约束的生成器而非 oracle 本身，通过确定性机制将候选 design intent 转化为可执行的运行时检查。系统首先用 multi-agent LLM pipeline 从框架源码和文档中提取 topology-aware 候选约束；随后通过 bidirectional adversarial verification 主动构造反例，过滤幻觉或过度泛化的规则；最后将存活的约束编译为 SQL 查询，在训练过程中以层次化方式 in-flight 执行，仅在违规时触发细粒度检查以定位根因。

**贡献：**

1. 我们提出 \textsc{TrainAudit}，首个直接从框架源码和文档自动挖掘 topology-aware semantic invariants 并在训练中 in-flight 执行的 silent error detection 系统（约 5% 运行时开销）。

2. 我们设计 bidirectional adversarial verification 和 topology-aware pruning 机制，在部署前过滤幻觉约束，并根据并行配置裁剪活跃约束集，消除因 intentional sharding 产生的误报。

3. 我们在 33 个注入 fault 和 15 个真实历史 bug 上评估，\textsc{TrainAudit} 实现 90.9% 检测率（TrainCheck 6.1%），零误报（TrainCheck 在 TP 上 100% 误报），约 5% 开销。
