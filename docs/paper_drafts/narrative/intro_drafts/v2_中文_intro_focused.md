# v2 · 中文 Introduction 草稿：聚焦 Semantic Oracle

> v2 的写作目标：保留 v1 中较顺的逻辑，但降低信息密度。参考 TrainCheck、TTrace、TrainVerify 和 MegaScale 的 intro/abstract 写法，先把论文的核心缺口讲清楚，再用少量细节支撑，不在 Introduction 中提前展开所有系统机制。

## 写作取舍

v1 的优点是逻辑完整，但信息层次太满：分布式训练背景、真实 bug、oracle 分类、DP/TP/PP 语义、LLM 机会、LLM 缺陷、系统机制都展开得比较细。v2 只保留一个主轴：

> 大规模训练需要 runtime correctness oracle；现有 oracle 要么可获得但语义弱，要么语义强但不可获得；repo-level LLM 让源码/文档中的 design intent 成为第三种 oracle 来源；TrainAudit 的贡献是把这种不可靠的候选 intent 转化成可执行的 topology-conditioned runtime contracts。

## 中文 Intro 正文草稿 v2

大规模语言模型训练已经成为一种长期运行的分布式系统工作负载。为了在数百到数万张 GPU 上训练千亿级模型，现代训练栈需要组合数据并行、张量并行、流水线并行、优化器状态切分、混合精度和通信计算重叠等机制 \citep{shoeybi2019megatron,narayanan2021efficient,rasley2020deepspeed,rajbhandari2020zero,huang2019gpipe,li2020pytorch}。这类系统的规模和复杂性使稳定性成为生产训练中的核心问题：MegaScale 等工业经验表明，许多训练稳定性问题只有在大规模运行时才会暴露，因而需要深入的软件栈可观测性和诊断能力 \citep{jiang2024megascale}。

在这些故障中，最难处理的是 silent errors。它们不会触发异常，也不一定立刻表现为 NaN、loss spike 或显著的 gradient norm 异常；训练任务可以持续运行，产出的模型却已经偏离了预期算法语义 \citep{silent_bug_2024,study_guan_2023,zang2025towards}。一个最简单的例子来自数据并行训练：不同 GPU 副本在本地 backward 之后拥有不同的局部梯度，这是正常现象；但在梯度同步和 optimizer step 之后，这些副本的参数应当重新保持一致。如果某个通信或状态更新错误使一个 rank 使用了错误的梯度，程序未必崩溃，loss 也可能暂时正常，但训练已经偏离了数据并行算法的语义。这个例子说明，silent error 的关键信号往往不是某个单独指标异常，而是运行时状态之间的关系在特定阶段不再成立。

从这个角度看，silent error detection 的困难不只是缺少更细粒度的监控，而是缺少能够解释监控信号的 correctness oracle。这里的 oracle 借用的是软件测试中的 test oracle 概念：给定一次程序执行，检测器需要某种依据来判断观察到的行为是否符合期望 \citep{barr2015oracle}。在本文中，这个依据通常来自某种 specification，并最终落成可执行的 invariant 或 contract。也就是说，oracle 回答的不是"系统记录到了什么"，而是"在当前训练阶段和并行配置下，这些状态本应满足什么关系"。训练过程可以记录 loss、API trace、communication events、tensor states 和 runtime metadata，但这些观测本身只描述"发生了什么"，并不说明"什么应该发生"。在分布式训练中，这一点尤其关键：同名参数在 data-parallel replica 间应在 optimizer step 后一致，而在 tensor/pipeline/ZeRO sharding 维度上只应满足切分、聚合或重构关系。没有这样的 oracle，更多 tracing 往往只是增加待解释的数据量，而不必然提升检测器判断 correctness 的能力。

现有技术各自选择了不同的正确性参照，但在真实 silent error 面前都存在关键盲区。基于结果指标的监控依赖 loss、gradient norm 等训练信号，成本低，但粗粒度且发现滞后。TrainCheck 通过从 healthy executions 中推断 invariants 降低了规则编写成本，但其检查能力受限于推断阶段覆盖的执行、采集到的信号，以及 invariant templates 能表达的关系 \citep{jiang2025traincheck}。TTrace 通过将中间 tensor 与 trusted reference implementation 对齐比较获得更强的差分信号，但要求分布式执行能够与 reference 对齐，且只能 offline 运行 \citep{ttrace2025}。TrainVerify 通过证明 distributed execution plan 与 logical specification 等价提供形式化保证，但其范围是 plan-level verification，而不是对运行时状态转移的持续检查 \citep{trainverify2025}。然而绝大多数真实 silent error 恰恰发生在运行时状态更新和通信关系中，这些关系是否正确取决于 topology、rank group 和 training phase——现有方法要么无法编码这类关系，要么无法在训练中持续检查。

TrainAudit 正是为检测这类错误而设计。训练框架的源码和文档精确记录了哪些状态应被复制、切分或同步，以及在什么拓扑条件下这些关系应当成立。TrainAudit 用 LLM 从这些 artifacts 中挖掘候选约束，经证据锚定和对抗过滤后，将其具体化为 topology-conditioned runtime contracts，在训练过程中持续在线检查。

基于这一观察，我们提出 TrainAudit。TrainAudit 从训练框架源码和文档中合成 topology-conditioned runtime contracts，用于检查分散的物理训练状态是否仍然符合框架设计意图。这些 contracts 不再是无条件的 tensor equality，也不是单纯的 API sequence pattern；它们描述的是在给定 rank group、parallel topology、training phase 和 tensor lineage 后，不同 rank、不同阶段、不同状态之间应满足的语义关系。

TrainAudit 的设计将 LLM 限定为语义候选生成器，并通过确定性机制把候选 intent 变成可执行 oracle。系统首先从 framework design artifacts 中生成候选 contracts；随后通过 evidence grounding 和 adversarial counterexample filtering 过滤缺少依据或过度泛化的规则；最后将保留下来的 contracts 绑定到运行时收集的 metadata 和 tensor states，并编译为关系查询进行在线检查。这样，TrainAudit 利用了 LLM 带来的 design-intent extraction 能力，同时避免把 LLM 输出本身当作可信 specification。

我们在 Megatron-LM/DeepSpeed 风格的分布式训练场景中评估 TrainAudit，覆盖 DP、TP、PP 下的 33 个注入 semantic faults，并结合 15 个来自开源 issue/PR 的真实历史 bug。实验表明，TrainAudit 能检测 trace-only 方法难以表达的拓扑条件化语义错误，在 clean runs 上观察到零误报，并保持约 5% 的运行时开销。结果支持本文的核心论点：大规模训练 silent error detection 的关键不是继续增加观测量，而是获得一种可执行的 semantic oracle；repo-level LLM 让从 framework design artifacts 中恢复这种 oracle 成为可能，而 TrainAudit 展示了使其可靠部署所需的系统机制。

本文作出以下贡献：

1. 我们提出 distributed LLM training silent error detection 中的 correctness-oracle bottleneck，并系统化比较 trace-based、reference-based 和 formal-spec-based oracle 的能力边界。
2. 我们提出 topology-conditioned runtime contract，用于把训练框架源码和文档中的 latent design intent 转化为可在线检查的语义关系。
3. 我们设计并实现 TrainAudit，一个将 LLM-mined design intent 经过 evidence grounding、adversarial filtering 和 runtime schema binding 后执行的 silent error detection 系统。
4. 我们通过注入错误、真实历史 bug、clean-run false positive 测试和 ablation 实验证明，TrainAudit 能覆盖 trace-only 方法难以表达的 silent semantic errors，并以较低开销在线运行。

## v2 相比 v1 的主要变化

1. **主 claim 更集中**：所有段落都围绕 "semantic oracle" 展开，不再并列展开过多系统细节。
2. **现有工作更压缩**：TrainCheck / TTrace / TrainVerify 只用一段按 oracle source 对比，避免相关工作提前膨胀。
3. **DP/TP/PP 细节降级**：保留 topology-conditioned 的必要性，但不展开 sharding、partial gradient、pipeline buffer 的完整列表。
4. **LLM 部分更克制**：不强调 "LLM 降低成本" 这种泛泛说法，而是说它提供 design-intent hypotheses，需要系统验证。
5. **TrainAudit 机制只讲三件事**：候选生成、证据/反例过滤、runtime binding/checking。其余细节留给 system section。
