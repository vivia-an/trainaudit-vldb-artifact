# 08 · 中文 Introduction 草稿：Correctness Oracle 瓶颈与 Design-Intent Contracts

> 本文档目标：先用中文把 Introduction 的论文逻辑打磨顺，再翻译成英文。写法尽量贴近正式论文 intro：问题背景、现有方法局限、关键机会、核心挑战、TrainAudit 方案、贡献总结。文中的 `\citep{...}` 尽量使用 `main.bib` 中已有或本次补充的 citation key。

## 中文 Intro 正文草稿 v1

大规模语言模型训练已经从单机程序演化为复杂的分布式系统工程。一次训练通常需要在数百到数万张 GPU 上连续运行数天甚至数周，并依赖 Megatron-LM、DeepSpeed、ZeRO、GPipe、PyTorch Distributed 等训练框架来组合数据并行、张量并行、流水线并行、优化器状态切分和通信计算重叠等机制 \citep{shoeybi2019megatron,narayanan2021efficient,rasley2020deepspeed,rajbhandari2020zero,huang2019gpipe,li2020pytorch}。生产系统经验表明，随着规模扩大，训练稳定性和可观测性本身已经成为大模型系统的核心问题 \citep{jiang2024megascale}。在这样的软件栈中，最危险的错误往往不是会立即崩溃的错误，而是 silent errors：程序继续运行，loss 曲线和常规健康指标看起来仍然合理，但实际执行已经悄悄偏离了训练算法或分布式实现的设计语义 \citep{silent_bug_2024,study_guan_2023,zang2025towards}。

这类错误在分布式训练框架中并不罕见。已有实证研究发现，大模型分布式训练和推理框架中的 bug 涉及通信组配置、并行策略、状态管理、数值类型和优化器逻辑等多个层面 \citep{zang2025towards}。例如，DeepSpeed ZeRO-2 中的一个真实 bug 会导致 `micro_step_id` 的阶段判断错误，使梯度缓冲区在本应累积时被覆盖；训练不会抛出异常，loss 也可能在多个 iteration 内保持正常，但部分 micro-batch 的梯度更新已经被静默丢弃 \citep{B10}。类似错误的共同特征是：它们不是简单的 NaN、OOM 或 API 调用失败，而是运行时状态和训练框架设计意图之间发生了语义偏离。

这个例子也暴露出一个更一般的问题：silent error 的信号并不一定缺席，它只是没有被解释为错误。对于 `micro_step_id` bug，系统完全可以记录 counter、buffer operation 和 backward/step 阶段；但除非检测器知道在特定 accumulation phase 中这些状态本应满足怎样的关系，这些观测并不会自动变成错误信号。换言之，检测 silent errors 本质上不是一个单纯的 monitoring 问题，而是一个 correctness oracle 问题。训练过程可以产生丰富的观测信号，包括 loss、gradient norm、API trace、communication event、tensor snapshot 和 runtime logs；但这些信号本身只说明"发生了什么"，并不说明"什么应该发生"。一个 rank 上的 tensor 与另一个 rank 不同，可能是合法的 tensor-parallel sharding，也可能是漏掉同步后的错误状态；一次 collective 调用序列看起来完整，也可能发生在错误的 training phase 或错误的 rank group 上。没有 oracle，更多观测只会产生更多未解释的数据。

现有工作可以理解为对这个 oracle 问题的三类近似。第一类是 behavioral oracle：以 Daikon 为代表的动态 invariant mining 从运行 trace 中归纳 likely invariants \citep{ernst2007daikon}；TrainCheck 将这种思想推进到深度学习训练场景，从健康运行的 API-call trace 中学习 proactive checks \citep{jiang2025traincheck}。这类方法的优势是无需人工 specification，且可以在线使用；但它们学习到的是"健康运行中通常出现的行为模式"，不是框架实现背后的设计语义。第二类是 differential oracle：TTrace 通过对比 distributed run 和 reference implementation 的中间 tensor 来定位训练错误 \citep{ttrace2025}。这类方法有明确语义参照，但依赖可信 reference，在 production-scale 训练、复杂并行配置或新框架特性中往往难以获得。第三类是 formal oracle：TrainVerify 验证分布式训练计划与逻辑 specification 的等价性 \citep{trainverify2025}。这类方法保证更强，但前提是人已经写出了可验证的 logical specification，且主要覆盖 plan-level correctness，而不是运行中的状态漂移、实现 bug 或拓扑边界条件。

这些方法并非不足够先进，而是受到同一个结构性瓶颈限制：分布式训练缺少一种既有语义、又可获得、还能在线执行的 correctness oracle。这个瓶颈在 DP/TP/PP 并行下尤其明显。并行化会故意把一个逻辑训练状态拆成多个物理状态：参数可能被切分，梯度可能只是局部和，优化器状态可能按 rank 分片，pipeline buffer 可能跨 micro-batch 滞后。因此，正确执行并不意味着所有物理状态始终相等；相反，正确性取决于这些物理状态在特定 topology、rank group、training phase 和 tensor lineage 下是否仍然对应同一个合法的逻辑训练状态。换言之，分布式训练需要检查的是 topology-conditioned semantic relations，而不是无条件的 tensor equality 或 API sequence。

过去，这类 oracle 很难自动获得，因为它主要存在于框架专家对源码、文档和配置语义的理解中。训练框架的源码和文档确实编码了大量 design intent：哪些 rank 属于同一个通信组，哪些 tensor 是 shard 而不是 replica，哪些梯度必须在 backward 后 all-reduce，optimizer state 应在何时同步，pipeline stage 之间允许哪些暂态不一致。传统静态分析和基于规则的文档抽取已经证明，文档和源码可以为软件测试提供有用的约束信息 \citep{xie2021docter}；近期 LLM invariant synthesis 也表明，大模型能够生成程序验证所需的候选不变量，但这些候选往往需要额外排序、验证或过滤 \citep{chakraborty2023ranking}. 这些进展共同带来一个新的机会：repo-level LLM 是否可以把训练框架中隐含的 design intent 恢复为分布式训练的 runtime correctness oracle？

然而，repo-level LLM 带来的只是新的 oracle 获取路径，而不是一个可以直接部署的 oracle。更准确地说，LLM 能从源码和文档中提出关于 design intent 的候选解释；这些候选解释只有在被证据锚定、补全适用条件、并绑定到运行时可观测状态之后，才可能成为可执行的 correctness contract。否则，LLM 可能生成源码中没有依据的约束，可能遗漏约束成立所需的 topology、phase 或 layer-type 条件，可能把某个局部实现细节错误推广为全局规则，也可能生成无法映射到 runtime schema 的自然语言描述。在分布式训练中，这些问题会被进一步放大：缺少 TP/PP 条件的同步规则会在其他拓扑下产生 false positive；没有 phase guard 的检查会把合法的中间暂态误判为错误；没有 tensor lineage 的 equality check 会混淆 intentional sharding 和真正的不一致。因此，真正的研究问题不是"LLM 能否写出一些检查"，而是：如何把 noisy、conditional、implicit 的 AI-mined design intent 转化成可信、可执行、低开销的 runtime contracts。

本文提出 TrainAudit 来回答这个问题。TrainAudit 从训练框架源码和文档中合成 topology-conditioned runtime contracts。这些 contracts 描述的不是单个 tensor 是否应该等于另一个 tensor，而是在给定并行拓扑、通信组、训练阶段和 tensor lineage 后，分散的物理状态应当满足什么关系，才能对应同一个合法的逻辑训练状态。TrainAudit 首先使用 LLM pipeline 从 framework design artifacts 中提出候选 contracts；随后通过 evidence grounding 和 adversarial counterexample filtering 剔除缺少源码证据、适用条件不完整或过度泛化的规则；最后将通过验证的 contracts 绑定到运行时收集到的 rank group、phase、tensor metadata 和中间状态上，并编译为关系查询进行在线检查。

这一设计将 LLM 放在"语义候选生成器"的位置，而不是把 LLM 本身当作可信 oracle。可靠性来自三个后续步骤：第一，contract 必须被源码或文档证据锚定，避免无根据的自然语言推断；第二，contract 必须显式携带 topology、phase 和 lineage 条件，避免把合法的分布式暂态误判为错误；第三，contract 必须经过对抗式反例过滤，并最终由确定性的 runtime verifier 执行。通过这种方式，TrainAudit 将过去只能由专家手工维护的分布式训练设计语义，转化为可批量生成、可验证、可在线执行的 runtime oracle。

我们在 Megatron-LM/DeepSpeed 风格的分布式训练场景中评估 TrainAudit。实验覆盖 DP、TP、PP 下的 33 个注入 semantic faults，并结合 15 个来自开源 issue/PR 的真实历史 bug。结果显示，TrainAudit 能检测 trace-only 方法难以表达的拓扑条件化语义错误，同时保持约 5% 的运行时开销；在 clean runs 上，我们观察到零误报。更重要的是，实验结果支持本文的核心论点：silent error detection 的关键瓶颈不是缺少更多观测，而是缺少可获得、可执行的 correctness oracle；repo-level LLM 使得从 framework design artifacts 中恢复这种 oracle 成为可能，而 TrainAudit 展示了将这一可能性转化为可靠系统所需的机制。

本文作出以下贡献：

1. 我们识别并形式化了分布式 LLM 训练 silent error detection 中的 correctness-oracle bottleneck，指出现有 trace-based、reference-based 和 formal-spec-based 方法分别在语义强度、可获得性和在线部署性之间做出取舍。
2. 我们提出 topology-conditioned runtime contract 作为分布式训练 silent error detection 的核心抽象，用于描述物理分布状态和逻辑训练状态之间在特定 topology、phase 和 lineage 下应满足的语义关系。
3. 我们设计并实现 TrainAudit，一个从 framework source code 和 documentation 中合成、验证并执行 runtime contracts 的系统，包括 design-intent mining、evidence grounding、adversarial counterexample filtering、schema binding 和 topology-aware verification。
4. 我们通过注入 faults、真实历史 bug、clean-run false positive 测试和 ablation 实验证明：基于 design intent 的 oracle 能覆盖 trace-only 方法结构性难以表达的 silent semantic errors，并以较低运行时开销在线部署。

## 段落功能说明

| 段落 | 功能 | 对应英文 intro 目标 |
|---|---|---|
| 1 | 大背景：LLM 训练规模、分布式软件栈复杂性、silent error 风险 | Motivation |
| 2 | 具体化问题：真实 bug + silent semantic deviation | Concrete example |
| 3 | 抽象问题：monitoring 不够，本质是 correctness oracle | Problem reframing |
| 4 | 现有工作分类：behavioral / differential / formal oracle | Prior work gap |
| 5 | 分布式训练为什么更难：topology-conditioned relations | Domain-specific challenge |
| 6 | 为什么现在可以：源码/文档中有 latent design intent，repo-level LLM 带来转机 | New opportunity |
| 7 | 为什么 LLM 还不够：hallucination、条件缺失、不可执行 | Technical challenge |
| 8 | TrainAudit 的定义：topology-conditioned runtime contracts | Approach |
| 9 | 系统可靠性机制：evidence、topology、adversarial filtering、deterministic verifier | Method rationale |
| 10 | 结果和核心论点回扣 | Results + thesis |
| 11 | contributions | Contribution list |

## 引用策略

Intro 中建议只放必要引用，避免文献综述过重：

- 分布式训练框架背景：`\citep{shoeybi2019megatron,narayanan2021efficient,rasley2020deepspeed,rajbhandari2020zero,huang2019gpipe,li2020pytorch}`
- 大规模训练稳定性动机：`\citep{jiang2024megascale}`
- silent bug / framework bug 实证：`\citep{silent_bug_2024,study_guan_2023,zang2025towards}`
- 当前直接相关方法：`\citep{jiang2025traincheck,ttrace2025,trainverify2025}`
- 动态 invariant mining 背景：`\citep{ernst2007daikon}`
- 文档/源码约束提取与 LLM invariant 先例：`\citep{xie2021docter,chakraborty2023ranking}`

## 需要进一步核实或补强的实验锚点

这版 intro 的核心 claim 需要下面这些结果支撑，否则容易显得像 positioning：

1. **Trace-only 表达力有限**：至少给出 1-2 个 bug，证明正确运行和错误运行的 API trace 相同，但状态语义已破坏。
2. **Design artifacts 有额外信息**：做 `source+docs`、`source-only`、`docs-only`、`trace-only` 的 mining ablation。
3. **LLM 产物不可信**：统计 raw contracts 中 hallucination、missing condition、unbound schema 的比例。
4. **Adversarial filtering 必要**：报告 filtering 前后的 precision、FPR、recall。
5. **Topology condition 必要**：展示 topology-unaware verifier 在 TP/PP 场景中的误报或漏报。
6. **Contract 类型分布**：统计 collective protocol、sharding lineage、optimizer state、pipeline schedule、mixed precision 等类别，证明不是手写少数 case。

## 可翻译成英文的核心句

> Existing systems either observe executions without knowing their intended semantics, or require explicit specifications that practitioners rarely have. TrainAudit exploits a third source of correctness oracle: the latent design intent already encoded in training framework source code and documentation.

> Distributed training correctness is not unconditional tensor equality. It is a topology-conditioned correspondence between fragmented physical states and a single logical training state.

> Repo-level LLMs make latent design-intent extraction feasible, but they do not make it reliable. TrainAudit bridges this gap by grounding, adversarially filtering, and compiling AI-mined intent into executable runtime contracts.
