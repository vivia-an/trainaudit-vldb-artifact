# 04 · 故事 C：Debugging 不断前移（AI 让 spec 成本与系统复杂度脱钩）

> **历史叙事，作为 §5 末段展望或 Conclusion 的高调收尾。最大胆，也最容易被批评 over-selling，谨慎使用。**

## 故事的核心 motif

**软件 debugging 的历史，是一部"人类把 invariant 不断往源头前移"的历史。**

每一次前移都收缩了 silent error 的窗口，但每一次都依赖人类把"应当满足的性质"显式写下来。系统越大，spec 越难写——这是过去几十年所有 verification 工作共同的天花板。

**AI 出现之前，spec 的成本曲线随系统复杂度爆炸式增长。AI 之后，这条曲线第一次和系统复杂度脱钩。**

## Debugging 的四个时代

| 时代 | 检查时机 | 代表技术 | 谁来写 spec | 复杂度上限 |
|---|---|---|---|---|
| 1. 事后 | 出错之后 | core dump、log analysis | 不需要 spec，反推根因 | — |
| 2. 运行时 | 出错当下 | assertion、type check、bounds check | 程序员手写 | 一个函数级 |
| 3. 编译时 | 编译期 | static analyzer、type system、symbolic execution | 程序员/类型设计者 | 一个模块级 |
| 4. 设计时 | 写代码前/写代码时 | formal verification、model checker、refinement type | 验证专家 | 一个组件级，且成本极高 |

每一次前移都把检查窗口提前——但 spec 的成本也跟着指数级增长。**Megatron-LM 这种数十万行、跨 5 种并行配置的框架，正好处在"spec 成本爆炸到没人愿意写"的位置——所以它没有任何 design-time 的形式化保障，运行时的 assertion 也只覆盖最显然的输入校验。**

这就是为什么 distributed training 的 silent error 一直是一个 "悬而未决" 的问题：**前四个时代的方法都到不了它的复杂度**。

## AI 打开了第五个时代：Source-anchored runtime verification

我们提出，正在诞生中的第五个时代具有以下结构性特征：

1. **Spec 的来源是 AI 从源代码与文档中提取的**——不需要程序员单独写。
2. **检查在 runtime 进行**——不依赖编译期信息或 design-time 形式化模型。
3. **检查粒度由 invariant 自身决定**——可以从一行代码的 boundary check 一直到跨节点的关系性 invariant。
4. **AI 的能力上限随基础模型同步抬升**——从而 spec 覆盖率随模型迭代而扩张，而不是停留在某个固定的人类成本预算内。

**TrainAudit 是这个第五时代的第一个 instantiation。**

## §5 末段（展望）样板文本

> **A new mode of debugging.** TrainAudit instantiates a debugging mode that did not exist before code-comprehending LLMs were viable. Classical debugging modes—postmortem, runtime assertion, static analysis, formal verification—each push the moment of detection earlier in the software lifecycle, but each also requires a human author to write down what should hold. As systems grow, this authoring cost has historically grown faster than the value of the verification it enables, leaving large complex systems—including modern distributed training frameworks—essentially unverified at runtime. By delegating the authoring step to LLMs operating directly on framework source code, TrainAudit decouples specification cost from system complexity for the first time. We expect future runtime verification systems for complex software—not only training frameworks but database engines, distributed file systems, consensus protocols—to inherit this overall shape: AI-mined specifications, adversarially filtered, executed as relational queries against runtime state.

## Conclusion 样板（如果论文有 Conclusion 章节）

> The bottleneck in detecting silent errors in distributed training has, for thirty years, been the cost of authoring rules at a granularity that matches the system's complexity. Loss curves dodged the bottleneck by collapsing state to a scalar; trace mining dodged it by inferring rules statistically; differential testing dodged it by paying for a parallel oracle; formal verification dodged it by restricting itself to static plans. None lifted the bottleneck itself. TrainAudit is the first system to demonstrate that, with the maturation of LLM-based source code comprehension, the bottleneck no longer needs to be dodged. The rule deficit becomes a workload, the right semantic questions become askable, and silent errors—which have always announced themselves in the data—can finally be heard.

## 这个故事的能量与风险

### 能量
- **格调最高**：把 paper 嵌入计算机科学叙事的主干（debugging 的演化），让审稿人感到这不是一篇普通系统论文。
- **铺路 follow-up**：明确说"我们是这个新时代的第一个 instantiation"——给后续工作留位置，也给自己留学术影响力。
- **独立于具体数字**：即使 90.9% 的检测率被审稿人质疑，故事本身的能量不减——因为故事是关于 *方向* 的，不是关于 *数字* 的。

### 风险
- **最容易被批评 over-selling**："第五个时代"、"new mode of debugging" 这种话审稿人会立刻警惕。
- **若没有故事 A、B 的实证锚点支撑，会被当成 marketing**。一定要先把 rule deficit 的数字证据钉死，再敢这样讲。
- **对 reviewer 1（最严格）很危险，对 reviewer 2、3 很有吸引力**——是高方差的话术。

## 使用建议

### 推荐
- 仅在 **§5 末段** 和 **Conclusion**（如有）使用故事 C 的全部强度
- Abstract / Introduction 主体使用 A + B；不在前半段触碰故事 C 的"四个时代"框架

### 不推荐
- 把"第五个时代"塞进 Abstract（直接劝退保守审稿人）
- 在 Contribution bullet 里使用"a new mode of debugging"措辞（写成 contribution 就必须用证据支撑，太重了）

### 折中
- 在 Conclusion 用故事 C，但把 "first instantiation" 改为 **"an early instantiation"** 或 **"one instantiation of an emerging pattern"**——降低 claim 强度，保留方向感。

## 一段总结

故事 C 是一把"双刃剑"：用得好让 paper 的格调跃升一个 tier；用不好让 paper 显得在 oversell。它适合放在审稿人已经基本接受了 A + B 之后的位置（即 §5 末段或 Conclusion），作为最后一击的"远景陈述"，而不是作为 paper 的主张。它的存在让 reviewer 在合上论文之后会想起这篇文章，但它不应该是 reviewer 第一眼读到的东西。
