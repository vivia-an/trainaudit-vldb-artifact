# 论文叙事重构 · 工作文件夹

本文件夹用于存放 TrainAudit 论文的"故事级"重写思路与中文大纲。当前论文（`main.tex`）的写法停留在"我们做了一个系统、它检测率很高"的工程报告层级，这里的目标是把整篇 paper 的叙事抬到"AI 改写了一个 30 年的瓶颈"的范式层级。

## 文件清单

| 文件 | 用途 | 优化阶段 |
|---|---|---|
| `01_核心论点.md` | 一句话命题 + 锚定故事的实证依据（rule deficit 的可量化证据） | 第 1 步：先把核心 claim 钉死 |
| `02_故事A_规则赤字.md` | 主推叙事：所有现有方法都是 rule deficit 下的妥协；AI 第一次直面瓶颈 | 第 2 步：作为 paper 主骨架 |
| `03_故事B_问错了问题.md` | 钩子叙事：过去十年都在问"是否健康"，应该问"是否忠于算法" | 第 3 步：用作 abstract / intro 第一句 |
| `04_故事C_debugging前移.md` | 历史叙事：debugging 不断前移，AI 让 spec 成本与系统复杂度脱钩 | 第 4 步：用作 §5 末段展望 |
| `05_论文大纲_中文版.md` | 在新叙事下重组后的章节大纲与每段核心论点 | 第 5 步：把故事翻译成 section-level 结构 |
| `06_待办与开放问题.md` | 故事要落地需要的实验、数字、引用核查 | 持续迭代 |
| `07_全流程思考.md` | 脱开 positioning 层，从数据多样性到 invariant 检验未知数据再到延续的完整 pipeline 工作机理 | **当前最重要的文件**——下一轮写作的核心依据 |
| `intro_drafts/` | Introduction 中文草稿版本目录，当前 v1 主线是 oracle 瓶颈、LLM 转机、TrainAudit 的 topology-conditioned runtime contracts | 用于持续迭代 v2 / v3 并最终翻译成英文 |

## 推荐迭代顺序

1. **先读 `07_全流程思考.md`**——它是脱开 positioning 之后真正落到工程现实的版本，也是后续所有改写的依据
2. 在 07 的基础上回头检查 `01_核心论点.md` 的那一句话有没有需要调整
3. `02_故事A_规则赤字.md` 是把 07 的工作机理升级到 paper 叙事高度的桥梁
4. 故事 B / C 是 A 的辅助武器，可以晚一点定
5. `05_论文大纲_中文版.md` 应该按 07 末尾给出的"映射回 paper structure"重写，特别是补 schema binding 小节和延续章节

## 设计原则

- **避免空洞：** 每一个范式级 claim 都要有一个可量化的实证锚点。"Rule deficit 是真的"——数字呢？"AI 改变了 economics"——成本数字呢？任何只能修辞、不能用数字落地的故事都会被审稿人撕掉。
- **不要承诺过头：** "first to apply LLM" 这种 claim 是 fragile 的；"first to formalize and instantiate the rule deficit lifting" 才是站得住的。
- **要让审稿人第一遍读完记得住一句话。** 现在记得住的是 "TrainAudit 90.9%"——这是工程数字。重写之后应该让人记得住 "rule deficit" 或 "intent-anchored invariants" 这种概念词。
