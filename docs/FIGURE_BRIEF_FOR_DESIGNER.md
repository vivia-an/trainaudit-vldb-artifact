# TrainAudit 论文图设计 Brief（给画图同学）

> 目的：帮你在不读完论文的前提下，准确理解 TrainAudit 系统**做什么**、**怎么做**、**有哪些可视化要点**，从而画出与论文 narrative 对齐的图。
> 相关文件：`main.tex`（VLDB 投稿），`PAPER_STORY.md`（中文 story）。

---

## 1. 一句话定位

**TrainAudit** = 一个**在线检测分布式 LLM 训练中 silent error（静默语义错误）的系统**。
核心做法：**离线**用 multi-agent LLM pipeline 从训练框架（Megatron-LM / DeepSpeed / OLMo / OLMo-core / FSDP）的源码中挖掘"语义不变量（semantic invariant）"，**在线**把这些不变量编译成 SQL 查询 / runtime hook / 静态断言，每个训练 step 都在低开销（<5%）下检查。

> **关键定位**：现有方案（TrainCheck、TTrace、TrainVerify）只能从"执行行为"找参照系；TrainAudit **第一次让系统从源码读出"应该如何"**。

---

## 2. 关键术语词汇表（画图时直接使用）

| 术语 | 含义 | 一句话解释 |
|---|---|---|
| **Silent Error** | 静默语义错误 | 训练没崩、loss 看起来正常，但语义被悄悄破坏 |
| **Invariant**（不变量）| 训练状态必须满足的谓词 $p(s)$ | 例："所有 DP rank 上的 weight 必须相等" |
| **Pattern Catalog**（模式库）| 8 个抽象 detection pattern | 从 128 个真实 bug 抽出来的"通用错误模式" |
| **Three-Layer Decomposition** | $p(s) = \pi_{\text{schema}} \wedge \pi_{\text{topo}} \wedge \pi_{\text{precond}}$ | 任何 invariant 可拆成 三层 |
| **Topology / Parallelism** | DP/TP/PP/EP | 数据并行 / 张量并行 / 流水线并行 / 专家并行 |
| **Hookpoint**（钩子点）| 5 个训练循环切入点 | before-forward / after-forward / main-grad-in-backward / after-backward / before-optimizer |
| **Trace Database** | DuckDB 滑窗存储 | 每 step 存 ~14MB；保留两步窗口 = ~28MB |
| **T0 / T1 rule** | 规则的层级 | T0 = PyTorch 根级（跨框架通用）；T1 = 框架级（依赖框架元数据） |
| **Type A / B / C** | 规则的执行模式 | A = SQL 查询；B = runtime hook；C = init 时静态断言 |
| **D1 benchmark** | 17 bug × {buggy, fixed} = 34 脚本 | 论文的 gold-standard 检测对比集 |

---

## 3. 系统三层架构（画图必懂）

```
                        ┌──────────────────────────────────────┐
                        │     OFFLINE  ❶ Invariant Miner        │  一次/框架
                        │   (multi-agent LLM pipeline + 对抗验证)│  ~$70-100
                        │                                        │  ~8 小时
                        │  输入: 框架源码 + 文档 + Pattern catalog │
                        │  输出: 25 条已验证的 invariant 规则     │
                        └─────────────────┬──────────────────────┘
                                          │ 持久化到 Invariant Library
                                          ▼
       Training Job ─────►  ❷ Data Collector  ─────►  ❸ Verifier
       (用户代码无修改)      (5个hookpoint抓状态)        (拿规则查 trace)
                            写入 DuckDB                每步检查
                                                       检测 → 诊断 → 报告
```

### ❶ Invariant Miner（离线，一次性）

**输入**:
- 框架源码（Megatron-LM 等）
- 设计文档
- 8 个 detection pattern 的 catalog（P1~P8）
- 训练启动脚本（用于知道并行拓扑）

**内部 5 个 LLM agent，按 FSM 流转**:
1. **S1 Gap Analysis Agent**：选下一个待实例化的 pattern
2. **S2 Evidence Agent**：从源码/文档收集相关上下文（类继承、初始化代码、文档说明）
3. **S3 Synthesis Agent**：生成候选 invariant + **对抗 counterexample 验证**（核心反幻觉机制）
4. **S4 Persistence Agent**：把通过的 invariant 编码为 Type A/B/C 三种执行形式之一
5. **S5 Reporting Agent**：日志 + 路由下一轮

**S3 内部的四步漏斗**（论文的核心反幻觉机制）：
```
candidate
   │
   ├── (i) Hypothesis Scoping       缩小 invariant 适用范围
   ├── (ii) Evidence Grounding      检索支持/反对的证据
   ├── (iii) Counterexample Construction  主动构造反例攻击自己
   └── (iv) Acceptance              没反例 + Conf ≥ 0.8 → 接受
```

公式：
- Accept: $\forall ce \in \mathcal{CE}: \text{Verify}(ce) \neq \text{HOLDS} \;\land\; \text{Conf}(c) \geq \theta$
- Conf：$\text{Conf}(c) = \alpha \cdot \frac{|\mathcal{E}^+|}{|\mathcal{E}^+|+|\mathcal{E}^-|} + (1-\alpha) \cdot \frac{|\mathcal{CE}_r|}{|\mathcal{CE}|}$，$\alpha = 0.4$，$\theta = 0.8$

**输出**: 25 条 active rules（Megatron-LM 部署：12 T0 + 9 T1 + 3 active probes）+ 13 dsl_native 模板。

### ❷ Data Collector（在线，每 step）

**功能**：用 PyTorch hook 注入到训练循环 5 个标准点，抓取最少必要的状态字段写入 DuckDB。

**5 个 hookpoint**:
1. `before-forward`
2. `after-forward`（覆盖 128 bug 中的 48 个，是其他工具最常忽略的）
3. `main-grad-in-backward`
4. `after-backward`
5. `before-optimizer`

**Tiered Trace Schema（核心 design 卖点之一）**：
| Tier | 增量字段 | 累计 bug 覆盖 |
|---|---|---|
| 0 | cksum, param_name, ranks, step, stage, dtype, shape | 30% |
| 1 | grad_norm, grad_cksum | 41% |
| 2 | loss_value | 51% |
| 3 | learning_rate, micro_step_id | 59% |
| 4 | optimizer_state_cksum | 61% |
| 5–6 | ep/cp_rank, zero_stage, has_nan/inf, num_tokens | 74% |

**关键观察**（图 `tier_coverage.pdf` 反映的）：
T0 → T1 的过渡同时**提升覆盖率**（30%→41%）**且降低开销**（~8% → ~1.5%），因为用 GPU-side scalar reduction（`param.norm()`）替代了全 tensor checksum 的 GPU→CPU 拷贝。

**存储**：DuckDB 滑窗，每 step ~72k 行 ~14 MB；保留两步窗口 = ~28MB；软件 silent error 是确定性的，每步都会触发，因此两步够用。

### ❸ Verifier（在线，每 step）

**Phase 1 — Job Launch Setup**（启动一次）:
- **Topology extraction**：从启动脚本读 DP/TP/PP/EP/zero_stage/dtype
- **Rule pruning**：去掉 $\pi_\text{topo}$ 不适用的规则 → 25 条剩 1~3 条/step
- **Topological lineage table**：每个参数标记 replicated / sharded（防止 TP 分片张量误报）
- **Hook & query installation**

**Phase 2 — Per-Step Verification**:
- **Type-B hooks**（实时）：在 5 个 hookpoint 拦截函数调用，记录 invocation count / 调用顺序
- **Type-A SQL** （step 完成后）：DuckDB 上查 trace database
  - WHERE 子句 = $\pi_\text{topo} \wedge \pi_\text{precond}$
  - HAVING 子句 = $\pi_\text{schema}$
  - 只返回违规行
- **Type-C 静态断言**：在 model init 时跑一次

**Phase 3 — Drill-down Cascade**（违规后做的根因定位）:
```
DP Consistency Check                  (粗粒度，整 tensor checksum)
   ↓ 触发
Optimizer State Dict Consistency      (中粒度，按 module / rank)
   ↓ 触发
Nested State Bitwise Check            (细粒度，bit-level)
   ↓
Final report: 违规 invariant + rank/step + 嫌疑模块
```

---

## 4. Three-Layer Invariant Decomposition（理论核心，画图必用）

任何 runtime-checkable invariant 可分解为：

$$p(s) = \underbrace{\pi_\text{schema}(s)}_{\text{看哪些字段}} \;\wedge\; \underbrace{\pi_\text{topo}(s;\tau)}_{\text{哪些 rank 必须一致}} \;\wedge\; \underbrace{\pi_\text{precond}(s;t,\phi)}_{\text{什么时机/模块有效}}$$

**举例：DP=2 跑 missing all-reduce 的 bug**
- 朴素 invariant：$\theta_\text{rank0} = \theta_\text{rank1}$
- 但要在生产可用，必须回答三个问题：
  - **what**：比 checksum 不是比 raw tensor → $\pi_\text{schema}$
  - **on which params**：只比 DP-replicated weight，不比 TP-sharded → $\pi_\text{topo}$
  - **when**：在 step-1 同步之后比，不在 init 比 → $\pi_\text{precond}$

**Coverage Staircase（覆盖楼梯，重要图素材）**:
- 只有 $\pi_\text{schema}$：覆盖 **20%** bug
- $\pi_\text{schema} + \pi_\text{topo}$：**51%**
- 三层全有：**74%**
- 剩下 26% 不可在 runtime 观测（不是框架缺陷，是物理上限）

**与现有方法对应**:
- TrainCheck 只有 $\pi_\text{schema}$（不懂拓扑）→ 在 TP 上 FPR=100%
- TTrace 加了 $\pi_\text{topo}$ 但需 reference 实现，离线
- TrainVerify 有 $\pi_\text{topo} + \pi_\text{precond}$，但是静态 plan-level，不查 runtime value
- **TrainAudit** 三层都有，且 in-flight

---

## 5. 8 个 Detection Pattern（Pattern Catalog）

从 128 真实 bug 蒸馏出来的"通用错误模式"，覆盖 66% bug：

| ID | 名字 | 一句话语义 | Type | 覆盖 |
|---|---|---|---|---|
| **P1** | Dtype Preservation | 非显式 cast 函数的输出 dtype 应等于输入 dtype | B | 12 |
| **P2** | Scaling Consistency | 跨 N 个 rank 聚合时应有 $1/N$ 或 $N$ 缩放因子 | B | 10 |
| **P3** | Cross-Rank Replication | 非 sharded 参数必须在并行组内 bit-wise 一致 | A | 25 |
| **P4** | Invocation Frequency | 某函数每 step 必须被调用 K 次 | B | 11 |
| **P5** | State Restoration | train→eval→train 之后所有训练状态必须恢复 | A | 5 |
| **P6** | Structural Integrity | 模型层数/形状必须匹配配置 | C | 10 |
| **P7** | Residual Stream Integrity | 残差连接用原始输入而非 norm 后的 | B | 5 |
| **P8** | Counter Consistency | 内部 counter 在初始化和 reset 时必须一致 | A | 6 |

**Pattern 形式化**：4-tuple $\langle r, \sigma, m, \rho \rangle$
- $r$ = rule（关系）
- $\sigma$ = scope（适用范围谓词）
- $m \in \{A, B, C\}$ = 执行模式
- $\rho$ = precondition

实例化后：$p(s) = r(s) \wedge \sigma(s; \tau) \wedge \rho(s; t, \phi)$，对应 three-layer。

---

## 6. 关键 Bug Survey 数字（图标必备）

- **295** 真实 silent error 案例（4 个框架）
- **128** 有定位到代码层的 root cause（用于分析）
- **12** 类语义故障（taxonomy 饼图 → `fig_taxonomy_donut.pdf`）
- 4 个框架分布：Megatron-LM 60、DeepSpeed 75、OLMo 70、OLMo-core 60

**故障类别（12 类）举例**：gradient sync、optimizer state、checkpoint、loss computation、data loading、dtype/precision、collective comm、counter、router/MoE、residual、norm、init...

---

## 7. 实验关键数字（图表素材）

### Detection Efficacy（D1 benchmark, 17 bug）
| 方法 | 检出 | 假阳 |
|---|---|---|
| **TrainAudit** | **17/17 (100%)** | 1/17 (5.9%, 仅 surrogate) |
| TrainCheck | 10/17 (58.8%) | 0/17 (self-check) |
| Naïve Monitoring | 0/17 (0%) | 0/17 |

### Real Reproduction Set（26 bugs，跨 4 框架）
- TrainAudit 检出 **23/26 = 88.5%**
- 3 个 boundary FN（B14、B15 sub-percent loss drift；B7 单 PP stage 被 dilute）
- **额外发现 3 个公共 issue 没记录的全新 bug**（OLMo-core HEAD）：
  - `OLMOCORE_EP_A2A_UNINIT_BUFFER`（EP all_to_all 缓冲区有 7.13% 未初始化 NaN）
  - `REORDERED_HYBRID_DEAD_BLOCK0`（block 0 永久 eps-clamp）
  - `REORDERED_DENSE_BLOCK0_ATTN`（dense 路径瞬态 eps-clamp）

### 三层 Coverage Staircase（128 bug 子集）
- 只 schema → 20%
- + topology → 51%
- + precondition → **74%**

### Topology-Aware Pruning 效果
- 朴素 cross-rank 检查在 TP=2 干净跑：误报 **57/492** 参数
- 加 $\pi_\text{topo}$ → 误报 **0**

### 在线开销
- **per-tick 评估上限 ~10 ms**（CPU-side 估算）
- 25 条规则 / 8 个 hookpoint → 每个 hookpoint 平均 ~3 条规则
- 每条 SQL 评估 2~3 ms
- 4/4 acceptance test 通过

### 离线 Mining 成本（一次性）
- ~350 LLM 调用迭代
- ~40~50 M tokens
- ~$70~100（GPT-4.1 价位）
- ~8 小时
- 输出 25 条 active rules + 13 dsl_native 模板

### Cross-Framework Portability（5 个 adapter）
| 框架 | T0 检出 | T1 检出 | Adapter LoC |
|---|---|---|---|
| Megatron-LM | 1 | 5 | ~150 |
| OLMo | 3 | 3 | ~150 (+probe) |
| OLMo-core | 4 | 2 | ~150 (+probe) |
| DeepSpeed | 2 | 3 | ~50 |
| FSDP | – | – | ~30 |

- T0 规则跨 5 个框架、~31 个月历史 commit、**100% 存活率**
- driver pool 覆盖 **288/295 = 97.6%** 真实 bug
- 集成测试 **104/104** 通过

### Hardware
- 单节点 8× NVIDIA H200，NVLink 900 GB/s
- PyTorch 2.1 + NCCL 2.18

---

## 8. 三个核心 Case Study（论文反复引用）

### Case 1: DeepSpeed ZeRO-2 Micro-Step Counter Mismatch
- **Bug**：`micro_step_id` 计数器 off-by-one，gradient buffer 在该累加时被 reset → silently 丢梯度
- **现象**：loss 曲线和健康跑几乎一样（>700 step 才偏离）
- **TrainAudit 命中**：P8（Counter Consistency）实例化为 SQL，**第一步**就触发
- **关键画图素材**：loss 曲线两条线长时间重合 + 紫色虚线在 step 1 就触发警报

### Case 2: Megatron-LM TP Router Weight Divergence
- **Bug**：SwitchMLP MoE 模块中 router weight 应在 TP rank 间复制相同，缺少 all-reduce 导致漂移
- **现象**：loss 只高 7%（无 reference 几乎不可见）
- **TrainAudit 命中**：P3（Cross-Rank Replication）+ topological lineage table
  - 朴素跨 rank 等值检查 → 干净跑误报 57/492
  - 加 $\pi_\text{topo}$ 区分 replicated vs intentionally sharded → **57 → 0**
- **关键画图素材**：参数差异柱状图 / FP 数量从 57 降到 0 的对比

### Case 3: MoE Aux-Loss Double Accumulation
- **Bug**：辅助 loss accumulator 每 step 调用 2 次（应为 1 次），引入 scaling 错误
- **现象**：前 20 步 checksum / grad-norm / loss 完全 bit-wise 一致 → 任何 value-level 检查都看不到
- **TrainAudit 命中**：P4（Invocation Frequency）→ Type-B hook 计数 invocation
  - 期望 = 1，实际 = 2 → **第一步就报警**
- **关键画图素材**：invocation count 柱（buggy = 2 vs fixed = 1），证明 Type-B hook 不可缺

---

## 9. Story Arc 与论文章节映射

| 章节 | 内容 | 已用 figure | 可能新增 figure |
|---|---|---|---|
| §1 Intro | Silent error 痛点 + DeepSpeed ZeRO-2 例子 | `figures/figure1.png`（micro-step 示意） | 重画一张更清晰的 ZeRO-2 流程对比 |
| §2 Background | Silent error 影响 + 4 大挑战 | `fig_sdc_impact.pdf`（loss/grad-norm 看不到 bug） | – |
| §3 Empirical Study | 295 案例调查 + 12 类 taxonomy + 三层架构 | `fig_taxonomy_donut.pdf` | Coverage staircase 阶梯图（20%/51%/74%） |
| §4 Overview | 三组件架构图 | `figures/figure2.png`（架构图） | 重画更清晰的 offline / online 分阶段架构 |
| §5.1 Pattern Catalog | 8 个 pattern + 4-tuple 定义 | – | Pattern catalog 总览 |
| §5.2 Invariant Miner | FSM + 5 agent + 4-step 漏斗 | `agents-cue2.png`（FSM 图）| FSM 流程图、4-step 漏斗 |
| §5.3 Data Collector | Tiered schema + hookpoint | `tier_coverage.pdf`（覆盖 vs 开销） | 5 个 hookpoint 在训练 loop 上的位置图 |
| §5.4 Verifier | 3 phase + drill-down | – | drill-down cascade 树状图 |
| §6 Evaluation | 三方对比、case study、cross-framework | 一系列 `fig_*.pdf` | – |

---

## 10. 必须避免的画图陷阱

1. **不要把 LLM 画在 online path**：LLM **只在 offline mining 阶段**调用。online 阶段 100% 是 deterministic SQL/hook/assertion。这是论文的核心 selling point（低开销）。
2. **不要把 invariant 画成单层公理**：必须显式画出 three-layer（schema / topology / precondition）的拆分。
3. **不要混淆 pattern 和 invariant**：pattern 是抽象模板（8 个），invariant 是 pattern 在具体框架代码上实例化后的可执行规则（25 条）。
4. **不要把 TP-sharded 张量画成"应该相等"**：TP 分片 by design 在不同 rank 持不同切片；任何把它们画成"相等"的图就违反论文的 motivating example。
5. **拓扑符号统一**：DP（数据并行）/ TP（张量并行）/ PP（流水线并行）/ EP（专家并行）/ CP（context parallel）/ ZeRO stage（0/1/2/3）。
6. **方向性箭头**：Miner → Invariant Library → Verifier；Training Loop → Data Collector → Trace DB → Verifier。**两条线汇合在 Verifier**。

---

## 11. 论文 narrative 关键信息（防误导）

- **TrainAudit 不是 reference-based 工具**：不需要 golden run，不需要 single-device baseline。
- **TrainAudit 是 in-flight 的**：每 step 实时检查，违规即报；不像 TrainCheck 是 post-mortem，不像 TTrace 是 offline。
- **74% 上限不是 framework 限制**：是任何 runtime checking 的物理上限（剩下 26% 需要 reference 或 offline 检查）。
- **核心反幻觉机制 = Bidirectional Adversarial Verification**：让 LLM 主动构造反例攻击自己生成的约束，没反例 + Conf ≥ 0.8 才接受。比单纯的 "LLM 生成 invariant" 高出一个量级的可靠性（文献里 LLM 生成 invariant 仅 16% 成功率）。
- **核心 design split**：LLM 找语义关系，deterministic 逻辑（runtime topology pruning）管 scope；不让 LLM 同时把两件事都做对。

---

## 12. 已完成的关键 figure（可作为风格参考）

`figures/` 目录下已有 PDF：
- `fig_sdc_impact.pdf` — silent error 影响 loss / grad-norm 的可视化
- `fig_taxonomy_donut.pdf` — 12 类 fault taxonomy donut
- `tier_coverage.pdf` — Tier coverage vs runtime overhead
- `fig_case_studies.pdf` — 三个 case study 综合图
- `fig_hunt_novel_bugs.pdf` — 3 个新发现 bug 的可视化
- `fig_cross_framework.pdf`、`fig_cross_version_timeline.pdf` — 跨框架/跨版本
- `fig_detection_boundary.pdf` — 检测边界
- `fig_adapter_loc.pdf` — adapter 代码量
- `agents-cue2.png` — Invariant Miner FSM 流程
- `figure1.png` / `figure2.png` — 整体架构图（可能要重画）

新画图前**先翻一下 `figures/` 看是否已有类似的**，避免重复劳动。

---

## 13. 我（论文作者）需要新画图的几种典型场景

如果你被指派的任务属于以下之一，重点参考章节：

| 任务类型 | 重点参考 |
|---|---|
| 重画系统架构图（offline+online） | §3、§4、第 11 节避坑 |
| 画 Three-Layer 拆解示意 | §4 |
| 画 Coverage Staircase（20% → 51% → 74%） | §7.3 三层 coverage |
| 画 Invariant Miner FSM / 4-step 漏斗 | §3 ❶、`agents-cue2.png` 风格 |
| 画 Hookpoint 在训练 loop 上的位置 | §3 ❷ |
| 画 drill-down cascade 树 | §3 ❸ phase 3 |
| 画 P3 实例化在 Megatron 上的端到端例子 | §5 + §8 case 2 |
| 画跨框架 / 跨版本对比柱状或时间线 | §7 表格 |

如果有具体的图想要画，告诉我图号或想法，我可以补充更详细的数据点和约束。

---

## 14. 联系作者校对清单

画完后请检查以下"硬性事实"是否对：
- [ ] 4 个目标框架名字拼写：**Megatron-LM**、**DeepSpeed**、**OLMo**、**OLMo-core**（外加 FSDP adapter）
- [ ] 25 条 active rules 这个数（不是 23、不是 30）
- [ ] 5 个 hookpoint 名字（before-forward / after-forward / main-grad-in-backward / after-backward / before-optimizer）
- [ ] 8 个 pattern 编号 P1~P8 + Type A/B/C 分类对应（3 个 A、4 个 B、1 个 C）
- [ ] D1 = 17 bug，real set = 26 bug，hunt 新发现 = 3 bug
- [ ] 三层符号：$\pi_\text{schema}$ / $\pi_\text{topo}$ / $\pi_\text{precond}$
- [ ] LLM **只在 offline** 调用，**online 无 LLM**
