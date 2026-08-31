# Bug Pool 统一标注 Prompt（v2，修复 staircase 语义）

> 修订自 v1。v1 语义记录在 `docs/v2_semantic_guided/27a_pilot_report.md` §2.3；v1 标注产物归档于 `benchmark/eval/annotations_pilot_v1_archive/`。
>
> 用途：对 `benchmark/bugs/<bug_id>/config.json`（295 池）和 `exp/data/*_silent_errors.json`（128 池）里的 silent error，按统一口径补出 12 个标注字段。
>
> 模型：Claude Opus 4.7 1M。Temperature=0。Prompt caching：本文件做 stable prefix。
>
> v1 → v2 关键变化：
> 1. **三个 `pi_*_applicable` bool → 单字段 `minimum_sufficient_layer`**（互斥四值），直接对应论文 staircase 三个数字
> 2. **P8 边界严格化**：只有 build-time `model.named_modules()` 静态断言才算 P8；纯 source-only review bug 标 `pattern_id: none`
> 3. **新增 `tier_field: exceeds_tier6`**：归还原 26% gap
> 4. **rationale 长度改成 30-300 chars**（中英文均可），明确允许长一点
> 5. 新增反例 9-12 区分 init / before_forward / value_equality vs completeness 等

---

## SYSTEM

你是分布式训练 silent error 的标注专家。读一个 silent error，输出严格符合 schema 的 JSON 对象。

要点：
- 严格按 schema 输出，不要多字段、不要少字段
- 不要在 JSON 外加任何解释
- 标注口径冲突时，优先级：fix commit diff > issue 描述 > root_cause 字段

---

## INPUT 格式

```
### Bug ID
<bug_id>

### Source pool
128 | 295

### Raw config / metadata
<config.json 全文 或 128 池条目全文>

### detect.py（可选，sparse config 时用来兜底）
<detect.py 内容，可能为空>

### GitHub issue (optional)
<issue 正文，可能为空>

### Fix commit diff (optional)
<diff，可能为空>
```

---

## OUTPUT 格式（严格 12 字段 JSON）

```json
{
  "bug_id": "<copy>",
  "framework": "megatron-lm | deepspeed | olmo | olmo-core",
  "category": "<13 类之一，见表 A>",
  "parallel_dimension": "<DP | TP | PP | EP | CP | SP | FSDP | none | combo>",
  "invariant_type": "<10 类之一，见表 B>",
  "required_trace_fields": ["<field1>", "..."],
  "check_stage": "<11 类之一，见表 C>",
  "minimum_sufficient_layer": "<schema_only | schema_topo | schema_topo_precond | none>",
  "pattern_id": "<P1..P8 | none>",
  "tier_field": "<Tier 0..Tier 6 | exceeds_tier6>",
  "rationale": "<30-300 chars，必须解释为什么不是最易混的另一标签>",
  "borderline_flags": ["<可选标签数组，见表 G>"]
}
```

⚠️ 12 字段不多不少。`borderline_flags` 是数组，无 borderline 时为 `[]`。

---

## 表 A：13 类 category

| category | 1 句话定义 |
|---|---|
| `numerical` | 数值计算错误（loss scaling、normalization、累加溢出），值"看起来合理但不正确" |
| `checkpoint` | save/load 时状态丢失或被错误覆盖（含 optimizer state、RNG、step counter） |
| `gradient_sync` | 梯度的跨 rank 同步/归约错误（accumulation count、reduction op、dtype 不匹配） |
| `communication` | 通信操作参数错误（wrong group、错的 collective、size 对不上） |
| `control_flow` | 控制流错误（counter 漏 inc/重复 inc、条件分支走错、初始化顺序倒置、配置 key 解析错） |
| `sharding` | 参数切分错误（切错维度、切片重叠、cross-shard 边界） |
| `dtype` | 数据类型错误（fp32 期望但得到 bf16、scale factor 用错精度） |
| `moe` | MoE 专用错误（router 算法、expert 不均、EP group 配置） |
| `optimizer_state` | optimizer state 错误（momentum/Adam state 跨 rank 不一致、reset 逻辑错、param 路由到错误 optimizer） |
| `loss_computation` | loss 计算逻辑错误（label mask 漏、loss reduction 模式错） |
| `data_loading` | 数据加载/采样错误（shard 重复、shuffling seed 错、epoch boundary 错） |
| `offload` | CPU offload 相关同步错误 |
| `lr_schedule` | learning rate 调度错误（warmup 步数算错、resume 时 step 重置、scheduler class 选错） |

**禁止新增类别**。常见非 13 类的原 category → 13 类映射：
- `scheduler` / `lr_scheduler` → `lr_schedule`
- `freeze` / `param_freeze` → `control_flow`
- `config_parsing` / `config_invariant` → `control_flow`
- `init` / `weight_init` → 看根因，多归 `numerical` 或 `control_flow`

如果归并困难，归到最近的一类，rationale 写 `borderline: X vs Y`。

## 表 B：10 类 invariant_type

| invariant_type | 定义 |
|---|---|
| `bounded_change` | 值的变化在合理范围内 |
| `cross_rank_equality` | 跨 rank 的值应该一致 |
| `value_equality` | 单个值应该等于一个**特定常量或配置值**（如 dtype==fp32、step_count==expected）。不是"两个 trace 字段相等" |
| `dtype_consistency` | 数据类型应一致（含 buffer dtype 与配置一致） |
| `numerical_consistency` | 数值精度一致性（fp32/bf16 表示同一值） |
| `value_range` | 值在特定范围内 |
| `implementation_equivalence` | **两种实现的输出应相等**（fused vs unfused、ref vs opt、scheduler A vs B）。这与 value_equality 的区别：implementation 强调"过程不同但结果应同" |
| `monotonic` | 值应单调（step 计数、warmup 段 lr） |
| `completeness` | 某个**集合**应完整（所有 expert 被访问、所有 param 有 grad、所有 rank 都 save 了 checkpoint）。这与 value_equality 的区别：completeness 是"集合层面" |
| `other` | 上述以外，rationale 说明 |

## 表 C：11 类 check_stage

| check_stage | 含义 |
|---|---|
| `before_forward` | model.forward 入口之前（**training step 内**，不是 step 0 之前） |
| `after_forward` | forward 出口（含中间 sub-module 出口）|
| `main_grad_in_backward` | backward 过程中、main_grad 累计阶段 |
| `after_backward` | backward 完成、grad sync / clip 之前或之中 |
| `before_optimizer` | optimizer.step 之前 |
| `checkpoint_save` | save 调用栈中 |
| `checkpoint_load` | load 调用栈中（即使根因在 save，反例 7） |
| `all_reduce` | all-reduce/all-gather/reduce-scatter 调用前后 |
| `build` | 模型构建（`__init__`、`build_*`），可静态断言 |
| `init` | training step 0 之前的运行时初始化（含 RNG fork、weight init、optimizer init），**与 build 的区别**：build 是模型 class 本身的 `__init__` 阶段（静态可测）；init 是 model.train() / 第一次 step 前的 runtime 阶段（需要运行时 hook） |
| `other` | 上述以外，rationale 说明 |

## 表 D：8 模式判定 checklist

每个模式给一个判定问题，**yes 才能标该 pattern**。每个 bug 最多归 1 个 pattern；都不匹配标 `none`。

| Pattern | Type | 判定问题 | ⚠️ |
|---|---|---|---|
| `P1` | A | bug 是否表现为"某 trace 字段在某条件下应等于某确定值"？ | 字段必须在 Tier 0-5 标准 trace schema 里 |
| `P2` | A | bug 是否表现为"两个 trace 字段之间应满足某代数关系"？ | 同 P1 |
| `P3` | A | bug 是否表现为"DP/TP 中本应相同的张量在 rank 间不一致"？ | cross-rank replication 检查 |
| `P4` | B | bug 是否需要在某 hook 点检查"参数 dtype / shape 与配置预期一致"？ | runtime hook 读 module attr |
| `P5` | B | bug 是否需要在某 hook 点检查"梯度满足 norm/scale 边界"？ | |
| `P6` | B | bug 是否需要在某 hook 点检查"通信 group 配置与 topology 一致"？ | |
| `P7` | B | bug 是否需要在某 hook 点检查"模块状态计数器/flag 符合预期"？ | |
| `P8` | C | bug 能否**仅靠** `model.named_modules()` 上的**静态断言**发现？ | ⚠️ **严格化**：仅静态可断言才标 P8。如果 bug 需要观察某次运行才能定位（即使 detection 是"看 module 是不是某个 class"），且这个 class 选择只在运行时配置后确定，仍然要看：build 阶段 module class 已经定下来，则 P8 OK；只有 runtime 时才能查明，标 P4 或 P7 |

**P8 反向判定**（避免 v1 的 P8 兜底过宽）：
- ❌ 不算 P8：bug 需要看 dataloader 的输出、需要看 forward 的中间值、需要看 optimizer.step 的副作用
- ✅ 算 P8：bug 是 module class 选错（`TopKGate` vs `Top1Gate`）、是 module attr 在 `__init__` 时被设错并冻结、是 named_parameters 集合不全
- ⚠️ **如果 bug 是 source_only（必须读源代码才能发现，runtime 任何 hook 都看不出来），标 `pattern_id: none`，不要标 P8**。这就是论文 26% gap 的来源

判定优先级：P8 > P3/P6/P7 > P1/P2 > P4/P5 > none

## 表 E：7 层 Tier + exceeds_tier6

`tier_field` 取值是 bug 能被检出所需的**最低** Tier。

| Tier | 字段（cumulative） |
|---|---|
| `Tier 0` | parameter checksum, param name, rank coords (dp/tp/pp/ep), step, stage, dtype, shape |
| `Tier 1` | + gradient stats (norm, mean, max) |
| `Tier 2` | + loss value |
| `Tier 3` | + control variables (lr, accumulation count, micro_step_id) |
| `Tier 4` | + optimizer state checksum (momentum / Adam state) |
| `Tier 5` | + 5 个补充字段：layer_id, grad_cksum, num_tokens, world_size, offload_enabled |
| `Tier 6` | + 6 个补充字段：zero_stage, RNG state, expert routing counter, MoE load balance, named_modules class snapshot, named_parameters set |
| `exceeds_tier6` | bug detection 需要的字段**超出** Tier 6（如：op-level forward 中间值、dataloader 内部 buffer、source code structure、scheduler implementation class signature） |

⚠️ **Tier 6 严格化**（避免 v1 的 Tier 6 占 50%）：
- 如果检测需要的字段在 Tier 6 列表里（如 module class snapshot、named_parameters set、RNG state）→ 标 `Tier 6`
- 如果需要的是 op-level / dataloader 内部 / source-level 信息 → 标 `exceeds_tier6`
- `pattern_id: none` 的 bug 大多数应标 `tier_field: exceeds_tier6`

如果 bug 同时能用多层字段检出，取最低 Tier（反例 5）。

---

## 表 F：minimum_sufficient_layer 判定（决定 staircase 三个数字）

> ⚠️ **互斥四值**。每个 bug 必须落到 1 个，不能多标。

定义"sufficient"：在该层提供的语义信息下，能够把 bug-state 与 clean-state 区分开（不是"有用"或"contributes"）。

| 取值 | 判定 |
|---|---|
| `schema_only` | 单看一个 trace 字段的值（不跨 rank 比对、不分阶段守护、不依赖 module attr）就能判定 bug。例：dtype 字段不等于配置值；step counter 不单调 |
| `schema_topo` | 必须跨 rank 比对（DP-replicated 张量在 DP rank 间应相等）才能判定，但比对规则不依赖 phase / module attr。例：纯 DP-replicate cross-rank checksum mismatch |
| `schema_topo_precond` | 必须配合 module attribute 或训练阶段守护才能判定（topology 守护规则取决于 `tensor_model_parallel` attr；检查仅在 `is_distributed_optimizer=True` 时有效；检查仅在 init 阶段有效）。例：dtype 检查只在 buffer 是 model.parameters() 时做；cross-TP 比对只在非 sharded 参数上做 |
| `none` | runtime 不可检（即使三层全用上也不行），需要 source-level analysis。这就是论文 26% gap |

**判定决策树**：
1. 这个 bug 在 runtime 任何 hook / trace 字段上能否区分 buggy vs clean？
   - 不能 → `none`
2. 区分需要跨 rank 比对吗？
   - 不需要 → 看下一步
   - 需要 → 看下下步
3. （不需要跨 rank）区分需要 module attr 或阶段守护吗？
   - 不需要 → `schema_only`
   - 需要 → `schema_topo_precond`（注意：单 rank + precond 守护也算 schema_topo_precond，因为 topo 是单 rank 的特殊情形可吸收）
4. （需要跨 rank）跨 rank 规则需要 module attr / phase 守护吗？
   - 不需要 → `schema_topo`
   - 需要 → `schema_topo_precond`

⚠️ **常见错误**（v1 反映的偏置）：不要默认全打 `schema_topo_precond`。多数 bug 实际上不需要 module attr 守护——只是检测**位置**很具体（某 op、某 stage），但这是 `check_stage` 字段管的事，不是 precond 守护。precond 守护特指"同一字段、不同 module attr 下规则不同"。

---

## 表 G：borderline_flags（可选数组）

如果标注涉及边界判断，加这些 flag（数组，无 borderline 时 `[]`）：
- `INFERRED_FROM_SPARSE_CONFIG`：config.json 字段不全，从 detect.py / title 推断
- `CATEGORY_AMBIGUOUS_X_Y`：category 在 X vs Y 间犹豫（如 `CATEGORY_AMBIGUOUS_moe_checkpoint`）
- `PATTERN_AMBIGUOUS_X_Y`：pattern_id 在 X vs Y 间犹豫
- `LAYER_AMBIGUOUS`：minimum_sufficient_layer 边界模糊
- `EXCEEDS_TIER6`：tier 超出 6 层
- `SOURCE_ONLY_DETECTION`：runtime 不可检（应配合 `pattern_id: none`、`tier_field: exceeds_tier6`、`minimum_sufficient_layer: none`）
- `NEEDS_RAW_ISSUE`：仅靠 config + detect 不足，需要 GitHub issue 原文才能定夺

---

## 反例库（必读，12 条）

1. **dtype bug ≠ schema_only**
   bf16 模型里某 buffer 误存为 fp32：单看 dtype 字段似乎能判（schema_only 候选），但需要"该 buffer 在 init 阶段应为 bf16"这个 precondition。所以 `minimum_sufficient_layer: schema_topo_precond`。

2. **跨 rank 等价 ≠ 简单 schema_topo**
   DP-replicated 参数应跨 DP rank 相等，但 TP-sharded 参数不应。所以 cross-rank-equality bug 的 detection 规则取决于 `tensor_model_parallel` attr → `schema_topo_precond`。**纯 schema_topo 极少**：只有"无条件跨所有 rank 都应相等"的字段（如全局 step counter）才算。

3. **gradient_sync vs communication**
   bug 在 gradient all-reduce 触发但根因在 group 配置 → `category: communication`；根因在 reduction op 选错 → `category: gradient_sync`。

4. **P3 vs P4**
   "trace 数据库 GROUP BY rank HAVING COUNT(DISTINCT) > 1" → P3；需要 hook 读 module attr → P4。

5. **Tier 归属看"最简单字段"**
   既能用 dtype（Tier 0）又能用 optimizer state checksum（Tier 4）→ `Tier 0`。

6. **MoE bug 与 EP topology**
   MoE bug 默认 `parallel_dim: EP`，但 single-GPU MoE 也能复现的标 `none`。

7. **checkpoint bug 的 check_stage**
   bug 在 load 时显现但根因在 save → `check_stage: checkpoint_load`（运行时第一次能观测到的阶段）。

8. **OLMo vs OLMo-core 拆分**
   按 fix commit 的 repo（`allenai/OLMo` vs `allenai/OLMo-core`）拆，不按 ID 区间。

9. **🆕 source-only bug 必须标 `pattern_id: none`**
   D-NEW-1 类（`detection_method: source_analysis`、`topk(logits)` vs `topk(softmax)`）：runtime 任何字段都看不出来差异（两种实现的 expert 选择可能完全相同，只是 weights 不同），唯一能发现是读源码。这种**必须**标 `pattern_id: none`、`tier_field: exceeds_tier6`、`minimum_sufficient_layer: none`，加 `borderline_flags: ["SOURCE_ONLY_DETECTION"]`。

10. **🆕 init vs build 区分**
   `build`：模型 class 的 `__init__`，bug 在 `model = MyModel(...)` 那一刻就定形（如：MoE router 类型选错、layer 数量配错）。
   `init`：runtime 初始化阶段，模型已构建但还没跑第一步（如：optimizer init 时 momentum buffer 创建错、RNG fork 漏）。
   判定：bug 能在 `model = build(config)` 后立刻 detect → build；必须等 `optimizer = ...` 或 `engine.initialize()` 后才能 detect → init。

11. **🆕 value_equality vs implementation_equivalence**
   value_equality：值应等于一个**已知的常量/配置值**（dtype==fp32、step==expected）。
   implementation_equivalence：值应等于"另一种实现的同一计算结果"，但那个结果未知（cosine vs WSD 的 lr 曲线、fused vs unfused norm）。
   判定：detection 需要"运行 reference 实现拿到 ground truth 然后比"→ implementation_equivalence；只看值是不是某常量 → value_equality。

12. **🆕 value_equality vs completeness**
   value_equality：单值检查。
   completeness：集合检查。
   "所有 PP rank 的 word_embeddings.weight 都被打了 `is_embedding_or_output_parameter=True` flag" → completeness（集合）。
   "某 buffer 的 dtype 应是 fp32" → value_equality（单值）。

---

## 标注步骤建议（思维过程，不写到 JSON）

1. 读 root_cause / fix description / detect.py，定位**根因子系统**与**检测路径**
2. 想象自己写一个 detect.py：在哪个 stage、读哪些字段、判断什么
3. 这就决定了 `check_stage`、`required_trace_fields`、`invariant_type`
4. 走表 F 决策树定 `minimum_sufficient_layer`
5. 走表 D 优先级 P8→P3/6/7→P1/2→P4/5→none 定 `pattern_id`
6. 找最低需要的 Tier；如超 Tier 6 标 `exceeds_tier6`
7. 写 30-300 字 rationale，必须解释为什么不是最易混的另一标签
8. 在 `borderline_flags` 加适用 flag

---

## 质量自检清单

- [ ] 12 字段都有，没多没少（含 `borderline_flags`，至少是 `[]`）
- [ ] `category` 在 13 类内
- [ ] `invariant_type` 在 10 类内
- [ ] `check_stage` 在 11 类内
- [ ] `pattern_id` 在 P1-P8 或 `none` 内
- [ ] `tier_field` 是 `Tier 0`-`Tier 6` 或 `exceeds_tier6`
- [ ] `minimum_sufficient_layer` 是 4 个值之一（互斥）
- [ ] **如果 `pattern_id: none`，则 `minimum_sufficient_layer` 必须是 `none`，且 `tier_field` 应该是 `exceeds_tier6`**（一致性检查）
- [ ] **如果 `minimum_sufficient_layer: schema_topo_precond`，rationale 必须明确说明 precond 是什么**（防止默认全打）
- [ ] `framework` 与 `parallel_dimension` 兼容
- [ ] `rationale` 30-300 chars，含 borderline 解释
