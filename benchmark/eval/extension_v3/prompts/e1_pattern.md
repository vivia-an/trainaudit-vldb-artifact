# E1 Prompt: 8-Pattern Coverage Annotation

> Output: `pattern_coverage_392.json` per-bug entry containing `patterns_hit` array.
> Pattern labels follow the released frozen catalog.

## TASK

You are classifying a silent error bug against 8 detection patterns derived from 128 hand-curated bugs. For the given bug, output **which pattern(s) can detect it**. A bug may match 0, 1, or multiple patterns.

## 8 PATTERN DEFINITIONS

| ID | Name | Type | Semantic Rule |
|---|---|---|---|
| **P1** | Dtype Preservation | B (runtime hook) | 非显式类型转换函数的输出 dtype 应与输入 dtype 一致 — output dtype of any non-cast function should match input dtype |
| **P2** | Scaling Consistency | B | 跨 N 个 rank 聚合的值应包含 1/N 或 N 的缩放因子 — values aggregated across N ranks should carry 1/N or N scaling factor |
| **P3** | Cross-Rank Replication | A (trace SQL) | 非分片参数必须在同一并行组的所有 rank 上逐位相同 — non-sharded parameters must be bit-identical across ranks of the same parallel group |
| **P4** | Invocation Frequency | B | 某个特定函数每个训练 step 必须被调用恰好 K 次 — a specific function must be invoked exactly K times per training step |
| **P5** | State Restoration | A | 所有训练相关状态必须在 train→eval→train 切换后被恢复 — all training-relevant state must be restored after train→eval→train transition (also covers checkpoint save/load round-trip) |
| **P6** | Structural Integrity | C (static check) | 模型结构（layer 数、维度、形状）必须与配置参数匹配 — model structure (layer count, dims, shapes) must match config |
| **P7** | Residual Stream Integrity | B | 残差连接必须使用原始输入而不是规范化/变换后的版本 — residual connection must use the original input, not normalized/transformed version |
| **P8** | Counter Consistency | A | 内部控制流计数器在初始化与重置之间必须保持一致 — internal control-flow counters (micro_step_id, accumulation count, etc.) must be consistent between init and reset |

## DECISION RULES

1. **Read** the bug's `root_cause`, `invariant`, `detection_method`, `category` fields.
2. **For each of P1..P8**, ask: "Can this pattern's rule, properly instantiated, detect this bug?"
3. **Multi-pattern OK**: a bug can match >1 pattern. Example: `dtype` bug in cross-rank context may match P1 + P3.
4. **No-pattern OK**: if the bug requires source-code inspection or off-line ground-truth (e.g., source_only bugs, runtime-unobservable behavior), output `[]` (empty list).
5. **Be strict on scope**: a pattern matches only if its rule actually catches the bug, not "tangentially related."

### Detailed disambiguation

- **P1 vs P3**: P1 is about a single function's input/output dtype consistency (single-rank check). P3 is about a parameter's value being identical across ranks. A bug like "all-reduce in wrong dtype causes cross-rank divergence" matches BOTH P1 (dtype mismatch in collective) and P3 (resulting cross-rank divergence).
- **P2 vs P4**: P2 is about a numerical scaling factor (1/N or N). P4 is about call count. "MoE aux-loss not divided by tp_size" → P2. "MoE aux-loss accumulator called twice" → P4.
- **P3 vs P6**: P3 is runtime cross-rank value check. P6 is build-time structural check. "Wrong layer count for config" → P6. "Layer count consistent but weights diverge" → P3.
- **P5 vs P8**: P5 is broader (any train state including optimizer/RNG/step). P8 is specifically about counter monotonicity/consistency between init and reset paths. "step counter increments correctly across micro-batches but resets wrongly on resume" → P8.
- **P7**: Specific to residual connection bugs (post-norm vs pre-norm, residual using transformed input).

### Few-shot examples

**Example 1** (matches P3):
- bug_id: B1
- title: SwitchMLP router torch.nn.Linear init not wrapped in data-parallel RNG tracker
- root_cause: Buggy init does `self.router = torch.nn.Linear(...)` outside of any RNG tracker fork. If the default CUDA RNG state is asymmetric across TP ranks, weights diverge.
- patterns_hit: ["P3"]
- rationale: Router weight should be replicated across TP ranks but diverges; classic Cross-Rank Replication violation.

**Example 2** (matches P1):
- bug_id: M-NEW-1
- title: MoE aux loss sigmoid computed in bf16 instead of fp32
- root_cause: torch.sigmoid(logits) where logits is bf16 → precision loss in routing scores
- patterns_hit: ["P1"]
- rationale: Sigmoid is a non-cast function; its input should be fp32 (per spec) but is bf16. P1 dtype preservation violated.

**Example 3** (matches P8 + P4):
- bug_id: B10 / D-NEW-X
- title: ZeRO-2 with CPU offload: micro_step_id off-by-one
- root_cause: micro_step_id counter increments wrongly, causing gradient buffer reset at wrong micro-batch boundary
- patterns_hit: ["P8"]
- rationale: Counter inconsistency between init (=0) and reset path (≠expected). P4 also possible if counter triggers function-call mismatch but P8 is more direct.

**Example 4** (no pattern, source-only):
- bug_id: D-NEW-1
- title: TopKGate applies topk to logits instead of post-softmax gates
- root_cause: torch.topk(logits, k) selects experts from raw logits, not softmax probabilities
- patterns_hit: []
- rationale: Source-level algorithmic bug; softmax is monotonic so topk indices may match between buggy and correct, only weights differ. No runtime pattern catches this; P3 won't trigger because gate weights are not cross-rank checked, P1 dtype is correct, P6 structural is fine.

**Example 5** (matches P2):
- bug_id: M-NEW-30
- title: aux_loss coefficient missing /tp_size when reducing across TP
- root_cause: aux_loss_coeff applied without dividing by TP world size, inflating gradient scale by tp factor after all-reduce
- patterns_hit: ["P2"]
- rationale: Cross-N-rank scaling missing 1/N factor. Classic P2 Scaling Consistency.

## INPUT (per bug, given to LLM)

```
### Bug ID
<bug_id>

### Framework
<framework>

### Source pool
<128_only | both | 295_only | 295_orphan>

### Title
<title>

### Category (13-class taxonomy, already determined)
<category>

### Root cause
<root_cause text>

### Invariant
<invariant text or null>

### Detection method
<detection_method text or null>

### Trigger conditions
<trigger_conditions list>
```

## OUTPUT (strict JSON, per bug)

```json
{
  "bug_id": "<id>",
  "patterns_hit": ["P3"],
  "patterns_uncertain": [],
  "rationale": "<30-200 chars; explain why these patterns AND why not the closest alternative>"
}
```

### Output rules

- `patterns_hit`: list of P1..P8 IDs that this bug matches. Empty array `[]` if no pattern.
- `patterns_uncertain`: optional list of patterns that are plausible but you're <60% confident. If you put a pattern here, do NOT also put it in `patterns_hit`.
- `rationale`: must explain primary choice and rule out the most-easily-confused alternative pattern.

### Self-check before output

- Did I consider all 8 patterns, not just the obvious one?
- If I picked multi-pattern, are they truly independent rules or is one subsumed by another?
- If empty list, is the bug really runtime-unobservable, or did I miss a pattern?
- Confidence: high (>80%) → patterns_hit; medium (60-80%) → patterns_hit + flag in rationale; low (<60%) → patterns_uncertain.
