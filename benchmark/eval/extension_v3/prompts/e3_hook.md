# E3 Prompt: Lifecycle Hook Coverage Annotation

> Output: `hook_coverage_392.json` per-bug entry with `hooks_observable` + `earliest_observable`.
> Reference: `28_392_extension_brief.md` §3.3, `main_cn.tex` :557 (5 hookpoints).

## TASK

For the given bug, identify (a) the **earliest** lifecycle hook at which the bug becomes observable in runtime trace, and (b) **all** hooks where the bug remains observable. If the bug is not observable at any hook, output `unobservable`.

## 5 LIFECYCLE HOOKS (in temporal order within one training step)

| Hook | Definition | Typical bugs caught |
|---|---|---|
| **before-forward** | Just before model.forward() entry. Param state from previous step's optimizer.step. | Param init / dtype / RNG state / model class structure |
| **after-forward** | After forward pass completes (incl. sub-module forward exits). Activations available. | Forward computation correctness, MoE routing, residual stream, loss formation, function-call counts in forward |
| **main-grad-in-backward** | During backward pass, when main_grad accumulates. | Gradient accumulation logic, grad reduction op, dtype mismatch in collective |
| **after-backward** | After backward completes, grad sync / clipping done. | Final grad norm anomalies, grad cross-rank consistency |
| **before-optimizer** | Just before optimizer.step. Optimizer state visible. | Counter consistency (e.g., micro_step_id), optimizer state corruption, lr schedule |

(The brief mentions auxiliary taps `checkpoint.{save,load}`, `distributed.all_reduce`, `build.snapshot`. For E3 we focus on the **5 main lifecycle hooks**. If a bug is only observable at checkpoint save/load, mark `earliest_observable: "checkpoint_save"` or `"checkpoint_load"` as a separate value.)

## DECISION RULES

1. **Earliest = first hook within a single training step where buggy run produces a different trace from clean run**.
2. A bug may be observable at multiple hooks (e.g., a corrupted parameter is visible at every subsequent hook until it's overwritten).
3. **Resume-time bugs** (checkpoint load): earliest is `before-forward` of the FIRST step after load (the first time hook fires post-load).
4. **Build-time bugs** (wrong layer count, wrong module class): earliest is `build.snapshot` (a separate value, not in the 5 main hooks). If the model wouldn't even initialize, mark `earliest_observable: "build"`.
5. **Unobservable**: if the bug requires multi-step convergence regression OR source-only inspection.

### Common patterns

| Bug type | Typical earliest_observable |
|---|---|
| Param init (dtype, value, RNG) | before-forward |
| MoE routing / forward function-call count | after-forward |
| Loss formation | after-forward |
| Grad accumulation / clipping | after-backward |
| Grad reduction op or dtype | main-grad-in-backward (or after-backward) |
| Optimizer state corruption | before-optimizer |
| Counter (micro_step_id, accumulation count) | before-optimizer |
| LR scheduler | before-optimizer |
| Checkpoint save | checkpoint_save |
| Checkpoint load | checkpoint_load (or before-forward of next step) |
| Build-time (wrong layer count) | build |

### Few-shot examples

**Example 1** (after-forward earliest):
- bug: B1 SwitchMLP router weight diverged across TP ranks
- After init, divergence is visible at param state. But `before-forward` of the first step is when we'd first observe diverged param via cross-rank cksum. So **earliest = before-forward**.
- All hooks observable (the param stays diverged): all 5 hooks.

Actually wait — for B1, the bug is visible at every hook from before-forward onward, because the param doesn't change unless optimizer step runs. So:
- earliest_observable: "before-forward"
- hooks_observable: ["before-forward", "after-forward", "after-backward", "before-optimizer"]

**Example 2** (after-forward only):
- bug: M-NEW-1 MoE aux-loss sigmoid in bf16
- The bug manifests inside forward (sigmoid op input dtype). Before-forward param state is correct. After-forward, the loss/aux-loss value reflects the dtype loss.
- earliest_observable: "after-forward"
- hooks_observable: ["after-forward"]

**Example 3** (before-optimizer):
- bug: B10 micro_step_id counter off-by-one
- Counter only mismatches at the boundary just before optimizer.step.
- earliest_observable: "before-optimizer"
- hooks_observable: ["before-optimizer"]

**Example 4** (after-backward):
- bug: gradient clipping threshold wrong, causing scale anomaly
- Grad anomaly observable post-backward. Param state still correct.
- earliest_observable: "after-backward"
- hooks_observable: ["after-backward", "before-optimizer"]

**Example 5** (checkpoint_load):
- bug: ZeRO checkpoint load corrupts optimizer state
- Bug surfaces during load. Earliest observation is checkpoint_load tap. After load, param state is corrupted, so subsequent hooks also see it.
- earliest_observable: "checkpoint_load"
- hooks_observable: ["checkpoint_load", "before-forward", "after-forward", ...]

**Example 6** (unobservable):
- bug: D-NEW-1 TopKGate topk on logits not softmax
- No hook sees the difference (softmax monotonic, indices match, weights differ subtly without changing visible state).
- earliest_observable: "unobservable"
- hooks_observable: []

**Example 7** (build):
- bug: wrong layer count due to config mismatch detected at build
- Bug visible at build.snapshot before any training step.
- earliest_observable: "build"
- hooks_observable: ["build"]

## INPUT FORMAT (per bug)

```
### Bug ID, Framework, Source pool
### Title
### Category
### Root cause
### Invariant
### Detection method
### Check stage (if available, from 128 pool)
### Trigger conditions
```

## OUTPUT FORMAT (strict JSON)

```json
{
  "bug_id": "<id>",
  "earliest_observable": "before-forward" | "after-forward" | "main-grad-in-backward" | "after-backward" | "before-optimizer" | "checkpoint_save" | "checkpoint_load" | "build" | "unobservable",
  "hooks_observable": ["after-forward", "after-backward"],
  "rationale": "<30-200 chars; explain earliest choice>"
}
```

### Output rules

- `earliest_observable` is one of the listed strings. **5 main hooks** + 3 auxiliary (checkpoint_save, checkpoint_load, build) + `unobservable`.
- `hooks_observable` is the full set of hooks where the bug remains observable (after the earliest one). Use only the 5 main hooks here, or auxiliary if relevant. Empty array if unobservable.
- For **earliest counting** (the 48/128 = 37.5% paper number), only the earliest hook counts. So `after-forward` earliest count = number of bugs whose `earliest_observable == "after-forward"`.

### Self-check

- Did I distinguish "earliest first hits" vs "all hooks where remains observable"?
- For checkpoint-load bugs, did I correctly mark `checkpoint_load` not `before-forward`?
- For function-call bugs (MoE aux-loss tracker), is the earliest observation really at `after-forward`?
- Unobservable means NO hook helps — am I sure?
