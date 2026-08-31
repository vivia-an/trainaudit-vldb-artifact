# Case 2 Walkthrough: What `π_topo` Buys Us on the SwitchMLP Router

Companion to Case 2 in `main.tex` (lines 916-917 and 389-390). The paper claims that dropping
`π_topo` from the SwitchMLP router-weight check fires on **57 of 492** parameter records on a
clean TP=2 run, every one a correctly-sharded `query_key_value.weight`. This document grounds
that number in the rule code, walks through which records fall in which bucket, and explains
why the same structure repeats across several other T1 rules.

---

## 1. The setup

The rule under examination is `T1-replica-cksum-equal`
(`trainaudit/trainaudit/rules/T1_replica_cksum_equal.py`). It reads one row per training step
from the `events` table where `hookpoint = 'build.snapshot'`, decodes the JSON payload, and
walks the `cross_rank_cksums` list. Each entry is a per-parameter record built by
`gather_param_cksums` (`trainaudit/trainaudit/core_trace/cross_rank.py`) and carries the
fields `{name, local_cksum, gathered_cksums, all_equal, group_size}`. `π_topo` lives in a
single line inside the rule loop:

```python
for entry in p.get("cross_rank_cksums") or []:
    if entry.get("group_size", 1) <= 1:
        continue  # nothing to compare
    n_checked += 1
    if entry.get("all_equal") is False:
        bad.append({...})
```

The trace this case study refers to is the `B1 / M-005` reproduction
(`benchmark/bugs/M-005`): a SwitchMLP run where the router-weight init goes through the
default CUDA RNG instead of `get_cuda_rng_tracker().fork(...)`, so router weights diverge
silently across TP ranks. The relevant configs are TP=2 and a paired clean replay.

---

## 2. What `π_topo` filters out

`group_size` in each record is set by `gather_param_cksums`: for every parameter, the
collector asks `replica_group_id_fn(name, p)` which process group the parameter is
*supposed* to be replicated over, and stores `dist.get_world_size(group=grp)`. For TP-only
runs of the SwitchMLP module, parameter records fall into three categories.

**TP-sharded matrices (`query_key_value.weight`, `dense.weight`, expert `w1`/`w2` columns).**
These are intentionally column- or row-partitioned across TP ranks. Their replica group is
the single owning rank, so `group_size = 1`. The guard `if group_size <= 1: continue` skips
them. Correct: there is no rank pair that should hold the same bytes.

**TP-replicated parameters (`router.weight`, biases, layer-norm `weight`/`bias`).** The MoE
router is the canonical example: every TP rank must hold an identical copy so that token-to-
expert routing is deterministic. Their replica group is the full TP group, so
`group_size = TP_size = 2`. The guard lets them through and the cross-rank gather compares
checksums. Correct: this is exactly where a divergence must be reported (the M-005 bug).

**Fully sharded across both axes (would appear under FSDP/HSDP).** Pure TP=2 with no FSDP
does not exercise this case, but the rule handles it the same way as category 1: the
collector returns `group_size = 1` and the guard skips. Correct: no replica relation to
check.

The 492 vs. 57 split reported in the paper has the following structure: of the 492 total
parameter records emitted at `build.snapshot` time across the run (summed over the recorded
build-snapshot events), 57 belong to category 1 (TP-sharded `query_key_value.weight` and its
siblings) and would be flagged by a naïve unscoped equality check. With `π_topo` in place
those 57 records carry `group_size = 1` and the rule skips them before the equality test
ever runs. The remaining 435 records are either category 2 (replicated, `group_size = 2`,
checked against `all_equal == True`) or duplicates of the same physical parameter across
build-snapshot events. On the clean replay all category-2 records pass; on the buggy replay
the `router.weight` entries flip `all_equal` to `False` and the rule fires with the correct
diagnostic.

Concretely: the FP-eliminating effect of `π_topo` is the difference between "treat every
parameter as if it must be cross-rank equal" and "only compare records whose replica group
has more than one member". The latter is what the one-line guard implements.

---

## 3. What happens without `π_topo`

Remove the guard (delete `if entry.get("group_size", 1) <= 1: continue`) and the rule loop
collapses to "for every parameter record, compare checksums across whatever group was
gathered". For category-1 records the collector never ran `dist.all_gather` (because
`world == 1` short-circuits in `_gather_tensor_cksums_for_params`), so `gathered_cksums` is
`None` and `all_equal` is `None`. A naïve patch would either (a) drop the
`if all_equal is False` test in favour of `if all_equal is not True`, or (b) gather over
the world group regardless of replica intent. Either path yields the same outcome: every
TP-sharded weight is now compared bit-for-bit across ranks and, by construction, the bytes
differ. The rule fires once per category-1 record on a perfectly clean run; the paper
quantifies that as **57 false positives**, with each FP carrying a `query_key_value.weight`
name in its evidence sample. With `π_topo` in place, the same run produces 0 false
positives while still catching the M-005 router divergence.

This is the directly checkable cost of the predicate: one `if` line removes 57 false alarms
on a single clean run and changes nothing about true-positive coverage on M-005.

---

## 4. Why this generalizes

The shape of the fix---collector emits `group_size` from a topology-aware
`replica_group_id_fn`, rule skips records with `group_size <= 1`---is reused verbatim by
every other cross-rank checksum rule in the catalog:

- `T1-buffer-replica-cksum-equal` (`T1_buffer_replica_cksum_equal.py`) gates the cross-rank
  comparison of replicated module buffers (e.g. MoE `expert_bias`) with the same `if
  entry.get("group_size", 1) <= 1: continue` line, and reports the skipped count
  explicitly: `skipped {n_buffers_total - n_checked} buffers with group_size <= 1`. Without
  the guard, every TP-sharded buffer would FP on a clean run for the same reason as the
  parameter rule.

- `T1-grad-replica-cksum-equal` (`T1_grad_replica_cksum_equal.py`) applies the identical
  guard to `cross_rank_grad_cksums` at `optim.step.pre`. The B2 bug (missing input-gradient
  all-reduce in `LinearWithFrozenWeight`) is detected exactly because the rule still checks
  the entries with `group_size > 1`; dropping `π_topo` here would flag every TP-sharded
  weight's gradient on every step, multiplying FPs by the number of optim steps.

- `T1-fwd-output-block-uniformity` (`T1_fwd_output_block_uniformity.py`) is a different
  shape---it compares per-block forward-output L2 norms across blocks of the same module
  class---but its `_SKIP_CLASSES` set (`RMSNorm`, `LayerNorm`, `Identity`,
  `RotaryEmbedding`, `FSDPMoELinearRouter`, ...) plays the same role: pre-filtering
  modules whose cross-block comparison is semantically meaningless. The structural pattern
  is identical: encode the topology/role constraint in metadata, let the rule body assume
  every surviving record is comparable.

The pattern is general: any rule whose predicate is "some relation must hold across a set
of ranks/blocks/replicas" requires a topology-aware filter to define what *the set* is.
Without it, the rule is forced to either over-fire (treat all records as comparable) or
silently miss bugs (compare nothing). `π_topo` is the place this filter lives.

---

## 5. Numbers that need verification

The following claims in this document are quoted from `main.tex` (lines 304, 390, 416, 917)
or extrapolated for narrative flow. I could not independently verify them from the trace
data under `benchmark/`---no `events`-table dump or per-run `cross_rank_cksums` summary was
found under `benchmark/bugs/M-005/runs/*` or in `benchmark/eval/*.md`. Please cross-check
before camera-ready:

1. **57 false positives** on the clean TP=2 SwitchMLP run. Quoted from the paper; no
   directly checkable artifact (e.g. a `bad`-list dump or rule log) was located.
2. **492 total parameter records** across the build-snapshot events. Same provenance as
   above. Whether 492 counts unique parameter names or `(event, name)` pairs across all
   recorded build-snapshot events is not pinned down by the rule code alone---the rule
   loops over `(event_id, entry)` pairs.
3. **The 57 records are all `query_key_value.weight`** (paper text). Plausible from the
   Megatron SwitchMLP module layout, but the actual category-1 set on the M-005 reproduction
   could also include other TP-column-parallel matrices (e.g. `dense_h_to_4h.weight`,
   `dense_4h_to_h.weight` if present in the configured architecture).
4. **The 435 = 492 − 57 remaining records are category-2 replicated parameters** under
   `group_size = 2` (TP=2). This is my decomposition for narrative purposes; the true
   split between category 2 and duplicate `(event, name)` rows across build-snapshot events
   was not extracted from the trace.
5. **TP=2 configuration** for the clean run. The bug manifest (`config.json` for M-005)
   lists `gpu_needed: 2` and `tensor_model_parallel_size > 1` as the trigger condition,
   consistent with TP=2, but the exact `(TP, DP, PP, EP)` of the run that produced the
   492 record count was not located.
6. **The 7% loss-gap and the "non-zero from the outset" parameter divergence** quoted in
   the Fig. \ref{fig:case-studies} caption (Case 2). Not checked against
   `benchmark/bugs/M-005/runs/*/loss_curve.jsonl` in this pass.

Items 1, 2, 5 should be re-derived by re-running `T1-replica-cksum-equal` against the
recorded `events` DB for an M-005 clean replay and printing `n_checked`, `len(bad)`, and
the `name` field of the skipped records (the `n_buffers_total - n_checked` accounting in
the sibling buffer rule already does this kind of bookkeeping and can be adapted in
~5 lines).
