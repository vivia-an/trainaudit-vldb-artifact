# RECON_S2 — Predicate Ablation Inventory

Read-only recon for SPEC v2 of the three-predicate ablation. Source of truth:
`trainaudit/trainaudit/rules/T*_*.py`, the rule registration in `base.py`, and
all `*.duckdb` stores under `benchmark/` (excluding `build/`).

The "π_topo signature" column quotes the literal code (or `none`) that gates
the rule on multi-rank topology evidence. The "π_precond signature" column
quotes the SQL `WHERE hookpoint ...` clause and any explicit Python-side
precondition guard (training-phase / framework-attribute filter). Ablation
recipes are described at code-line granularity — there is no central pruner.

---

## Section A — Per-rule predicate inventory (32 rules)

| rule_id | min_tier | π_topo signature | π_precond signature | how to ablate |
|---|---|---|---|---|
| T0-attention-head-uniformity | T0_PYTORCH | `none` | `WHERE hookpoint = 'attention.fwd.post'` | rewrite-WHERE (drop hookpoint filter → would scan all events) |
| T0-build-has-modules (file: T0_layer_count.py) | T0_PYTORCH | `none` | `WHERE hookpoint = 'build.snapshot'` | rewrite-WHERE |
| T0-checkpoint-preserve-rng | T0_PYTORCH | `none` | (a) `WHERE hookpoint = 'checkpoint.call'` (b) `WHERE hookpoint IN ('build.snapshot', 'module.fwd.post')` to detect Dropout (c) Python guard `if not has_dropout: return ... rule N/A` | rewrite-WHERE on both queries; delete `if not has_dropout` short-circuit |
| T0-clip-grad-bounded | T0_PYTORCH | `none` | `WHERE hookpoint = 'utils.clip_grad.post'` | rewrite-WHERE |
| T0-dtype-propagation | T0_PYTORCH | `none` | `WHERE hookpoint = 'module.fwd.post'` plus Python skip `if any(s in cls for s in _DTYPE_CHANGING): continue` (filters cast layers) | rewrite-WHERE + delete dtype-changing skip-list line |
| T0-evaluator-eval-mode | T0_PYTORCH | `none` | (a) Python early-return: `if cp_count > 0: return … rule passive` (skips when `torch.utils.checkpoint` is used) (b) `WHERE hookpoint = 'module.fwd.pre'` (c) `if len(bad) < 2: return passive` 2-event threshold | both: delete cp_count early-return AND drop the <2 threshold; rewrite-WHERE on the module.fwd.pre filter |
| T0-grad-norm-finite | T0_PYTORCH | `none` | `WHERE hookpoint = 'optim.step.pre'` | rewrite-WHERE |
| T0-initial-lr-present | T0_PYTORCH | `none` | (a) `WHERE hookpoint = 'scheduler.init'` (b) Python guard `if last_epoch == -1: continue` (skips fresh-start scheduler) | delete-guard-line on `last_epoch == -1` continue; rewrite-WHERE |
| T0-loss-per-token-dispersion | T0_PYTORCH | `none` | `WHERE hookpoint = 'loss.call'` plus Python skip `if median <= 0 or pt.get("all_ignored"): continue` | rewrite-WHERE + delete `all_ignored` skip |
| T0-loss-reduction-mode-correct | T0_PYTORCH | `none` | `WHERE hookpoint = 'loss.call'` plus Python guard `if red not in _OK_REDUCTIONS` (`{"mean"}` — this is the violation predicate, not π_precond; no precond beyond hookpoint) | rewrite-WHERE only (the `red not in` test is π_schema) |
| T0-module-grad-output-alive | T1_FW_METADATA (note: file starts `T0_…` but `min_tier=Tier.T1_FW_METADATA`) | `none` | (a) `WHERE hookpoint = 'module.bwd'` (b) Python skip `if cls not in _PARAM_MODULES: continue` (filters to Linear/Embedding/RMSNorm/LayerNorm/MoEMLP/Conv*/MoELinearRouter) | rewrite-WHERE + delete `_PARAM_MODULES` skip line |
| T0-no-nan-inf | T0_PYTORCH | `none` | `WHERE hookpoint IN ('module.fwd.pre', 'module.fwd.post', 'module.bwd', 'optim.step.pre', 'optim.step.post', 'comm.pre', 'comm.post', 'build.snapshot')` | rewrite-WHERE (drop IN list → scan everything) |
| T0-norm-output-unit-rms | T0_PYTORCH | `none` | (a) `WHERE hookpoint = 'module.fwd.post'` (b) Python guard `if not p.get("is_normalizer"): continue` | rewrite-WHERE + delete `is_normalizer` guard |
| T0-optim-lr-positive | T0_PYTORCH | `none` | `WHERE hookpoint = 'build.snapshot'` | rewrite-WHERE |
| T0-optim-step-counter-monotonic | T0_PYTORCH | `none` | `WHERE hookpoint = 'optim.step.post' ORDER BY event_id` + `if len(rows) < 2: passive` | rewrite-WHERE + delete the <2 threshold |
| T0-param-update-applied | T0_PYTORCH | `none` | `WHERE hookpoint = 'optim.step.post'` | rewrite-WHERE |
| T0-softmax-degenerate | T0_PYTORCH | `none` | Two queries: (a) `WHERE hookpoint = 'module.fwd.post'` + Python skip `if not any(h in cls for h in _ROUTER_NAME_HINTS): continue` (b) `WHERE hookpoint = 'functional.softmax'` | rewrite-WHERE on both + delete the `_ROUTER_NAME_HINTS` skip line |
| T0-token-id-in-vocab | T0_PYTORCH | `none` | `WHERE hookpoint = 'dataloader.batch'` (passive in Phase 1) | not-applicable (no events of this type captured in any current store) |
| T1-buffer-replica-cksum-equal | T1_FW_METADATA | `if entry.get("group_size", 1) <= 1: continue` (line 46) | `WHERE hookpoint = 'build.snapshot'` plus implicit filter that the loop iterates `cross_rank_buffer_cksums` (only present when at least one parallel state has >1 rank) | delete-guard-line (remove the `group_size <= 1` continue); precond is just the hookpoint |
| T1-comm-dtype-matches-training | T1_FW_METADATA | `none` directly; but rule is passive if no `training_dtype` is declared (a framework-config precond, not topology) | (a) `WHERE hookpoint = 'build.snapshot'` to discover declared dtype (b) `WHERE hookpoint = 'comm.post'` (c) Python skip `if op not in _GRADIENT_COMM_OPS: continue` (filters reduce/all_reduce/all_gather/all_to_all/reduce_scatter/all_gather_into_tensor/all_to_all_single) (d) early-return `if declared is None: passive` | rewrite-WHERE on `comm.post`; delete `_GRADIENT_COMM_OPS` skip; the declared-dtype guard is π_precond, removing it makes rule fire ambiguously |
| T1-expert-bias-fp32 | T1_FW_METADATA | `none` | (a) `WHERE hookpoint = 'module.fwd.post'` (b) Python guards `if not sem.get("is_router"): continue` and `if eb_dtype is None: continue` (c) early-return `if not n_router: passive` | rewrite-WHERE + delete `is_router` and `eb_dtype is None` guards |
| T1-fwd-output-block-uniformity | T1_FW_METADATA | `none` | (a) `WHERE hookpoint = 'module.fwd.post'` (b) skip `if cls in _SKIP_CLASSES: continue` (RMSNorm/LayerNorm/Identity/RotaryEmbedding/FSDPMoELinearRouter/MoELinearRouter/FSDPEmbedding/FSDPLMHead/TorchAttentionBackend) (c) skip `if bidx is None: continue` (only blocks.N.*) (d) `len(block_dict) < 4` and `len(recs) < 3` sample-size thresholds | rewrite-WHERE + delete `_SKIP_CLASSES` + delete bidx-required line + lower thresholds |
| T1-grad-flow-block-uniformity | T1_FW_METADATA | `none` | Same as `T1-fwd-output-block-uniformity` but `WHERE hookpoint = 'module.bwd'` and a slightly different `_SKIP_CLASSES` (adds Attention/TorchAttentionBackend, removes the FSDPEmbedding/LMHead) | same as above |
| T1-grad-replica-cksum-equal | T1_FW_METADATA | `if entry.get("group_size", 1) <= 1: continue` (line 47) | `WHERE hookpoint = 'optim.step.pre'`; iterates `cross_rank_grad_cksums` (only emitted when grad gather happened) | delete-guard-line (remove `group_size <= 1` continue) |
| T1-jitter-preserves-dtype | T1_FW_METADATA | `none` | `WHERE hookpoint = 'jitter.probe'` (probe is only installed when the Megatron router-jitter adapter is active — that is the precond) | rewrite-WHERE; ablating still requires probe events to exist (none in current single-GPU stores) |
| T1-layer-count-strict | T1_FW_METADATA | implicit: relies on `pipeline_model_parallel_size` from build.snapshot and rule passes when product matches; computes `actual_per_rank * pp_size` (pp_size > 1 is topology) but no explicit `if pp_size <= 1: continue` line | `WHERE hookpoint = 'build.snapshot'` + early-return `if declared_num_layers is None` + `if actual_per_rank is None` | rewrite-WHERE; the rule conflates π_topo into the predicate math, so "ablating" is mostly cosmetic |
| T1-multi-backward-per-step-fragile-config | T1_FW_METADATA | `none` directly (framework-config precondition) | (a) `WHERE hookpoint = 'module.fwd.pre' AND (json_extract(payload,'$.module_name') = 'null' OR = '""')` (b) `WHERE hookpoint = 'build.snapshot'` (c) Python `_is_fragile_config` requires DeepSpeed ZeRO-1/2 + offload + ga=1 — `violated=True` only if both multi-fwd AND fragile config; otherwise downgraded | rewrite-WHERE on fwd.pre; delete `_is_fragile_config` gate to "always violated when multi-fwd seen" |
| T1-process-group-size-correct | T1_FW_METADATA | `if int(declared) != int(actual) and int(declared) > 1` (line 57) — the `> 1` is a topology guard preventing FP on single-rank | `WHERE hookpoint = 'build.snapshot'` + Python skip `if declared is None or actual is None: continue` | delete-guard-line on the `> 1` clause (keep `!=` check); rewrite-WHERE if desired |
| T1-replica-cksum-equal | T1_FW_METADATA | `if entry.get("group_size", 1) <= 1: continue` (line 40) | `WHERE hookpoint = 'build.snapshot'`; iterates `cross_rank_cksums` | delete-guard-line (remove the `group_size <= 1` continue) |
| T1-residual-stream-preserved | T1_FW_METADATA | `none` | Primary: `WHERE hookpoint = 'residual.probe'`. Fallback: `WHERE hookpoint IN ('module.fwd.pre','module.fwd.post') ORDER BY event_id` + `sem.get("is_residual_block")` + `"Norm" in cls` substring guard | rewrite-WHERE + delete `is_residual_block` and `"Norm" in cls` guards |
| T1-router-has-calculate-per-token-loss (file: T1_router_attribute.py) | T1_FW_METADATA | `none` | (a) `WHERE hookpoint = 'build.snapshot'` (b) Python early-return `if not feature_enabled: rule N/A` keyed on `framework_invariants.megatron.calculate_per_token_loss` (c) `WHERE hookpoint = 'module.fwd.post'` (d) Python skip `if not sem.get("is_router"): continue` | both: delete `feature_enabled` early-return + delete `is_router` skip; rewrite-WHERE on module.fwd.post |
| T1-sqrt-decay-front-loaded | T1_FW_METADATA | `none` | `WHERE hookpoint = 'decay.probe' ORDER BY event_id` + `len(rows) < 5` and `len(samples) < 5` thresholds | rewrite-WHERE + lower thresholds; relies on adapter probe events |

Notes:
- Several rules wrap the precondition in a `if … return RuleResult(passive)` shape rather than a SQL `WHERE`. Counting both styles, **every** rule has at least one Python-side precondition beyond the hookpoint filter.
- `T0_module_grad_output_alive.py` is named `T0_…` but its decorator declares `min_tier=Tier.T1_FW_METADATA`; treat this as a T1 rule for tier-gating purposes (it is the only such inconsistency).

---

## Section B — Aggregate counts

```
n_T0_rules:                                                 18
n_T1_rules:                                                 14
n_rules_with_pi_topo (explicit group_size or *_size guard): 4
    (T1-buffer-replica-cksum-equal,
     T1-grad-replica-cksum-equal,
     T1-replica-cksum-equal,
     T1-process-group-size-correct)
n_rules_with_pi_precond:                                   32
    (every rule has at least a hookpoint filter; 22 also
     have an explicit Python-side phase/framework/role guard)
n_rules_where_no_topo_ablation_can_fire_on_single_rank_data: 28
    (all 18 T0 + 10 of 14 T1 — they have no π_topo to remove.
     The 4 π_topo-bearing rules still need group_size > 1 in
     payload to fire after the guard is removed.)
```

Filed correction to the SPEC's "18 T0" claim: file count matches but `T0_module_grad_output_alive.py` is internally T1. By file prefix: 18 T0_*.py + 14 T1_*.py = 32.

---

## Section C — Trace data inventory

Total: **120** `.duckdb` files across 6 families, ~9.2 GB.

| family | n_stores | sample bug_id / config | is_multi_rank? (any `group_size > 1`) | total bytes |
|---|---|---|---|---|
| benchmark/sweep/_runs | 14 | `dense_190M` (parameter sweep, clean baselines) | **No** — checked `dense_190M`: max `group_size=1`, no `*_size_actual` declared. All sweeps are single-rank. | 663,920,640 (~633 MB) |
| benchmark/fault_injection/_runs | 15 | `FAULT-001-freeze-block0` (synthetic 13-fault injection set) | **No** — checked `FAULT-001`: max `group_size=1`. All injection runs are single-rank. | 357,748,736 (~341 MB) |
| benchmark/dd_candidate1_quality | 2 | `ab_buggy` / `ab_fixed` (DD A/B run pair) | **No** — checked `ab_buggy`: max `group_size=1`. Single-rank. | 277,897,216 (~265 MB) |
| benchmark/eval/hunt_log/novel_hunt | 29 | `olmo_core_baseline`, `olmo_core_moe_ep2`, `olmo_core_tp2`, `deepspeed_zero2`, … | **No** for those with build.snapshot. Sampled `olmo_core_baseline`, `olmo_core_olmoe`, `olmo_core_moe_ep2/rank0`, `olmo_core_moe_ep2/rank1`, `olmo_core_dense_actckpt_2rank/{0,1}`, `deepspeed_zero2/{0,1}`, `deepspeed_zero3` → all `max_group_size=1`, no `*_size_actual` populated. **Empty traces** (0 events, no schema): `olmo_core_tp2/rank{0,1}`, `olmo_core_moe_ep2_actckpt/rank{0,1}`. These are the only `tp2` / `ep2-actckpt` directories — they look like failed/aborted hunts and cannot be used. | 575,762,432 (~549 MB) |
| benchmark/eval/rebuttal_v1/C1_overhead_gpu/_runs | 32 | `megatron_gpt_tiny_TP2PP1_sync_r1` (8 ranks × 4 configs) | **Yes for Megatron TP2**, No for OLMo2 FSDP. All 16 `megatron_gpt_tiny_TP2PP1_{async,sync}_r1/trace_rank{0..7}.duckdb` files have `cross_rank_cksums` with **max `group_size=2`, 50 of ~99 entries with `group_size > 1`** (the TP-sharded params). All 16 `olmo2_190M_fsdp_ep1_{async,sync}_r1/*` files have `max_group_size=1` (FSDP with sharded params not labelled as replica). None of these stores have `cross_rank_buffer_cksums` or `cross_rank_grad_cksums` with `group_size > 1`. | 4,533,911,552 (~4.22 GB) |
| benchmark/eval/rebuttal_v1/E_clean_run_fp_audit/_runs | 28 | `dp8_ddp_r1` (8-rank DDP), `dp8_fsdp_r1`, `dp2_fsdp_long_r1`, `ep2_dp4_moe_r1`, `smoke_r0` | **No** — sampled every directory. `dp8_ddp_r1/*`, `dp8_fsdp_r1/*`, `dp2_fsdp_long_r1/*`, `smoke_r0/*`: max `group_size=1` (the DDP/FSDP adapters do not populate replica labels for the cross-rank gather). **Empty traces** (0 events): all 8 ranks of `ep2_dp4_moe_r1/*`. | 2,744,205,312 (~2.55 GB) |

Read pattern used: `duckdb.connect(path, read_only=True)` then `SELECT payload FROM events WHERE hookpoint='build.snapshot' LIMIT 1`, then `json.loads(payload)`, then iterate `cross_rank_cksums` / `cross_rank_buffer_cksums` / (separately for `optim.step.pre`) `cross_rank_grad_cksums`. No locked stores. The empty traces are real (`SELECT COUNT(*) FROM events = 0`), not access errors.

The mismatch with the original SPEC: the assumed path `benchmark/bugs/<bug_id>/runs/` does not contain any `.duckdb` files (`benchmark/bugs/` exists but holds source artefacts, not trace stores).

---

## Section D — Multi-rank data we can use for π_topo ablation

**Existing usable data:** Only one configuration in the entire tree has `group_size > 1` evidence — `benchmark/eval/rebuttal_v1/C1_overhead_gpu/_runs/megatron_gpt_tiny_TP2PP1_{async,sync}_r1`. Each provides 8 rank-files × 2 (async/sync) = 16 stores, 50 of ~99 `cross_rank_cksums` entries have `group_size = 2`. These are **clean runs**, not buggy runs, so they only exercise the FP-rate side of the π_topo ablation, not the detection side.

What this means for the π_topo half of S2:

- **For detection rate:** there is currently **zero on-disk multi-rank buggy data**. Every D1 bug case ran single-GPU, so removing the `group_size <= 1` guard cannot change detection on existing traces — the `cross_rank_cksums` list is either empty or all size-1, so the loop body never executes regardless of the guard.
- **For FP rate:** the C1 TP2 clean runs **can** demonstrate the impact. With the guard in place, `T1-replica-cksum-equal` skips all 50 sharded entries (they have `all_equal=False` by construction — different shards on different TP ranks). Remove the guard, and the rule will fire on every sharded param → 50 violations per build.snapshot per rank × 16 ranks → on the order of hundreds-to-thousands of FPs on a clean run. This is the **only existing experiment** that lets us numerically demonstrate the FP-rate explosion claim from `main.tex:917` (Case 2 router 57/492 FP).

**Cheapest way to produce buggy multi-rank data:** re-run B1 (or B2) on a single node with `torchrun --nproc_per_node=2` and Megatron TP=2 or DP=2. Both bugs (B1 routing-RNG divergence, B2 LinearWithFrozenWeight backward all-reduce miss) require TP ≥ 2 to express. None of the existing trainaudit `_runs/` directories have such a configuration for a buggy case. Estimated cost: ~30 minutes GPU time per bug × 2 phases × 2 bugs ≈ 2 GPU-hours.

**Alternative without re-running:** synthesize a trace augmentation. Take an existing single-rank duckdb (e.g. `FAULT-001`) and post-process its `build.snapshot` payloads to inject synthetic `cross_rank_cksums` entries with `group_size=2` and `all_equal=False`. This loses fidelity but lets the FP-rate ablation be run without GPU. The mining/predicate logic doesn't care where the JSON came from.

**For π_precond ablation:** the SPEC's "precond all True" framing maps to removing the **hookpoint WHERE filter** and the **Python-side phase / role guards**. I skimmed 5 representative rules:

1. `T0-clip-grad-bounded` — `WHERE hookpoint='utils.clip_grad.post'`. Drop the WHERE and the rule reads every event in the store (~30k–160k events vs ~10 clip events). It will iterate everything and try to extract `max_norm`/`post_norm`; non-clip events return `None` and are skipped. Effect: massive read amplification, **no FP** (the extracted fields don't exist on other events). This rule's hookpoint filter is purely a perf optimisation, not a semantic guard.

2. `T0-checkpoint-preserve-rng` — the `if not has_dropout: return passive` Python guard. Removing it causes the rule to FP on every codebase that uses `torch.utils.checkpoint` without dropout (legitimate — there is no RNG-replay correctness issue when nothing draws random numbers).

3. `T0-evaluator-eval-mode` — the `if cp_count > 0: return passive` guard. Removing it FPs on every model that uses `torch.utils.checkpoint` reentrantly (the recompute forward runs under `no_grad` while `training=True`, which is correct framework behaviour, not a user bug).

4. `T0-initial-lr-present` — the `if last_epoch == -1: continue` guard. Removing it FPs on every fresh-start training run (PyTorch sets `initial_lr` on first scheduler `step()`, not at `__init__` — the check is meaningful only after resume).

5. `T1-router-has-calculate-per-token-loss` — the `if not feature_enabled: return N/A` guard. Removing it FPs on every Megatron run that doesn't pass `--calculate-per-token-loss` (most of them — the attribute is genuinely absent by design).

**Conclusion:** the π_precond ablation gives a much more direct demonstration of FP-rate explosion than π_topo does on current data. Rules 2–5 above each generate FPs on **all** clean runs in `benchmark/sweep/_runs/` (14 runs, all single-GPU, all using checkpoint or fresh schedulers). That alone is a publishable headline number without needing any new GPU runs.

---

## Section E — Anomalies / things that surprised me

- **No central pruner.** The SPEC v1 assumed a `runtime pruner` toggleable by flag. In reality π_topo is hand-written as a one-line `if entry.get("group_size", 1) <= 1: continue` inside exactly 3 rules' inner loops (replica-cksum, grad-replica-cksum, buffer-replica-cksum), plus one `if … and int(declared) > 1` inside `T1-process-group-size-correct`. There is no central registry, no decorator, no shared helper — ablation must edit each rule file individually (which the SPEC says is forbidden via the "git diff in rules/ should be 0 lines" check).
- **π_precond is mostly SQL hookpoint filters, not Python phase guards.** Of 32 rules, 22 have non-trivial Python preconditions (training-phase, framework-attribute, role-label, sample-size threshold). The rest are pure hookpoint filters whose removal is a perf hit but not a FP source.
- **The 17 D1 bug-case stores do not exist.** `benchmark/bugs/<bug_id>/runs/` is empty of `.duckdb`. SPEC v2 needs to either (a) drop the per-bug ablation framing and use the rebuttal_v1 C1 TP2 traces as the single multi-rank exemplar, (b) re-run a subset on GPU, or (c) shift to FP-rate-on-clean-runs as the headline metric (which gives a cleaner story).
- **`T0_module_grad_output_alive.py` declares `min_tier=Tier.T1_FW_METADATA`** — a file-name vs decorator mismatch. Treat as T1 for tier filtering.
- **`olmo_core_tp2/*`, `olmo_core_moe_ep2_actckpt/*`, `ep2_dp4_moe_r1/*` are empty traces** (0 events, no schema). These look like aborted hunts. The only directories with usable multi-rank topology data are `rebuttal_v1/C1_overhead_gpu/_runs/megatron_gpt_tiny_TP2PP1_*`.
- **OLMo2/OLMo-core `_runs` populate `cross_rank_cksums` with all `group_size=1` even when running 2-rank/8-rank**. The FSDP/DDP adapter does not label any param as a replica (every entry is treated as "not in a multi-rank replica group"). This is a real adapter gap: the only existing trace data exercising the topology code path is the Megatron TP2 run.
- **`*_size_actual` / `*_size_declared` fields used by `T1-process-group-size-correct` are not populated by any current adapter** — sampled across all 6 families, every `framework_invariants` block produced empty `sizes={}`. The rule is effectively unreachable on current data. Re-running with adapter fixes would be needed to exercise it.
- **`T1-jitter-preserves-dtype`, `T1-residual-stream-preserved`, `T1-sqrt-decay-front-loaded`** depend on adapter-installed probes (`jitter.probe`, `residual.probe`, `decay.probe`). These probes only fire in their target frameworks. The fallback path in `T1-residual-stream-preserved` is the only one with non-probe coverage.
- **Total trace volume is 9.2 GB / 120 files**, dominated by the 8-rank rebuttal runs. A full re-run of trainaudit over everything for an ablation sweep is feasible CPU-side (no GPU needed for replay) in well under an hour.
- **The `T1-multi-backward-per-step-fragile-config` rule has a built-in `violated=False unless fragile config` predicate that *is itself a π_precond layer*** (downgrades detection rather than skipping). Removing it would convert every multi-forward step (RLHF, gradient accumulation) into a "violation" — a clean demonstration of why precond matters.
