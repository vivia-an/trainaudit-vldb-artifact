# Miner Agent Prompt Extension — P9-P16 Pattern Hints (Spec)

> Per brief 32 §2.2 Phase 1: extension to Training Expert / Aggregation Expert / Coordinate Agent.
> This file is the **prompt-side spec** to inject into paper §5 Invariant Miner FSM.
> Implementation lives in paper repo's `agents/{training_expert,aggregation_expert,coordinate_agent}/`.

## Instructions to paper miner team

For each P9-P16 below, inject the hint into **Training Expert** (which proposes candidate invariants per pattern). Reuse the prompt skeleton of the closest existing pattern (mapping in brief §2.3).

---

## P9 — Init Distribution Consistency (Type C, build-time static assert)

**Skeleton**: copy from P6 Structural Integrity.

**Hint to add**:
> "When mining for P9, scan `model.named_parameters()` and look for params with declared `init_dist` (normal_, uniform_, kaiming_, xavier_, normal init in nn.init). The candidate invariant is `param.std() ∈ [declared_std × 0.5, declared_std × 1.5]` and `|param.mean()| < declared_std × 0.1`. Scope σ: `param.requires_grad and declared_init_dist != None`. Precond ρ: `step == 0 and framework_init_done == True`."

**Anchor bugs (root cause + commit diff)**:
- O-NEW-12 OLMo SwiGLU init wrong fan
- O-NEW-22 muP readout zero init violation
- OC-NEW-7 olmo-core MoE expert init scaling

**Source-code检索 hint (per framework)**:
- Megatron-LM: search `_initialize_affine_weight_gpu`, `tensor_parallel.layers`, `LayerNorm.reset_parameters`
- DeepSpeed: search `init_optim`, `Init.weight`
- OLMo: search `mitchell_init`, `mup_*` modules
- OLMo-core: search `Embedding.reset_parameters`, `LMHead.reset_parameters`

---

## P10 — Config-Implied Coupling (Type C, build-time)

**Skeleton**: copy from P6.

**Hint**:
> "When mining for P10, scan `config` object for boolean/enum flags (e.g. `zero_stage`, `mixed_precision`, `tp_size`, `use_fsdp`, `mup_base_width`). For each flag value v, derive the implication v ⟹ X(model_state). Candidate invariant: assert X holds when v is set. Scope σ: `flag.value == v` (specific). Precond ρ: `build phase + flag is read by framework`."

**Anchor bugs**: D-NEW-12 (zero_stage missed branch), M-NEW-11 (config-derived layer count), CC1-style.

**Source검색**: search `config.zero_stage`, `if cfg.use_*`, `assert config.*` chains.

---

## P11 — Position Encoding & Document Boundary Integrity (Type B, forward hook)

**Skeleton**: copy from P1 Dtype Preservation (forward hook).

**Hint**:
> "When mining for P11, look for forward functions accepting `cu_doc_lens` / `cu_seqlens` / `attention_mask`. The candidate invariant: when packed-doc mode is active, position tensor must reset to 0 at each `cu_doc_lens[k]` boundary. Scope σ: `module is RoPE/PositionEmbedding`. Precond ρ: `cu_doc_lens used in forward`."

**Anchor bugs**: OC-NEW-22 packed-doc rope, OC-NEW-13 cu_doc_lens not threaded.

**Source검색**: `apply_rotary_pos_emb`, `cu_doc_lens`, `position_ids`, `RotaryEmbedding`.

---

## P12 — Algorithm Variant / Formula Equivalence (Type B, forward hook)

**Skeleton**: copy from P2 Scaling Consistency.

**Hint**:
> "When mining for P12, look for fused vs unfused variants (FlashAttn vs eager attn, fused norm vs unfused). Candidate invariant: `max |fused(x) - reference(x)| / |reference(x)| < ε` for ε ∈ {1e-5 fp32, 1e-3 bf16}. Scope σ: `module has multiple impl branches`. Precond ρ: `branch_active = fused`."

**Anchor bugs**: O-NEW-49 fused-vs-ref, O-NEW-22 muP-vs-sp-base-width.

---

## P13 — Tensor Aliasing & Stale State (Type B, post-mutation hook)

**Skeleton**: copy from P5 State Restoration.

**Hint**:
> "When mining for P13, look for cached tensors / norms / view-shared pairs. Candidate invariant: when `cache.data_ptr() == underlying_param.data_ptr()`, after `param.data.copy_(...)` or `optim.step()`, cache must be invalidated/refreshed. Scope σ: `cache linked to param via view/share`. Precond ρ: `after parameter mutation`."

**Anchor bugs**: M-NEW-4 no-op cast aliasing, M-NEW-52 detach-required-on-view.

---

## P14 — Sharded State Completeness (Type A, trace SQL)

**Skeleton**: copy from P3 Cross-Rank Replication.

**Hint**:
> "When mining for P14, look for save/load functions emitting per-rank files. Candidate invariant: `set(saved_files matching mp_rank_NN_*) == {f'mp_rank_{r:02d}_*' for r in [0, tp_size)}`. Scope σ: `save_phase active`. Precond ρ: `tp_size > 1 OR ep_size > 1`."

**Anchor bugs**: D-022 ZeRO save short-circuit, D-043 MoE TP missing rank, M-042 partial reduction.

---

## P15 — Counter Width Adequacy (Type B, counter check)

**Skeleton**: copy from P4 Invocation Frequency.

**Hint**:
> "When mining for P15, scan all int counters in optimizer/dataloader/checkpoint state. Candidate invariant: `counter.dtype.itemsize >= ceil(log2(max_steps + safety) / 8)`. Scope σ: `counter is monotonically incremented per step`. Precond ρ: `max_steps known`."

**Anchor bugs**: D-006 int32 overflow in FusedAdam, M-027 sample_idx overflow.

---

## P16 — Loss Component Normalization (Type B, post-loss hook)

**Skeleton**: copy from P2 Scaling Consistency.

**Hint**:
> "When mining for P16, look for compound loss = sum(component_i / divisor_i). Candidate invariant: `divisor_i == granularity_i.cardinality()`, where granularity ∈ {token, sample, micro_batch, expert_pair}. Scope σ: `component declared as token-level/sample-level/etc.`. Precond ρ: `forward post-step + loss computed`."

**Anchor bugs**: OC-NEW-42 BLT distill /num_micro_batches, OC-NEW-58 weighted-sum bypass.

---

## Implementation status

| Pattern | Skeleton ready | Anchor bugs ready | Source hint ready |
|---------|---------------|-------------------|-------------------|
| P9      | ✓ from P6     | ✓ 3 anchors       | ✓ per-framework   |
| P10     | ✓ from P6     | ✓                 | ✓                 |
| P11     | ✓ from P1     | ✓                 | ✓                 |
| P12     | ✓ from P2     | ✓                 | ✓                 |
| P13     | ✓ from P5     | ✓                 | ✓                 |
| P14     | ✓ from P3     | ✓                 | ✓                 |
| P15     | ✓ from P4     | ✓                 | ✓                 |
| P16     | ✓ from P2     | ✓                 | ✓                 |

**Next step (paper engineering team)**: insert these 8 hints into the FSM agent prompts, run miner on 4 frameworks, expect 30-60 new rules total. Output to `miner_runs/miner_rules_P{9..16}.json` per brief §2.4.
