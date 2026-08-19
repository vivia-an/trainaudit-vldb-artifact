# Initial Constraint-Template Codebook (Seed stage)

- Stage: seed | n_cases: 128 | date: 2026-07-16
- Induced under the FROZEN merge (§2) / split (§3) / admission (§4) rules.
- 27 templates admitted, 9 singletons, 0 not_codable. Coverage = 119/128 = 0.930.

## Operator normalization (operator_alias_map)

Extractors coined overlapping operator names. Canonicalization (§3.1 keeps relation
semantics distinct):

- `invariance_across_config`, `reference_equivalence`, `selection_correctness` -> **reference_equivalence** (all express "computed output must equal a reference/other computation"). Split further into T03 (reference = same code under another config) vs T04 (reference = external spec/impl), because no single observable signal covers both (§3.4).
- `config_state_consistency` -> **config_effectiveness** ("runtime-effective value must equal user-configured value").
- `proportionality` -> **value_scaling_consistency** (both are "quantity must carry the specified multiplicative factor"; equality-of-scale relation).
- `structural_integrity` was an extractor umbrella; per §3.2 it was resolved case-by-case into finer operators (update_effectiveness, state_preservation, sharding_layout_consistency, value_scaling_consistency) by reading each `relation_statement`. Residual single-incident structural cases remain singletons.
- Left distinct (different relation semantics, §3.1): equality_across_ranks, dtype_preservation, conservation, ordering, restoration_after_reload, boundedness, monotonicity, count_frequency_match, exclusivity, index_consistency, gradient_flow, determinism, state_preservation, sampling_uniformity.

Same-incident duplicates counted once toward support (§4.1): B13/O-002 (residual clobber); D-005/D-023 (CPUAdam subgroup); M-015/M-030 (LR override ignored).

---

## T01 — cross-rank-replica-equality
- **Semantic obligation:** State the recipe declares replicated (params, reduced grads, loss-scale/overflow state, dropout RNG, derived grad-norm scalars) must be checksum-equal across all ranks of its replication group at the same step.
- **Schema predicate:** `FORALL x in replicated_state, FORALL r1,r2 in replica_group(x): checksum(x,r1,step)==checksum(x,r2,step)`
- **Permitted topology guards:** DP>1; EP>1 and DP>1; DP>1 and TP>1
- **Permitted preconditions:** ZeRO-1/2 reduce_scatter; grad accumulation; MoE gate/expert params under ZeRO/dist-opt; fp16 dynamic loss scaler; model-parallel RNG init
- **Positive examples:** D-003, D-011, D-031, D-035, M-018, M-021, D-NEW-61 (support 7)
- **Counterexamples:** TP/EP-sharded tensors compared as replicas; local grads read before the reduction collective.
- **Merge boundary:** Must not absorb T02 (logged metrics are legitimately rank-local until the explicit reduce; merging flags legal runs, §3.5).

## T02 — logged-metric-reduction-consistency
- **Semantic obligation:** A logged metric must be the designated cross-rank reduction of per-rank values, identical on every rank.
- **Schema predicate:** `logged_metric(m,r,step) == designated_reduce({local_metric(m,r',step)}) AND equal across r`
- **Permitted topology guards:** DP>1
- **Permitted preconditions:** distributed metric/optimizer-metric collection; MoE aux/z-loss logging; reduction operands on collective's device
- **Positive examples:** O-014, O-NEW-29, O-NEW-30 (support 3)
- **Counterexamples:** per-rank locals before all-reduce; deliberately rank-local diagnostics.
- **Merge boundary:** Must not absorb T01 (reporting artifact, not training state; different role, §3.2).

## T03 — config-invariant-execution-equivalence
- **Semantic obligation:** Outputs/grads/converged loss must be invariant (within tol) to performance-only config: parallel degree, seq divisibility, compile mode, recompute, FSDP wrapping/init path, bucket size.
- **Schema predicate:** `max_abs_diff(output(cfg,x,w), output(cfg_ref,x,w)) <= tol` where cfg vs cfg_ref differ only in a performance-neutral knob
- **Permitted topology guards:** TP>1; CP>1; SP>1; DP>1; FSDP; any
- **Permitted preconditions:** seq len not divisible by world size; CP/SP attention; torch.compile over collectives/FSDP set_; recompute+FSDP wrapping; param_init_fn+sync_module_states; reduce_bucket_size
- **Positive examples:** D-004, D-039, M-002, M-009, M-029, O-018, O-027, O-035, O-036, O-037 (support 10)
- **Grounding concerns:** O-036 needs a paired reference-init run; root cause not isolated.
- **Counterexamples:** tolerance-level reduction-reorder diffs; loss diffs from knobs that change semantics.
- **Merge boundary:** Must not absorb T04 (external spec/impl oracle vs same-code-other-config; §3.4).

## T04 — spec-reference-conformance
- **Semantic obligation:** A computed quantity must equal the value mandated by its authoritative spec or reference implementation.
- **Schema predicate:** `|computed(q,x) - spec_reference(q,x)| <= tol`
- **Permitted topology guards:** any; TP>1
- **Permitted preconditions:** MLA/yarn scaling; TE fused attn backward on deep models; group-limited top-k; local-vs-TE ViT spec; warmup backend mapping; RMSNorm; HF-parity RoPE dtype
- **Positive examples:** M-022, M-032, M-035, M-NEW-24, M-NEW-40, O-017, O-NEW-1, O-NEW-64, OC-NEW-10 (support 9)
- **Grounding concerns:** M-022 observable only vs external ref, no runtime anomaly; M-032 only "abnormal loss", low confidence.
- **Counterexamples:** intentional documented deviation (configured non-standard scale); fused-vs-unfused tolerance diffs.
- **Merge boundary:** Must not absorb T03 (§3.4); must not absorb T18 (oracle is user config, not algorithm spec; §3.2).

## T05 — designated-dtype-fidelity
- **Semantic obligation:** Every tensor carries its designated dtype at use: grads accumulate/reduce in high precision, buffers/inputs keep declared dtype, recomputed activations match originals.
- **Schema predicate:** `FORALL (t,site): dtype(t,site) == designated_dtype(t, mixed_precision_recipe)`
- **Permitted topology guards:** DP>1; EP>1; FSDP; any
- **Permitted preconditions:** BF16/fp16 with fp32 accumulation; ZeRO-3 all-gather under autocast; grad_accum dtype != model dtype; full-precision eval; recompute; large-world bf16 reduction
- **Positive examples:** D-028, D-036, D-042, D-045, D-NEW-62, M-012, M-024, O-004, O-028, O-031, O-034 (support 11)
- **Counterexamples:** intentionally configured low-precision reduction; recipe-designated low-precision sites.
- **Merge boundary:** Must not absorb T06 (on-disk mismatch invisible to runtime dtype tag; witnessed only by file-size arithmetic, §3.4).

## T06 — storage-dtype-interpretation-match
- **Semantic obligation:** The dtype used to read an on-disk token dataset must equal the write dtype, so token ids are not reinterpreted.
- **Schema predicate:** `itemsize(reader_dtype) == file_size/token_count AND max(token_id) consistent with vocab_size`
- **Permitted topology guards:** any
- **Permitted preconditions:** vocab_size > 65536 with no explicit reader dtype; memmap/packed token files
- **Positive examples:** O-NEW-38, OC-NEW-16 (support 2)
- **Counterexamples:** legit uint16 storage for vocab < 65536; files whose declared dtype matches reader.
- **Merge boundary:** Must not absorb T05 (config, not runtime tensor; never appears in any dtype tag, §3.2/§3.4).

## T07 — gradient-accumulation-conservation
- **Semantic obligation:** Accumulated gradient equals the exact sum of all per-micro-batch/per-rank contributions from a zeroed buffer, each added once, with no foreign bytes.
- **Schema predicate:** `accum_grad(p, N) == SUM_i grad_i(p) with accum_grad(p,0)==0 and padding_region==0`
- **Permitted topology guards:** DP>1; PP>1; any
- **Permitted preconditions:** ZeRO with CPU offload; grad_accum_steps>1; immediate grad hooks on PP non-last stages; reduce-scatter padding; accumulation outside no_sync with offload; first-iter buffer zeroing
- **Positive examples:** D-012, D-017, D-NEW-65, D-NEW-71, M-040, O-029 (support 6)
- **Counterexamples:** documented averaging mode; grads reset at accumulation-boundary steps.
- **Merge boundary:** Must not absorb T08 (wrong multiplicative factor vs missing/extra additive term; different relation, §3.1).

## T08 — scale-factor-consistency
- **Semantic obligation:** Every scaled/normalized quantity in the loss/grad/metric path carries exactly the specified factor (parallel degree, token/sample/micro-batch count, loss scale), applied once.
- **Schema predicate:** `applied_scale(q) == spec_scale(q; dp,cp,sp,num_tokens,num_micro_batches,loss_scale,samples), applied exactly once`
- **Permitted topology guards:** DP>1; EP>1; TP>1; CP>1; any
- **Permitted preconditions:** fp16/bf16 loss scaling; MoE bucket/aux-loss coeffs; SP grad averaging; per-token loss norm; incomplete minibatch at DP boundary; metric accumulation over micro-batches
- **Positive examples:** D-013, D-026, D-034, D-037, D-040, M-003, M-004, M-033, OC-NEW-42 (support 9)
- **Counterexamples:** documented normalization-convention change; dynamic-loss-scale policy behavior.
- **Merge boundary:** Must not absorb T09 (derived global-norm/clip metric, different role, §3.2) or T27 (init statistics, §3.2).

## T09 — gradient-norm-computation-fidelity
- **Semantic obligation:** Reported/applied global grad norm and clip coefficient equal a reference recomputation over actual gradients, correct group divisor, epsilon once globally.
- **Schema predicate:** `|reported_norm - norm(actual_grads)|/norm <= tol AND clip_coef == max_norm/(global_norm+eps), eps once, norm by dp_size`
- **Permitted topology guards:** DP>1; TP>1; FSDP
- **Permitted preconditions:** CPU-offload clipping; precision-aware optimizer grad source; DTensor sharded clipping; TP norm normalization
- **Positive examples:** D-016, M-NEW-67, O-032, O-038 (support 4)
- **Counterexamples:** norm changing because grads genuinely changed; configured non-L2 norm per its own spec.
- **Merge boundary:** Must not absorb T08 (derived monitoring/clipping metric, not a training-path scale, §3.2).

## T10 — producer-consumer-completion-ordering
- **Semantic obligation:** Any consumer of a shared buffer (collective, forward read, deallocation, bucket copy) is ordered after the producer completes across streams/async handles.
- **Schema predicate:** `FORALL (prod,cons,buf) in hazard_pairs: happens_before(complete(prod(buf)), start(cons(buf)))`
- **Permitted topology guards:** DP>1; PP>1; DP>1 and PP>1
- **Permitted preconditions:** non-default backward stream; overlap-grad-reduce/param-gather; interleaved 1F1B async P2P + dealloc; DDP init on separate stream; fp8 param-gather buffer reuse; zero-bubble PP with FSDP manual accumulation
- **Positive examples:** D-021, M-007, M-019, M-026, M-031, M-034, M-NEW-46, O-030 (support 8)
- **Counterexamples:** overlapped comm correctly synchronized before use; dealloc after send-handle wait.
- **Merge boundary:** Must not absorb D-022 sort-key ordering (value-ordering over file names, not happens-before; §3.2/§3.4).

## T11 — checkpoint-restore-state-equality
- **Semantic obligation:** After load, every element of training state equals its save-time value, keyed to the correct parameter (renames preserved, derived caches refreshed).
- **Schema predicate:** `FORALL k: value_after_load(k) == value_at_save(k)`
- **Permitted topology guards:** DP>1; any
- **Permitted preconditions:** ZeRO++ secondary tensors; ZeRO-3 universal ckpt multi-subgroup; load_optimizer_state with changed block_group_size; load_state_dict between fwd/bwd under FSDP FULL_SHARD; initial_lr in restored param_groups
- **Positive examples:** D-024, D-044, M-017, O-016, O-026, O-NEW-33 (support 6)
- **Grounding concerns:** M-017 witnessed by loss deviation vs uninterrupted run; failing sub-state not isolated.
- **Counterexamples:** state intentionally overridden by a user flag (T18); optimizer intentionally re-initialized.
- **Merge boundary:** Must not absorb T13 ("equal-to-saved" vs "reset-to-initial" are unrelated OR branches, §3.3).

## T12 — checkpoint-save-completeness
- **Semantic obligation:** A saved checkpoint contains the complete inventory: every param key, every optimizer key, one shard/file per rank/partition group.
- **Schema predicate:** `keys(saved)==keys(expected) AND count(shard_files)==expected_shard_count(topology) AND all partition_groups saved`
- **Permitted topology guards:** TP>1; DP>1; single-rank
- **Permitted preconditions:** MoE ckpt with TP>1; MiCS partition groups; FSDP use_orig_params+FULL_SHARD inner-module dict; early-iteration save
- **Positive examples:** D-043, D-046, M-042, O-033 (support 4)
- **Counterexamples:** sharded ckpts storing each param on one owner rank per spec; model-only save with optimizer intentionally absent.
- **Merge boundary:** Must not absorb T11 (inventory/count at save vs value equality after reload; unrelated branches, §3.3/§3.4).

## T13 — dataloader-bookkeeping-reset
- **Semantic obligation:** Data-loader bookkeeping is re-initialized to its defined start at run/epoch boundaries so realized order matches the schedule.
- **Schema predicate:** `at boundary b: loader_position(after b) == defined_initial(b)`
- **Permitted topology guards:** any
- **Permitted preconditions:** resume restore_dataloader=False at new epoch; LM evaluator repeated within a session
- **Positive examples:** O-006, O-019 (support 2)
- **Counterexamples:** mid-epoch resume intentionally restoring position; by-design per-epoch shuffle.
- **Merge boundary:** Must not absorb T11 (reset-to-initial vs equal-to-saved; different roles, §3.2/§3.3).

## T14 — scaling-scalar-finiteness
- **Semantic obligation:** Every scalar multiplying/dividing losses/grads (loss scale, clip coef, loss-norm denominator) is finite and its denominator nonzero at every use.
- **Schema predicate:** `FORALL s in scaling_scalars: isfinite(s) AND (denominator IMPLIES s>0)`
- **Permitted topology guards:** any
- **Permitted preconditions:** fp16 loss-scale input validation; NaN/inf clip handling; instance masking with a fully masked batch
- **Positive examples:** D-007, D-038, O-015 (support 3)
- **Counterexamples:** dynamic loss scale halving after genuine overflow; large-but-finite scales in range.
- **Merge boundary:** Must not absorb T08 (bound relation vs equality-to-factor; different operator, §3.1).

## T15 — counter-monotonic-progress
- **Semantic obligation:** Cumulative counters (optimizer step, global FLOPs) advance monotonically by their defined increment; never stall at zero or wrap.
- **Schema predicate:** `counter(t+1) >= counter(t) AND counter(t+1)-counter(t)==defined_increment(t)`
- **Permitted topology guards:** any
- **Permitted preconditions:** SkipStepAdamW-style step maintenance; long runs exceeding integer range
- **Positive examples:** O-010, O-025 (support 2)
- **Counterexamples:** step not advanced on a legitimately skipped overflow step; counters reset at documented boundaries.
- **Merge boundary:** Must not absorb T22 (scalar counter vs parameter tensor; different object/witness, §3.2).

## T16 — positional-encoding-position-fidelity
- **Semantic obligation:** The positional (RoPE) encoding of each token equals the spec encoding of that token's intended global position under the configured layout (per-doc resets, CP shard mapping, interleaving).
- **Schema predicate:** `FORALL tokens t on rank r: applied_rope(t,r) == rope_spec(intended_position(t), layout_cfg)`
- **Permitted topology guards:** CP>1; any
- **Permitted preconditions:** --reset-position-ids packed seqs; CP sequence chunking; rotary_interleaved=True yarn; per-block CP-sharded RoPE buffers
- **Positive examples:** M-011, M-028, M-037, O-011 (support 4)
- **Counterexamples:** contiguous 0..L-1 when no resets/sharding; equivalent buffer layouts with identical angles.
- **Merge boundary:** Must not absorb M-027 sample-index integrity (dataset sample selection role, disjoint observable, §3.2).

## T17 — loss-parameter-gradient-connectivity
- **Semantic obligation:** Every trainable parameter on an active loss path receives nonzero gradient; no loss term detached, no routing construction that zeros the gradient.
- **Schema predicate:** `FORALL p on active loss path: grad_norm(p) > 0 over a window AND graph(loss) reaches p`
- **Permitted topology guards:** any
- **Permitted preconditions:** top-k=1 post-topk softmax; combined losses built from possibly-detached tensors
- **Positive examples:** M-014, OC-NEW-9 (support 2)
- **Counterexamples:** intentionally frozen params; experts with zero grad at a step because no tokens routed.
- **Merge boundary:** Must not absorb T22 (zero grad at loss side vs grad that never becomes a parameter update; different witnesses, §3.2/§3.4).

## T18 — configured-value-effectiveness
- **Semantic obligation:** The runtime-effective value of an accepted user config (LR overrides, softmax scale, optimizer-state layout) equals the configured value.
- **Schema predicate:** `FORALL knob k in accepted_config: effective(k, runtime) == configured(k)`
- **Permitted topology guards:** any; single-rank
- **Permitted preconditions:** resume with --override-opt_param-scheduler; config.softmax_scale set; bf16-without-ZeRO ckpt layout
- **Positive examples:** M-015, M-030, M-NEW-8, D-NEW-21 (support 3; M-015/M-030 same incident, §4.1)
- **Counterexamples:** checkpointed values taking precedence when no override flag; defaults where knob left unset.
- **Merge boundary:** Must not absorb T04 (reference is user config, not algorithm spec, §3.2).

## T19 — packed-sequence-attention-isolation
- **Semantic obligation:** With multiple documents packed in one sequence, cross-document attention mass is exactly zero (block-diagonal mask honored by the backend).
- **Schema predicate:** `FORALL (i,j) with doc(i)!=doc(j): attention(i,j)==0`
- **Permitted topology guards:** any
- **Permitted preconditions:** TE backend with block-diagonal mask; cu_doc_lens with packing
- **Positive examples:** M-036, OC-NEW-47 (support 2)
- **Counterexamples:** causal attention within one document; modes intentionally allowing cross-doc context.
- **Merge boundary:** Must not absorb T20 (attention mass vs loss contributions; different objects/witnesses, §3.2/§3.4).

## T20 — padding-loss-exclusion
- **Semantic obligation:** Padding (ignore-index) positions contribute zero to loss; the loss mask excludes exactly the padding positions.
- **Schema predicate:** `FORALL t with token(t)==pad: loss_mask(t)==0; count(loss_mask==0)==count(padding_tokens)`
- **Permitted topology guards:** any
- **Permitted preconditions:** CE API with ignore_index; padded fixed-seq-length datasets
- **Positive examples:** O-007, O-023 (support 2)
- **Counterexamples:** unpadded sequences (all positions contribute); masking that intentionally includes special tokens.
- **Merge boundary:** Must not absorb T19 (loss/mask tensors vs attention mass; §3.2/§3.4).

## T21 — overflow-skip-step-validity
- **Semantic obligation:** An optimizer step is skipped iff a genuine gradient overflow occurred; skip frequency matches overflow frequency.
- **Schema predicate:** `skipped(t) <=> EXISTS g in grads(t): !isfinite(g); count(skipped)==count(overflow_events)`
- **Permitted topology guards:** DP>1; any
- **Permitted preconditions:** fp16 fused optimizer overflow guard; fp16 loss scaler with overlapped handle init
- **Positive examples:** D-015, M-008 (support 2)
- **Counterexamples:** steps correctly skipped during genuine early overflow bursts; loss-scale warmup skips per policy.
- **Merge boundary:** Must not absorb T22 ("skip iff overflow" vs "non-skipped step must update" are unrelated OR branches, §3.3).

## T22 — parameter-update-effectiveness
- **Semantic obligation:** A non-skipped step with finite nonzero grads actually changes each trainable parameter, and the forward-used tensor is the optimizer-registered object.
- **Schema predicate:** `NOT skipped(t) AND grad_norm(p,t)>0 => checksum(p,t+1)!=checksum(p,t); id(forward_used(p))==id(optimizer_registered(p))`
- **Permitted topology guards:** single-rank; any
- **Permitted preconditions:** tensors >= 2^31 elements (size metadata width); MoE gate rebinding under ZeRO+downcast
- **Positive examples:** D-006, D-NEW-81 (support 2)
- **Counterexamples:** frozen params that must not change (see singleton D-NEW-43); params unchanged on legitimately skipped steps.
- **Merge boundary:** Must not absorb T21 (conditioned on step not being skipped; folding in skip-validity needs an unrelated OR branch, §3.3).

## T23 — unrelated-operation-state-preservation
- **Semantic obligation:** A stateful object (residual-stream input, module id attribute, global RNG) retains its value across operations not designated to write it.
- **Schema predicate:** `FORALL (x,op) with op NOT IN designated_writers(x): state(x,after)==state(x,before)`
- **Permitted topology guards:** single-rank; EP>1; any
- **Permitted preconditions:** pre-LN sequential block (norm must not clobber residual var); ZeRO-3 hook registration over custom-id modules; dataset-config build in a process relying on global RNG
- **Positive examples:** B13, O-002, D-009, O-022 (support 3; B13/O-002 same incident, §4.1)
- **Counterexamples:** in-place updates by the state's designated owner; documented in-place activation ops all consumers expect.
- **Merge boundary:** Must not absorb T11 (clobber by non-designated writer during live exec vs restore-equality across save/load, §3.2).

## T24 — intra-run-recomputation-determinism
- **Semantic obligation:** Re-executions of the same logical computation within one step (recompute passes, per-subgroup metadata) produce values identical to the original.
- **Schema predicate:** `FORALL re-executions e of c within step t: value(c,e)==value(c,original)`
- **Permitted topology guards:** any
- **Permitted preconditions:** activation checkpointing with dropout (and torch.compile); ZeRO-3 CPU-offloaded Adam with multiple subgroups
- **Positive examples:** D-005, D-023, D-008 (support 2; D-005/D-023 same incident, §4.1)
- **Counterexamples:** runs without recompute; documented tolerance-equivalent alternate paths.
- **Merge boundary:** Must not absorb T03 (self-comparison inside one run, no reference run/config; §3.4).

## T25 — sharded-layout-conformance
- **Semantic obligation:** The layout/shape/distribution kind of every tensor an operation touches equals the layout mandated by the active parallel config (correct shard slice, CP-adjusted p2p shapes, local vs distributed tensor kind).
- **Schema predicate:** `FORALL (t,op): layout(t,op) == expected_layout(t, parallel_cfg)`
- **Permitted topology guards:** TP>1; PP>1 and CP>1
- **Permitted preconditions:** PP schedules with CP; MoE gate with kv_head < tp_size; fused-linear loss under TP with FSDP wrappers
- **Positive examples:** M-023, M-039, O-012 (support 3)
- **Counterexamples:** replicated tensors correctly used whole on every rank; shapes that legitimately differ per declared schedule.
- **Merge boundary:** Must not absorb T16 (communicated/sharded tensor layout vs token-position index mapping, §3.2).

## T26 — exactly-once-side-effect
- **Semantic obligation:** A per-boundary side effect (aux-loss tracker increment per step, final-ckpt save per step) occurs exactly once per boundary, including under recompute and restarts.
- **Schema predicate:** `FORALL boundaries b: count(event,b)==1`
- **Permitted topology guards:** any
- **Permitted preconditions:** activation recompute re-running router forward; resume reaching final step again
- **Positive examples:** M-NEW-21, O-039 (support 2)
- **Counterexamples:** documented per-micro-batch logging; periodic checkpoints at every interval.
- **Merge boundary:** Must not absorb T21 (no overflow condition; merging would create unrelated OR branches, §3.3).

## T27 — init-distribution-conformance
- **Semantic obligation:** Parameter statistics immediately after init conform to the configured init spec: correct (depth-scaled) std, never all-zero unless specified, never uninitialized memory.
- **Schema predicate:** `FORALL W: |std(W)-spec_std(W,depth,cfg)|<=tol AND NOT(spec_std>0 AND std(W)==0) AND stats not consistent with uninitialized memory`
- **Permitted topology guards:** any
- **Permitted preconditions:** depth-scaled output-projection init; muP readout zero-init interactions; optional projections materialized from meta device
- **Positive examples:** O-024, O-NEW-21, OC-NEW-66 (support 3)
- **Counterexamples:** architecture-specified zero-init weights; small-tensor fluctuation within tolerance of spec std.
- **Merge boundary:** Must not absorb T08 (init-time weight statistics vs runtime scaled quantity; one-shot statistical test at init, §3.2/§3.4).

---

## Singletons (held, NOT promoted — only 1 independent case each; §4.1)

- **D-022** (ordering) — checkpoint shard files must merge in numeric, not lexicographic, order (sort-key correctness, not a happens-before race; not T10+guards).
- **D-027** (conservation) — MoE softmax coefficients must sum to one over the designated dimension (objects differ from T07 grad accumulation, §3.2).
- **D-041** (structural_integrity) — checkpoint dir step label must equal internal global_step (label/content consistency; not T12 completeness nor T11 restore).
- **D-NEW-43** (exclusivity) — frozen requires_grad=False params must be excluded from trainable partitions/optimizer state (converse of T22 but a distinct exclusion relation).
- **M-025** (dtype_preservation) — master weights created directly in params_dtype so init values equal source elementwise (runtime dtype tag ends up correct; T05's signal cannot witness it, §3.4).
- **M-027** (index_consistency) — stored sample index must equal source doc index without int32 truncation (data-sample selection role differs from T16, §3.2).
- **O-001** (structural_integrity) — attention and FFN must use distinct normalization modules (parameter-aliasing distinctness, not state clobbering T23).
- **O-020** (sampling_uniformity) — downsampling must draw a random subset, not the file prefix (randomness shares no observable with O-021, §3.4).
- **O-021** (count_frequency_match) — realized per-source sample counts must match configured mixture ratios without duplicates (kept separate from O-020, §3.4).

## Not codable
None. Every case yielded a normalizable primary relation.
