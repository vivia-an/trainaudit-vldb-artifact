# Paper §4.1 — Fault injection benchmark

- Total injected faults: **34** (31 bug-class + 3 subthreshold boundary cases)
- **Detection rate: 31/31 = 100.0%** (severe + moderate faults only)
- Boundary cases: **3/3 true negative** (zero FP at sensitivity floor)

## By tier

| tier | faults | detected | det_rate |
|---|---:|---:|---:|
| T0 | 22 | 22 | 100.0% |
| T1 | 9 | 9 | 100.0% |

## By category

| category | faults | detected | det_rate |
|---|---:|---:|---:|
| checkpoint | 1 | 1 | 100.0% |
| data_loading | 2 | 2 | 100.0% |
| dtype | 2 | 2 | 100.0% |
| gradient | 3 | 3 | 100.0% |
| lr_schedule | 3 | 3 | 100.0% |
| moe_router | 3 | 3 | 100.0% |
| normalization | 3 | 3 | 100.0% |
| numerical | 3 | 3 | 100.0% |
| optimizer | 4 | 4 | 100.0% |
| replica | 3 | 3 | 100.0% |
| residual_connection | 1 | 1 | 100.0% |
| structural | 3 | 3 | 100.0% |

## Per-fault verdicts

| fault_id | expected_rule | severity | tier | category | verdict | description |
|---|---|---|---|---|---|---|
| nan_in_fwd_post | `T0-no-nan-inf` | severe | T0 | numerical | **DETECTED** | NaN in module forward output tensor |
| inf_in_optim_pre | `T0-no-nan-inf` | severe | T0 | numerical | **DETECTED** | Inf in optimizer step pre-event param |
| nan_in_comm_post | `T0-no-nan-inf` | severe | T0 | numerical | **DETECTED** | NaN in distributed all-reduce output tensor |
| clip_2x_violation | `T0-clip-grad-bounded` | moderate | T0 | gradient | **DETECTED** | post_norm 2x max_norm — moderate clip bypass |
| clip_10x_violation | `T0-clip-grad-bounded` | severe | T0 | gradient | **DETECTED** | post_norm 10x max_norm — severe clip bypass |
| clip_100x_violation | `T0-clip-grad-bounded` | severe | T0 | gradient | **DETECTED** | post_norm 100x max_norm — extreme clip bypass (B11 fingerpri |
| lr_zero | `T0-optim-lr-positive` | severe | T0 | optimizer | **DETECTED** | AdamW param_group with lr=0 — optimizer disabled |
| lr_negative | `T0-optim-lr-positive` | severe | T0 | optimizer | **DETECTED** | AdamW param_group with lr<0 — sign-flipped updates |
| zero_params_in_build | `T0-build-has-modules` | severe | T0 | structural | **DETECTED** | Build snapshot reports zero parameters |
| zero_modules_in_build | `T0-build-has-modules` | severe | T0 | structural | **DETECTED** | Build snapshot reports zero modules |
| scheduler_resume_no_initial_lr | `T0-initial-lr-present` | severe | T0 | lr_schedule | **DETECTED** | scheduler.init last_epoch=10 + has_initial_lr=False (B12) |
| scheduler_partial_initial_lr | `T0-initial-lr-present` | moderate | T0 | lr_schedule | **DETECTED** | scheduler.init resume + 1 of 2 groups missing initial_lr |
| token_id_overflow | `T0-token-id-in-vocab` | severe | T0 | data_loading | **DETECTED** | DataLoader batch with absurdly large token id (5M, O-NEW-9) |
| token_id_extreme | `T0-token-id-in-vocab` | severe | T0 | data_loading | **DETECTED** | DataLoader batch with int64 max id (2^31) |
| norm_rms_low | `T0-norm-output-unit-rms` | severe | T0 | normalization | **DETECTED** | RMSNorm output rms ≈ 0.33 (O-NEW-1 fingerprint) |
| norm_rms_high | `T0-norm-output-unit-rms` | moderate | T0 | normalization | **DETECTED** | RMSNorm output rms ≈ 3.0 (over-amplified) |
| norm_rms_extreme | `T0-norm-output-unit-rms` | severe | T0 | normalization | **DETECTED** | RMSNorm output rms ≈ 10.0 (numerical instability ahead) |
| router_size1_softmax | `T0-softmax-degenerate` | severe | T0 | moe_router | **DETECTED** | TopKRouter (4,1) softmax — every row is 1.0 (M-014) |
| functional_softmax_one_hot | `T0-softmax-degenerate` | severe | T0 | moe_router | **DETECTED** | F.softmax of post-topk(1) tensor — one-hot signature |
| step_frozen | `T0-optim-step-counter-monotonic` | severe | T0 | optimizer | **DETECTED** | state['step'] frozen at 5 across calls (OC-NEW-2 fingerprint |
| step_regress | `T0-optim-step-counter-monotonic` | severe | T0 | optimizer | **DETECTED** | state['step'] regresses from 10 → 8 (counter rollback) |
| checkpoint_no_rng_with_dropout | `T0-checkpoint-preserve-rng` | severe | T0 | checkpoint | **DETECTED** | checkpoint(preserve_rng_state=False) with Dropout module (O- |
| replica_outlier_rank0 | `T1-replica-cksum-equal` | severe | T1 | replica | **DETECTED** | router.weight diverged at rank 0 (4-rank group) |
| replica_outlier_rank2 | `T1-replica-cksum-equal` | severe | T1 | replica | **DETECTED** | router.weight diverged at rank 2 (4-rank group, M-005) |
| replica_2rank_disagree | `T1-replica-cksum-equal` | moderate | T1 | replica | **DETECTED** | norm.weight diverged in 2-rank group — outlier ambiguous |
| residual_clobbered | `T1-residual-stream-preserved` | severe | T1 | residual_connection | **DETECTED** | OLMo block output closer to normed than original input (B13) |
| router_missing_attr | `T1-router-has-calculate-per-token-loss` | moderate | T1 | moe_router | **DETECTED** | Megatron router missing calculate_per_token_loss when featur |
| expert_bias_bf16 | `T1-expert-bias-fp32` | severe | T1 | dtype | **DETECTED** | TopKRouter.expert_bias silently demoted to bfloat16 (M-012) |
| layer_count_mismatch | `T1-layer-count-strict` | moderate | T1 | structural | **DETECTED** | num_layers=24 % pp_size=5 != 0 (M-020) |
| jitter_promotes_dtype | `T1-jitter-preserves-dtype` | severe | T1 | dtype | **DETECTED** | Megatron apply_input_jitter promotes bf16 → fp32 (M-024) |
| sqrt_decay_inverted | `T1-sqrt-decay-front-loaded` | severe | T1 | lr_schedule | **DETECTED** | OLMo-core _sqrt_decay slope inverted (slow-then-fast, OC-NEW |
| bnd_clip_within_tolerance | `` | subthreshold | T0 | gradient | **TRUE_NEGATIVE** | post_norm = max_norm * 1.005 — within 1% rel tol |
| bnd_norm_rms_at_boundary | `` | subthreshold | T0 | normalization | **TRUE_NEGATIVE** | RMSNorm rms = 0.55 — just inside [0.5, 2.0] band |
| bnd_token_id_just_under | `` | subthreshold | T0 | data_loading | **TRUE_NEGATIVE** | max_id = 1048576 (= 2^20 ceiling, edge inclusive) |
