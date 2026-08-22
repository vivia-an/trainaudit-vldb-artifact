# Rule expressibility mapping (A0 + A3)

Classification of every rule in `trainaudit/trainaudit/rules/` against the
four internal predicate shapes (`TENSOR_STAT_BOUND`, `PAYLOAD_FIELD_COMPARE`,
`CONDITIONAL_CHECK`, `STRUCTURAL_PRESENCE`).

Three categories:
- **dsl_native** — rule body fits the DSL with the built-in predicate-shape operators.
- **probe_derived** — rule body is trivial *given* an active probe that pre-computes
  the analysis (a framework adapter wraps a function and emits structured events;
  the rule then reads one field on the probe event). The probe is bespoke per
  bug-family but the rule body itself is dsl_native-shaped over the probe events.
- **python_fallback** — needs cross-event joins or shape-aware logic that the
  four predicate shapes cannot naturally express. Kept as Python; documented
  exceptions.

A3 hard constraint: at most **two** new predicate-shape fields.

## Summary

| Category | Count | Notes |
|---|---:|---|
| dsl_native (current shapes) | 10 | no shape extensions needed |
| dsl_native (after `tensor_signature` + `monotonic` extensions) | +3 → 13 | 2 small operator additions |
| probe_derived | 3 | rule body is dsl_native over probe-emitted events |
| python_fallback | 2 | genuinely cross-event / cross-step joins |
| **Total** | **18** | |

A1 PoC scope: convert the 10 unconditionally-dsl_native rules first, ship in `registry/T0/*.yaml` and `registry/T1/*.yaml`. A3 then ships `tensor_signature` and `monotonic` extensions to bring 3 more in.

## Per-rule classification

| ID | File | Predicate shape | Class | Reason |
|---|---|---|---|---|
| T0-no-nan-inf | T0_no_nan_inf.py | TENSOR_STAT_BOUND | dsl_native | scan tensor summaries for has_nan / has_inf flags; trivial OR over nested dict |
| T0-clip-grad-bounded | T0_clip_grad_bounded.py | PAYLOAD_FIELD_COMPARE | **dsl_native** | `post_norm > max_norm * 1.01` per row |
| T0-softmax-degenerate | T0_softmax_degenerate.py | TENSOR_STAT_BOUND + shape | **dsl_native after `tensor_signature` ext** | `abs_max ≈ 1.0 AND l2² ≈ n_rows` — needs shape-aware n_rows derivation |
| T0-token-id-in-vocab | T0_token_id_range.py | TENSOR_STAT_BOUND | dsl_native | `min_id < 0 OR max_id > 2^20` |
| T0-initial-lr-present | T0_initial_lr_present.py | CONDITIONAL_CHECK | **dsl_native** | `last_epoch != -1 → param_groups.has_initial_lr == True` |
| T0-norm-output-unit-rms | T0_norm_output_rms.py | TENSOR_STAT_BOUND + shape | **dsl_native after `tensor_signature` ext** | `rms = l2 / sqrt(numel) ∈ [0.5, 2.0]` — shape-aware divisor |
| T0-dtype-propagation | T0_dtype_propagation.py | — | python_fallback | needs JOIN on (module_class, step) between fwd.pre and fwd.post — not in the current predicate shapes |
| T0-optim-lr-positive | T0_optim_lr_positive.py | PAYLOAD_FIELD_COMPARE | **dsl_native** | `lr > 0` over param_groups |
| T0-optim-step-counter-monotonic | T0_optim_step_counter.py | PAYLOAD_FIELD_COMPARE + monotonic | **dsl_native after `monotonic` ext** | window-pair check across optim.step.post events; one new bound kind |
| T0-build-has-modules | T0_layer_count.py | STRUCTURAL_PRESENCE | **dsl_native** | `model.n_modules > 0 AND model.n_parameters > 0` |
| T0-checkpoint-preserve-rng | T0_checkpoint_preserve_rng.py | — | python_fallback | needs precondition computed by walking build.snapshot for any "Dropout" class; cross-event state machine |
| T1-replica-cksum-equal | T1_replica_cksum_equal.py | STRUCTURAL_PRESENCE | **dsl_native** | iterate `build.snapshot.cross_rank_cksums`, fail if `all_equal == False AND group_size > 1` |
| T1-residual-stream-preserved | T1_residual_stream.py | PAYLOAD_FIELD_COMPARE on probe | **probe_derived** | OLMo adapter emits residual.probe with d_to_input/d_to_normed; rule body is `d_normed < d_input` |
| T1-router-has-calculate-per-token-loss | T1_router_attribute.py | CONDITIONAL_CHECK | **dsl_native** | precondition: `framework_invariants.megatron.calculate_per_token_loss == True`; check: `semantic.has_calculate_per_token_loss != False` |
| T1-expert-bias-fp32 | T1_expert_bias_dtype.py | PAYLOAD_FIELD_COMPARE | **dsl_native** | `semantic.expert_bias_dtype == 'float32'` for is_router events |
| T1-layer-count-strict | T1_layer_count_strict.py | STRUCTURAL_PRESENCE-like | **dsl_native** | `actual_per_rank * pp_size == declared_num_layers` from build.snapshot.framework_invariants |
| T1-jitter-preserves-dtype | T1_jitter_dtype.py | PAYLOAD_FIELD_COMPARE on probe | **probe_derived** | `jitter.probe.dtypes_match == True` (probe pre-computes match) |
| T1-sqrt-decay-front-loaded | T1_sqrt_decay_direction.py | curve-fit on probe series | **probe_derived** | OLMo-core adapter wraps _sqrt_decay → decay.probe events; rule fits slope at 2 quartiles. Probe pre-collected; rule body is window aggregate. |

## Predicate-shape extensions (A3, ≤2)

1. **`tensor_signature`** — derived field for tensor-stat predicates that
   need shape-aware divisors:
   - `n_rows = product(shape[:-1])`
   - `numel  = product(shape)`
   - `rms   = l2_norm / sqrt(numel)`
   - `is_one_hot = abs_max ≈ 1.0 AND l2² / n_rows ∈ tolerance`

2. **`monotonic`** — bound kind for cross-step comparisons:
   - `monotonic.strictly_increasing(field)` over events sorted by event_id

These two together convert 3 more rules to dsl_native (softmax_degenerate, norm_output_rms, optim_step_counter), reaching 13/18 in DSL.

## Probe-derived contract (B1/B2 mining language)

For any active probe to be "DSL-compatible" it must:
- Emit events with a single fixed `hookpoint` (e.g. `decay.probe`, `residual.probe`, `jitter.probe`)
- Pre-compute the comparison fields the rule will read (`d_to_input`, `dtypes_match`, etc.)
- Carry sufficient context fields for diagnosis (block_class, module_class)

This keeps the rule itself dsl_native — adapter authors take the heavy lifting,
DSL stays simple.

## M1 gate closeout (synthetic equivalence)

A2 compiler + the 7 dsl_native YAMLs were validated end-to-end:

- **`tests/dsl/test_compiler_equivalence.py`** — 14 tests, one buggy + one clean
  per rule. For each rule the DSL-compiled SQL and the registered Python rule
  produce **identical** violation `event_id` sets on the same synthetic trace.
- **Verifier wiring** — `trainaudit.run_rules(use_dsl=True)` reads
  `dsl/registry/` and runs the DSL path; `use_dsl=False` (default) keeps the
  Python registry. Tier filter applies to both.
- **Coverage** — At T0_PYTORCH tier, 6 rules overlap (DSL + Python),
  zero mismatches. The 5 Python-only rules are exactly the MAPPING-classified
  `python_fallback` (2) and `needs-extension` (3 — A3 will close those with
  `tensor_signature` + `monotonic` predicate-shape extensions).

**Live-trace M1 (follow-up)** — Re-run the same equivalence on captured
`trainaudit_run.sh` traces from B11 / M-014 / O-NEW-1 / O-005 / B12. This is a
GPU-side verification and is naturally absorbed by D1 (the 32/48 bug eval
harness re-runs every driver and writes `.duckdb`). Until then,
`tests/dsl/test_compiler_equivalence.py` is the load-bearing M1 evidence.

## Why the 2 python_fallback rules are kept

- **`T0-dtype-propagation`**: passive at T0 by design (doc 22 says the
  full version belongs to T1). Promote to dsl_native when adapter labels
  `input_dtype` on every fwd.post event.
- **`T0-checkpoint-preserve-rng`**: precondition needs a *global model
  property* ("does the model have a Dropout layer anywhere?"). DSL does
  not have a way to express set-existential queries over the full build
  snapshot without a bespoke operator. Keep as Python; document the
  bound clearly.

These two are intentional fallbacks in SQL compilation coverage; they do not
define or modify the semantic Pattern Catalog.
