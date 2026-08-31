# Real Bug Detection Runbook

This note describes how to turn the existing experiments under
`/volume/qscai/lsk/Megatron-LM` into auditable TrainAudit detections for real
training bugs.

## Existing Assets

- Collector: `/volume/qscai/lsk/Megatron-LM/new_megatron_collector.py`
- Direct checker: `/volume/qscai/lsk/Megatron-LM/batch_constraint_check.py`
- Existing traces: `/volume/qscai/lsk/Megatron-LM/*_test_db/Collector/*.db`
- Current trace schema: `coredump(step INTEGER, stage TEXT, data JSON)`

The current setup already supports model/gradient/optimizer/scheduler/RNG
snapshots. The missing piece for production bugs is event-level signals: router
auxiliary-loss coefficient events, TE wgrad-write events, and eval RNG
boundary events.

## Audit Protocol

Each real bug should be evaluated through the same protocol:

1. Define the invariant before replay.
2. Add only the minimum collector signal required by that invariant.
3. Run a buggy replay and a fixed replay with the same script, seed, and config.
4. Count the bug as detected only if the buggy run triggers the expected
   invariant and the fixed run has no violation.
5. Store the trace, checker output, commit hash, config, and environment
   fingerprint under a stable `replays/<bug_id>/` directory.

Private production bugs can be reported as deployment evidence, but should not
be included in the public primary metric unless a replayable artifact is
available.

## Pilot 1: MoE Aux Loss Coeff

Bug:

- Commits: `02c138202` and `498749b15`
- Failure: `TopKRouter.get_aux_loss_coeff()` returns `None` or divides by
  `self.layer_number` instead of `self.config.num_layers`.

Invariant:

```text
For each MoE router layer:
actual_aux_loss_coeff is not NULL
actual_aux_loss_coeff > 0 when global_moe_aux_loss_coeff > 0
actual_aux_loss_coeff equals the config-derived expected coefficient
```

Collector event:

```json
{
  "type": "moe_aux_coeff",
  "module_name": "...",
  "layer_number": 12,
  "num_layers": 48,
  "moe_aux_loss_coeff": 0.001,
  "moe_aux_loss_divide_by_num_layers": true,
  "actual_aux_loss_coeff": 0.0000208333,
  "expected_aux_loss_coeff": 0.0000208333
}
```

Suggested hook point:

- `megatron/core/transformer/moe/router.py`
- Inside `TopKRouter.apply_load_balancing_loss`, immediately after the actual
  `moe_aux_loss_coeff` is computed and before `aux_loss` is calculated.

Checker predicate:

```sql
SELECT step, data
FROM coredump
WHERE json_extract_string(data, '$.type') = 'moe_aux_coeff'
  AND (
    json_extract(data, '$.actual_aux_loss_coeff') IS NULL
    OR CAST(json_extract(data, '$.actual_aux_loss_coeff') AS DOUBLE) = 0
    OR ABS(
      CAST(json_extract(data, '$.actual_aux_loss_coeff') AS DOUBLE)
      - CAST(json_extract(data, '$.expected_aux_loss_coeff') AS DOUBLE)
    ) > 1e-12
  );
```

Pass criterion:

- Buggy replay: at least one violation.
- Fixed replay: zero violations.

## Pilot 2: PLT Shared Weight With Delayed Wgrad

Bug:

- Failure: `loop_0` and `loop_1` share the same TE Linear weight. With
  delayed wgrad computation, the first forward captures
  `is_first_microbatch=True`; reverse-order `backward_dw` lets a later
  overwrite erase a previous loop's gradient contribution.

Invariant:

```text
For the same weight storage within one optimizer step:
there must be at most one overwrite-mode wgrad write.
If a shared weight receives multiple wgrad writes, all later writes must
accumulate.
```

Collector event:

```json
{
  "type": "wgrad_write",
  "global_step": 7,
  "microbatch_id": 1,
  "module_name": "...",
  "weight_storage_id": "0x...",
  "logical_loop_id": "loop_0",
  "is_first_microbatch": true,
  "delay_wgrad_compute": true,
  "accumulate": false,
  "main_grad_cksum_before": "...",
  "main_grad_cksum_after": "..."
}
```

Required detail:

- `module_name` is not enough because two logical loops may share the same
  physical weight.
- The collector must record a stable physical identity, such as
  `weight_storage_id = weight.untyped_storage().data_ptr()`.

Checker predicate:

```sql
SELECT
  json_extract_string(data, '$.global_step') AS global_step,
  json_extract_string(data, '$.weight_storage_id') AS weight_storage_id,
  COUNT(*) AS overwrite_count
FROM coredump
WHERE json_extract_string(data, '$.type') = 'wgrad_write'
  AND json_extract_string(data, '$.delay_wgrad_compute') = 'true'
  AND json_extract_string(data, '$.accumulate') = 'false'
GROUP BY global_step, weight_storage_id
HAVING COUNT(*) > 1;
```

Pass criterion:

- Buggy replay: at least one `(global_step, weight_storage_id)` with multiple
  overwrite writes.
- Fixed replay: zero such groups.

## Pilot 3: Eval RNG Pollution

Bug:

- Failure: an eval path modifies Python, NumPy, Torch CPU, or CUDA RNG state
  and returns to training without restoring it.

Invariant:

```text
When eval returns to training, each RNG state checksum must match the snapshot
taken immediately before eval.
```

Collector event:

```json
{
  "type": "rng_boundary",
  "event": "eval_start",
  "global_step": 1000,
  "python_rng_cksum": "...",
  "numpy_rng_cksum": "...",
  "torch_cpu_rng_cksum": "...",
  "torch_cuda_rng_cksum": "..."
}
```

Suggested hook points:

- `megatron/training/training.py`
- At the start of `evaluate`: emit `event = eval_start`
- Immediately after switching the model back to training mode: emit
  `event = train_resume`

Checker predicate:

```sql
WITH start_state AS (
  SELECT step, data
  FROM coredump
  WHERE json_extract_string(data, '$.type') = 'rng_boundary'
    AND json_extract_string(data, '$.event') = 'eval_start'
),
resume_state AS (
  SELECT step, data
  FROM coredump
  WHERE json_extract_string(data, '$.type') = 'rng_boundary'
    AND json_extract_string(data, '$.event') = 'train_resume'
)
SELECT s.step, s.data AS before_rng, r.data AS after_rng
FROM start_state s
JOIN resume_state r ON s.step = r.step
WHERE json_extract_string(s.data, '$.python_rng_cksum')
      != json_extract_string(r.data, '$.python_rng_cksum')
   OR json_extract_string(s.data, '$.numpy_rng_cksum')
      != json_extract_string(r.data, '$.numpy_rng_cksum')
   OR json_extract_string(s.data, '$.torch_cpu_rng_cksum')
      != json_extract_string(r.data, '$.torch_cpu_rng_cksum')
   OR json_extract_string(s.data, '$.torch_cuda_rng_cksum')
      != json_extract_string(r.data, '$.torch_cuda_rng_cksum');
```

Pass criterion:

- Buggy replay: RNG mismatch after eval.
- Fixed replay: zero RNG mismatches.

## Minimal Implementation Order

1. Add generic `dump_event(stage_name, event)` to the collector.
2. Implement the MoE aux coeff event and checker first.
3. Add eval RNG boundary events.
4. Add TE wgrad-write events after the first two are stable.
5. Add `replays/<bug_id>/bug.yaml` for each real bug and store every run output
   with a stable `run_id`.

## Paper Reporting Rule

Use this wording for private bugs:

```text
Private production incidents are evaluated with the same fixed audit protocol
as public replays, but proprietary code and raw traces are not released. We
report their invariant, collected signals, violation predicate, and fixed-run
negative control. Unless a replayable artifact can be released, these incidents
are used as deployment evidence and are not included in the primary public
benchmark metric.
```
