# The two guards, measured on merged multi-rank clean traces

`fig:predicate-ablation` reports clean-trace false positives of **25.8 FP/1M** guarded,
rising to **83.3** without π_topo. The data behind those figures is missing (GAP_AUDIT
O21, and see [`O21_S2_ABLATION_RECOVERY.md`](O21_S2_ABLATION_RECOVERY.md)). What *is*
available is the code for all three arms — `rules`, `rules_no_topo`, `rules_no_precond`
— and the multi-rank clean runs from the §6.2 audit. This runs the arms over those runs.

```
PYTHONPATH=core/trainaudit_pkg python3 benchmark/eval/guard_arms_multirank.py \
    --runs .../E_clean_run_fp_audit/_runs
python3 benchmark/eval/guard_arms_multirank.py --check   # against the recorded results
```

The traces are written **one DuckDB file per rank**, and the replica-checksum rules the
π_topo guard protects compare ranks inside a replica group. Opened one file at a time
those rules see a single rank and can never fire, so any π_topo ablation comes back empty
for a reason that has nothing to do with the guard. This script merges a run's ranks into
one in-memory store first, offsetting `event_id` per rank. An earlier version of
`guard_invariant_singlerank.py` got this wrong and read the forced-empty result as a
measurement; that script now says so and defers here.

## What the guards are

Diffing the shipped rule sets, π_topo is exactly **four** sites: `if group_size <= 1:
continue` in `T1-replica-cksum-equal`, `T1-grad-replica-cksum-equal` and
`T1-buffer-replica-cksum-equal`, plus dropping `and int(declared) > 1` in
`T1-process-group-size-correct`. That matches the recovered bytecode's "/3 replica-cksum
rules" and "/4 RECON §D rules".

## Result 1 — π_precond is strongly load-bearing

| run | ranks | events | full | Δ no_topo | Δ no_precond |
|---|---|---|---|---|---|
| `dp2_fsdp_long_r1` | 2 | 221,754 | 5 | **0** | 4 |
| `dp8_ddp_r1` | 8 | 1,227,896 | 5 | **0** | 5 |
| `dp8_fsdp_r1` | 8 | 1,311,528 | 4 | **0** | 5 |
| `smoke_r0` | 2 | 6,628 | 5 | **0** | 4 |

Dropping π_precond does not nudge the false-positive count, it detonates it. On
`dp8_ddp_r1`: `T1-expert-bias-fp32` goes from silent to **393,960** routers,
`T1-residual-stream-preserved` to **234,768** blocks, `T0-norm-output-unit-rms` from
silent to **227,698**. On `dp2_fsdp_long_r1` that last rule goes 41 → **32,078**. The
paper's qualitative claim for the precondition predicate is well supported — on this
evidence, understated.

## Result 2 — π_topo is inert on every clean trace available

Zero delta on all four runs, including a merged 8-rank DP=8 trace where the replica
groups are real and `group_size=8` is recorded in the payloads. The guard is not
protecting anything here, and that is expected: under DDP every rank *is* a true replica,
so the checksums agree and the guarded rules stay silent either way. The guard bites only
where entries carry `group_size <= 1` **and** comparing them anyway would manufacture a
violation — Megatron TP=2, where tensor-parallel ranks are not replicas and their
checksums legitimately differ.

There is no such trace to test on. The only TP=2 run in the workspace,
`hunt_log/novel_hunt/olmo_core_tp2`, holds **0 events** (12 KB of DuckDB header and a
311-byte WAL), one of five empty shells the authors' own `e_status.md` already flags as
"fires=0 是因为 events=0 … 不是真 clean". So the 83.3 figure cannot be reproduced *or*
contradicted from what is here; it rests entirely on the absent Megatron TP=2 corpus.

Worth noting the recovered bytecode annotates one of its own π_topo rows
**"cosmetic, unreachable"** — the authors had already found at least one guard site that
does nothing.

## Result 3 — a rank-blind rule, firing once per rank boundary

`T0-optim-step-counter-monotonic` fires on every merged run. Its query is

```sql
SELECT event_id, payload FROM events
 WHERE hookpoint = 'optim.step.post' ORDER BY event_id
```

with **no rank partition**, and it then demands a strictly increasing series. Feed it more
than one rank and the step counter resets at each rank boundary, so it reports a
violation there. The prediction is exact — *n_ranks − 1* — and it holds on all four runs:

| run | ranks | bad transitions | n_ranks − 1 |
|---|---|---|---|
| `dp2_fsdp_long_r1` | 2 | 1 / 399 | 1 |
| `dp8_ddp_r1` | 8 | 7 / 1599 | 7 |
| `dp8_fsdp_r1` | 8 | 7 | 7 |
| `smoke_r0` | 2 | 1 | 1 |

The authors' own `rule_results.json` for these runs does **not** list this rule, which
fits: their harness evaluates per-rank files, where the rule is structurally safe. It is
a latent false positive that appears the moment a trace holds more than one rank — which
is how the §4.4 events schema, with its `rank` column, is designed to store them. The fix
is a `PARTITION BY rank`, and it is a rule-level bug rather than anything the paper
claims.

## Caveat

These runs are clean, so the arms measure false positives only; nothing here speaks to
detection rate. Rule counts, not violation counts, are the unit — `n_violations` in the
recorded JSONs saturates at 50 (see [`CLEAN_RUN_FP_MULTIRANK.md`](CLEAN_RUN_FP_MULTIRANK.md)),
though the messages this script reads carry the true counts. The traces are 2.6 GB and are
not shipped; `benchmark/eval/clean_run_fp/guard_arms_results.json` records the outcome and
`--check` guards it.
