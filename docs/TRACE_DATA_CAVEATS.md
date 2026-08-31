# Trace data caveats

Found by executing the recovered SQL against the published traces, then comparing the
per-rank captures at content level. Relevant to anyone re-deriving a clean-trace
false-positive rate.

> **Three corrections to earlier versions of this file.** (1) I first described this as the
> collector mislabelling the second DP rank; it is not — the captures are identical in
> content. (2) I then listed seven affected runs, three of them fault-injected. That was an
> artifact of hashing only the `.db` file: DuckDB keeps uncommitted data in a `.wal`
> sidecar, so every un-checkpointed database shares an identical 12 KB header and appears
> to be a duplicate. Content-level comparison clears those three. **Four runs are affected,
> all clean baselines.**

## Method

Hashing `coredump_*.db` is not sound here — 7 of the 43 runs keep their per-rank data in
`.db.wal` sidecars, leaving a 12 KB header that collides across runs. The check below is
order-independent and reads every row:

```sql
SELECT md5(string_agg(s,'' ORDER BY s)) FROM (
  SELECT step||'|'||stage||'|'||CAST(data AS VARCHAR) s FROM coredump)
```

It must cover the **whole** `data` payload. Projecting onto
`(step, stage, name, cksum)` produced false matches — seven TP injection runs collided on
it while differing in `grad_cksum`, `requires_grad` and `shape`, because an injected fault
need not change a parameter checksum.

## Finding 1 — four clean baselines have rank 1 identical to rank 0

| Run | rank 0 | rank 1 | |
|---|---|---|---|
| `normal_db/dp_normal` | `dc64169f95d5` | `dc64169f95d5` | identical |
| `dp_normal_db` | `18229edd5147` | `18229edd5147` | identical |
| `normal_db/dist_optimizer_normal` | `18229edd5147` | `18229edd5147` | identical |
| `normal_db/mixed_precision_normal` | `6d0f43bc17dc` | `6d0f43bc17dc` | identical |

185,135 rows each, and the merge is arithmetically right (185,135 × 2 = 370,270) — it
merges rank 0 with a copy of itself. Every cross-rank comparison on these traces therefore
compares rank 0 with itself: a rule of the form
`GROUP BY param_name, step HAVING COUNT(DISTINCT cksum) > 1` sees one distinct checksum per
group by construction and **passes vacuously**. That is the dominant rule shape, §4.1's
cross-rank replication rule included.

Guarding still behaves normally — the harness takes topology from its `--dp/--tp/--pp`
arguments, not from the trace — so the rules are enabled and simply cannot find anything.

**How much of the ablation rests on them** (`experiments/guard_ablation/d1_results.csv`):

| arm | clean-arm FP | of which from these four | share |
|---|---:|---:|---:|
| `lib_full` | 9 | 6 | 67% |
| `lib_no_precond` | 13 | 9 | 69% |
| `lib_no_topo` | 35 | 28 | 80% |

Across all 126 cells the four contribute 43 of 1,369 false positives (3%) — the aggregate
is dominated by the fault-injected databases. But the clean arm is where a false positive
is unambiguous, and there two thirds to four fifths of it comes from traces on which
cross-rank rules cannot fire. Only `tp_normal` among the clean databases has two genuine
ranks. The direction is against the paper: real rank-1 captures could only add fires.

Detection results are unaffected — those come from the Real-SE method-level replays in
`benchmark/eval/real_sdc/`, not these databases.

## Finding 2 — a stray directory duplicates an injected run

`tp_normal_db` and `tp_router_test_db` are the same trace at full payload — identical
per-rank files (`378708b49c5d`, `2e79b9d18eca`) and identical merged database
(`f44095a479327e17`), 50,330 rows each.

This does **not** touch the ablation. The clean TP database it uses is
`normal_db/tp_normal` (370,270 rows, hash `9e88f53b228d`), a different directory that is
genuinely distinct from `tp_router_test_db`. `tp_normal_db` does not appear in
`experiments/guard_ablation/d1_results.csv` at all. It is a stray copy — worth deleting so
nobody mistakes it for a clean TP baseline, but not a validity problem.

### Withdrawn: "two of the 42 databases are the same data"

An earlier version of this file claimed `dp_normal_db` and `normal_db/dist_optimizer_normal`
were the same trace, and offered their differing recorded results as evidence of run-to-run
nondeterminism. **Both claims were wrong** — an artifact of the projected hash. At full
payload they differ (`ae8bd95e5c91e56e` vs `ad03ed650a166eb5`; per-rank `5367d4ae` vs
`c7d17763`), so the ablation really does run over 42 distinct databases.

The nondeterminism implied by O23 is real, but the evidence for it is elsewhere and
cleaner: `core/config/generated_sql.json` records **193 of 228 constraints with more than
one distinct SQL variant** across the logged runs, because the SQL is generated per run.

## Finding 3 — the published bundle omits the WAL sidecars

`scripts/fetch_trace_dbs.sh` and release `trace-dbs-v1` were built from `Collector/*.db`
only. For the 7 runs whose per-rank data lives in `.db.wal` (~11.6 MiB across
`dtype_test_db`, `grad_existence_test_db`, `requires_grad_test_db`,
`requires_grad_after_backward_test_db`, `requires_grad_before_backward_test_db`,
`shape_test_db`, `tp_requires_grad_test_db`) the per-rank files shipped as empty 12 KB
headers.

**The ablation path is unaffected**: `run_d1_phase3.sh` reads `merged_coredump.db`, and
every merged database is checkpointed and self-contained (no `merged_coredump.db.wal`
exists anywhere). Per-rank files are used only for topology inference, from their
filenames. Anyone wanting per-rank data for those 7 runs needs the sidecars, which are
included from `trace-dbs-v2` onward.

## Reproduce

```bash
bash scripts/fetch_trace_dbs.sh --dest /path/for/traces
python3 benchmark/injection/audit_rank_captures.py --root /path/for/traces
```
