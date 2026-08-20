# Trace data caveats

Found by executing the recovered SQL against the published traces, then comparing the
per-rank captures at content level. Relevant to anyone re-deriving a clean-trace
false-positive rate.

> **Two corrections to earlier versions of this file.** (1) I first described this as the
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
  SELECT step||'|'||stage||'|'||coalesce(json_extract_string(data,'$.name'),'')
         ||'|'||coalesce(json_extract_string(data,'$.cksum'),'') s FROM coredump)
```

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

## Finding 2 — two of the "42 databases" are the same data

`dp_normal_db` and `normal_db/dist_optimizer_normal` have the **same content hash**
(`18229edd5147`). The leave-one-database-out ablation treats them as two of its 42
databases.

They also recorded **different results from identical input**:

| arm | `dist_optimizer_normal` | `dp_normal_db` |
|---|---|---|
| `lib_full` | pass 87, FP 2, err 4 | pass 85, FP 1, err 7 |
| `lib_no_precond` | pass 84, FP 2, err 7 | pass 83, FP 2, err 8 |
| `lib_no_topo` | pass 74, FP **9**, err 10 | pass 81, FP **4**, err 8 |

Same trace, same library, and the `no_topo` arm reports 9 false positives against 4 — a
factor of two. This is the clearest evidence in the artifact of the nondeterminism implied
by O23: the SQL is generated per run by an LLM, so two runs of the same cell are not the
same experiment. It also means the 42 databases are 41 distinct traces.

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
