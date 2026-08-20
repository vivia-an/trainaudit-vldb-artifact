# Trace data caveats

Found by executing the recovered SQL against the published traces, not by reading them.
These matter for anyone re-deriving a clean-trace false-positive rate.

## The DP rank is recorded as 0 on both ranks in four clean runs

For four of the 43 published traces, both per-rank collector files carry `"dp": 0` in
their JSON payload, even though the filenames distinguish the ranks:

```
normal_db/dp_normal/Collector/coredump_dp0_tp0_pp0_cp0.db   185,135 rows   dp values ['0']
normal_db/dp_normal/Collector/coredump_dp1_tp0_pp0_cp0.db   185,135 rows   dp values ['0']
normal_db/dp_normal/Collector/merged_coredump.db            370,270 rows   dp values ['0']
```

The merge is correct — 185,135 + 185,135 = 370,270 — so this is a **collector labelling
issue on the second DP rank**, not a merge that dropped data. Affected runs, all of them
clean baselines:

| Run | rank combinations in the merged DB | per-rank files |
|---|---:|---:|
| `dp_normal_db` | 1 | 2 |
| `normal_db/dp_normal` | 1 | 2 |
| `normal_db/dist_optimizer_normal` | 1 | 2 |
| `normal_db/mixed_precision_normal` | 1 | 2 |

The other 39 traces are consistent: `tp_router_test_db` reports TP ranks 0 and 1,
`cksum_before_backward_bitwise_level_test_db` two rank combinations, and so on.

## What it does and does not affect

Of the 228 recovered queries:

| | count |
|---|---:|
| reference the `dp` field at all | 182 |
| **count or group by the dp rank** — see a single rank on these four traces | **31** |
| reference the `tp` field | 100 |

- **Unaffected**: rules that group by parameter and step and test
  `HAVING COUNT(DISTINCT cksum) > 1`. Both ranks' rows are present in the merged database,
  so a genuine cross-rank divergence still shows up as two distinct checksums. This is the
  common shape, including the cross-rank replication rule of §4.1.
- **Affected**: the 31 queries that compute `COUNT(DISTINCT dp_rank)` or group by the rank
  to decide whether a parameter is replicated. On these four traces they see one rank and
  cannot fire.
- **Guarding is unaffected.** The harness takes the topology from the `--dp/--tp/--pp`
  arguments, not from the trace, so `applicable_conditions` such as `{"dp": "> 1"}` still
  gate correctly.

## Why it matters for the paper

Clean traces are exactly where false positives are counted. Four of the five clean
databases in the leave-one-out ablation
(`experiments/guard_ablation/d1_results.csv`: `dp_normal`, `dp_normal_db`,
`dist_optimizer_normal`, `mixed_precision_normal` — only `tp_normal` is not affected) are
among the four above.

For the 31 rank-counting rules, a clean-trace false-positive rate measured on those
databases is therefore a **lower bound**: those rules cannot fire there. The direction is
against the paper — a corrected trace would show the same or more false positives, not
fewer. It does not affect detection results, which are measured on the fault-injected and
Real-SE replays.

Reproduce this audit:

```bash
bash scripts/fetch_trace_dbs.sh --dest /path/for/traces
python3 core/validate_generated_sql.py --db /path/for/traces/normal_db/dp_normal/Collector/merged_coredump.db
```
