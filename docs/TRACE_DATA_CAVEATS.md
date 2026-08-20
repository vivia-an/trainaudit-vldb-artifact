# Trace data caveats

Found by executing the recovered SQL against the published traces and then comparing the
per-rank files, not by reading them. These matter for anyone re-deriving a clean-trace
false-positive rate.

> **Correction.** An earlier version of this file described the problem as the collector
> mislabelling the second DP rank. That was wrong. The per-rank files are **byte-identical
> duplicates**, so the second rank's capture is not mislabelled — it is missing, with a copy
> of rank 0 in its place. The scope is also wider than first stated: 7 runs, not 4.

## Seven runs have duplicate per-rank captures

```
$ md5sum normal_db/dp_normal/Collector/coredump_dp*.db
c5a616609918000fe0f30ba234192ad8  coredump_dp0_tp0_pp0_cp0.db
c5a616609918000fe0f30ba234192ad8  coredump_dp1_tp0_pp0_cp0.db
```

Same digest, same 185,135 rows, same 3,941 distinct checksums, identical row-by-row. The
merge is arithmetically correct — 185,135 + 185,135 = 370,270 — but it merges rank 0 with
a copy of itself. Both files also carry `"dp": 0` in their payload, which is consistent
with them being one capture rather than two.

Of the 43 published runs, **36 have genuinely distinct per-rank captures and 7 do not**:

| Run | kind |
|---|---|
| `normal_db/dp_normal` | clean baseline |
| `normal_db/dist_optimizer_normal` | clean baseline |
| `normal_db/mixed_precision_normal` | clean baseline |
| `dp_normal_db` | clean baseline |
| `requires_grad_test_db` | fault-injected |
| `requires_grad_before_backward_test_db` | fault-injected |
| `shape_test_db` | fault-injected |

## What this does to a cross-rank check

Every cross-rank comparison on these seven traces compares rank 0 with itself. A rule of
the form

```sql
GROUP BY param_name, step HAVING COUNT(DISTINCT cksum) > 1
```

sees one distinct checksum per group by construction and **passes vacuously**. That is the
dominant rule shape in the library, including the cross-rank replication rule of §4.1, so
the effect is not limited to the 31 recovered queries that count or group by the rank
explicitly — it covers the whole cross-rank family.

Guarding is unaffected: the harness takes topology from its `--dp/--tp/--pp` arguments,
not from the trace, so `applicable_conditions` such as `{"dp": "> 1"}` still gate normally.
The rules are enabled; they just cannot find anything.

## How much of the reported ablation rests on them

From `experiments/guard_ablation/d1_results.csv` (126 cells, 42 databases):

| | false positives |
|---|---:|
| the seven duplicated-rank databases | **147** |
| all 126 cells | 1,369 |
| share | **11%** |

Narrowing to the clean arm, where "false positive" is unambiguous, the four affected clean
databases contribute **6 of the 9** false positives recorded for `lib_full`
(`dist_optimizer_normal` 2, `dp_normal` 2, `dp_normal_db` 1, `mixed_precision_normal` 1;
only `tp_normal`'s 3 come from a database with two genuine ranks).

So the leave-one-out ordering the paper reports — 342 full, 429 without π_precond, 551
without adversarial verification, 598 without π_topo — is computed with 11% of its false
positives coming from traces on which cross-rank rules cannot fire. The direction is
against the paper: with real rank-1 captures those rules could fire, so every arm's count
would be the same or higher. Whether the *ordering* survives is not something this artifact
can settle, since the guard effect is about which rules are enabled rather than about the
data. It needs a re-run.

Detection results are unaffected: they are measured on the Real-SE replays
(`benchmark/eval/real_sdc/`), which are method-level drivers, not these databases.

## Reproduce

```bash
bash scripts/fetch_trace_dbs.sh --dest /path/for/traces
cd /path/for/traces && md5sum normal_db/dp_normal/Collector/coredump_dp*.db
python3 core/validate_generated_sql.py --db /path/for/traces/normal_db/dp_normal/Collector/merged_coredump.db
```
