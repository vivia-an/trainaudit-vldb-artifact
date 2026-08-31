# The compiled SQL, recovered

## Why this file exists

The constraint libraries carry rule *specifications* — name, type, guard — with an empty
`logic` field; the SQL is produced per constraint by an LLM (`SQLAgent`) at check time.
That left the deterministic object §4.5 describes ("$\pi_{\text{topo}}$ and
$\pi_{\text{precond}}$ become `WHERE` filters, and $\pi_{\text{schema}}$ becomes the
violation condition in the `HAVING` clause") absent from the artifact — the one thing a
reviewer would most want to read.

Every one of those generation calls was logged during the runs that produced the paper's
numbers. `core/extract_generated_sql.py` lifts the SQL back out of those logs, so this is
the SQL that actually ran, not a regeneration.

```
scanned 150 log(s)
  sqlagent_records    17556
  sql_recovered       16408
  distinct constraints 228
  with >1 SQL variant  193
```

`generated_sql.json` keeps, per constraint, its specification and up to three SQL variants
ordered by how often they were generated (a rule can have several because generation is
per run). 228 constraints, 0.9 MiB.

Coverage against `ablation_libraries/lib_full.json`: **151 of its 242 named rules**. The
other 91 never had SQL generated in any logged run, because their `applicable_conditions`
gated them out under the tested topologies — consistent with each recorded cell evaluating
93 constraints rather than 249. So the recovered set covers every rule that contributed to
a reported number.

## It executes, and it behaves as §4.5 says

`core/validate_generated_sql.py` runs the recovered queries against a trace. An empty
result means the constraint holds; a non-empty result *is* the violation set.

| Trace | empty | non-empty | SQL error |
|---|---:|---:|---:|
| `normal_db/dp_normal` (clean) | 203 | 12 | 13 |
| `dp_normal_db` (clean) | 203 | 12 | 13 |
| `tp_router_test_db` (fault-injected) | 171 | **44** | 13 |

**94% of the recovered SQL executes without error**, and the injected-fault trace yields
3.7× as many violating rules as a clean one — the mechanism, demonstrated from the artifact
alone.

Two caveats on reading that table:

- These are the **raw queries, with the guard not applied**. In a real run the harness first
  disables rules whose `applicable_conditions` cannot hold under the active topology. That
  is why a clean `DP=2, TP=1` trace still shows 12 non-empty: rules such as
  *model-after-backward阶段TP分片参数main_grad存在性与完整性检查* are TP-specific and would
  be gated off before execution. Running them anyway is a small demonstration of the same
  effect §4.4's SwitchMLP example describes — unguarded checks fire on correct runs.
- The 13 errors are all `Referenced column "data" not found`: queries for
  pipeline-parallel activation-tensor rules that expect trace fields above the S0 tier
  present in these databases. They are schema-tier gated, not broken.

```bash
python3 core/validate_generated_sql.py --db <trace>/Collector/merged_coredump.db
```

Regenerate the extraction (needs the research workspace's `sdccheck/logs/`):

```bash
python3 core/extract_generated_sql.py --logs /path/to/sdccheck/logs
```
