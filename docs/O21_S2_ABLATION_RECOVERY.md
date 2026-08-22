# What the missing S2 ablation measured

`fig:predicate-ablation` and the third column of `tab:db-baselines` report clean-trace false
positives of **25.8 FP/1M**, rising to **83.3** without π_topo, over a six-case subset and
504 K rule evaluations. `paper_v2/mechanism_precond.csv` names its source as
`benchmark/eval/ablation_s2_results.csv`, which is absent (GAP_AUDIT O21).

The script that produced it, `run_ablation_s2.py`, is also absent — but it survives as
compiled bytecode, `benchmark/eval/__pycache__/run_ablation_s2.cpython-312.pyc` in the
research checkout, and it is the **only** `.pyc` in that directory without a source file.
Bytecode retains names and string constants, so the methodology is recoverable even though
the code is not.

## Recovered methodology

| | |
|---|---|
| engine | `trainaudit.store.TraceStore` and `trainaudit.tiers.Tier` — so it runs the **trainaudit** Python rules, not `sdccheck`'s LLM-generated SQL |
| arms | `no_precond` (labelled E2) and `no_topo` (E3), against the full rule set |
| rule source | a `rules_dir`, which must be built first: *"missing — run `scripts/build_ablation_rules.py --mode all` first"* |
| traces | discovered per "SPEC §3.1", each as `{path, family, n_ranks}`, drawn from `rebuttal_v1/C1_overhead_gpu/_runs`, `c1_tp2` and `fault_injection` |
| outputs | `ablation_s2_results.csv` with `rule_id, violated, n_violations, duration_s`, and a delta-framed `paper_table_ablation_s2.md` |
| aggregation | per `(trace, setting)` and per `(trace, setting, rule_id)` |

Two of its message templates pin down the headline framing:

> Removing the `π_topo` guard on the 16 Megatron TP=2 traces fires …
> Removing the `π_precond` guard fires …

And it carried a self-check worth keeping in any reconstruction:

> single-rank traces had `delta_no_topo != 0` (must be 0)

— removing the topology guard must have **no** effect on a single-rank trace. That is a sound
invariant, and it is the same reasoning behind GAP_AUDIT O26 and the correction to
`audit_guard_groups.py`: a skipped record in a single-rank run costs nothing.

## What it would take to reproduce

Present in this artifact now:

- all three rule sets — `core/trainaudit_pkg/trainaudit/{rules,rules_no_topo,rules_no_precond}/`, 34 modules each (O45)
- `TraceStore` and `Tier`
- one Megatron TP=2 trace, published in `trace-events-v2` (O32)

Still missing:

- `run_ablation_s2.py` (bytecode only) and `scripts/build_ablation_rules.py`
- `ablation_s2_results.csv`
- the remaining Megatron TP=2 traces — the research checkout holds 5 run directories under
  `C1_overhead_gpu/_runs`, against the script's "16 Megatron TP=2 traces"

**Deliberately not attempted here.** Rewriting the script from its bytecode and re-running it
would produce numbers from my reconstruction of someone else's methodology, presented as
theirs. The same reasoning applies as to the tier-coverage mapping (O16): re-aggregating
recorded facts is fair, inventing a missing premise is not. What this file does instead is
make the reconstruction cheap for whoever holds the original.
