# What can and cannot be re-run, and why

Established by actually running the shipped verifier against a shipped trace, not by
reading the code. Recorded here because two of my earlier notes overstated what the
artifact supports.

## The constraint libraries carry no SQL

`core/config/ablation_libraries/lib_{full,no_topo,no_precond,no_adversarial}.json` each
hold **249 rules across 9 categories**, differing only in which guard is stripped — that
part matches §5.3 exactly. But every rule's `logic` field is **empty**:

```
lib_full.json                rules=249   empty_logic=249  (100%)
lib_no_topo.json             rules=249   empty_logic=249  (100%)
lib_no_precond.json          rules=249   empty_logic=249  (100%)
lib_no_adversarial.json      rules=249   empty_logic=249  (100%)
config/dynamic_constraints.json  rules=162   empty_logic=162  (100%)
```

The same is true of the originals in the research workspace, so this is not a copying
artifact. What each rule does carry is `name`, `description`, `type`, `tables`, and
`applicable_conditions` — the guard, e.g. `{"dp": "> 1", "stage": "= 'model-after-optimizer-step'"}`.

## The SQL is synthesised per run by an LLM

The invocation each ablation cell uses is

```
python3 -m sdccheck <trace.db> --dp N --tp N --pp N \
    --constraints-file <lib>.json --provider deepseek --reports-out <out>.json
```

and in `core/main.py` that path is `PredefinedConstraintGenerator` (loads the rule specs)
followed by `LLMSQLGenerator.generate(constraint)`, which delegates to `SQLAgent` — an LLM
call per constraint. With no API key the agent returns nothing, `execute("")` returns
`None`, and every check reports
`检查过程中出错: 'NoneType' object has no attribute 'fetchdf'`.

Observed on `normal_db/dp_normal` with `lib_full`: **93 report entries, all `error`** —
the entry count matches the recorded cell's `total=93` exactly, so the library and harness
line up; only the SQL is absent. The recorded cell is `pass=83, fail_FP=2, error=8`.

## Consequences

1. **Re-running the guard ablation needs a DeepSeek API key**, not just the traces. My
   earlier note that "§5.3 is re-runnable off this repository" was wrong on that point;
   the traces, libraries, harness and driver scripts are all here, but the SQL is not.
2. **It is not bit-reproducible.** Two runs of the same cell can generate different SQL
   for the same rule, so `pass`/`fail_FP`/`error` counts need not match
   `d1_results.csv` exactly.
3. **A reviewer may read this as a different mechanism than §4.5 describes.** The paper
   says the verifier "compiles every constraint into a parameterized SQL query" before
   the first training step, with $\pi_{\text{topo}}$ and $\pi_{\text{precond}}$ becoming
   `WHERE` filters and $\pi_{\text{schema}}$ the `HAVING` condition, and attributes the
   2–3 ms per query to DuckDB planning. That reads as a deterministic compiler. In the
   shipped code the translation from rule to SQL is an LLM call.

   These are reconcilable — the LLM step can be a one-off compilation whose output is
   then executed deterministically every step, which is consistent with both the 2–3 ms
   figure and the paper's determinism claim. But **the compiled SQL is exactly what is
   missing from the artifact**, so a reviewer cannot see the deterministic object the
   paper describes. Shipping the generated SQL per rule would resolve this and is the
   single highest-value addition left.

## What is fully re-runnable today

| Runs offline, no key | Command |
|---|---|
| mining-funnel counts | `python3 core/scripts/reproduce_funnel_counts.py` |
| catalog integrity | `sha256sum -c core/config/frozen_template_catalog.sha256` |
| pipeline smoke | `python3 core/run_smoke.py` |
| every published number | `python3 scripts/verify_paper_numbers.py` |
| figure-vs-data agreement | `python3 scripts/verify_figures.py` |
| `tab:overhead` from raw logs | `python3 benchmark/injection/parse_overhead_logs.py --check` |
| Real-SE per-case outcomes | `python3 benchmark/eval/real_sdc/extract_replay_outcomes.py` |
| trace integrity | `bash scripts/fetch_trace_dbs.sh --verify-only` |
| corpus dates and time split | `python3 benchmark/eval/{resolve_record_dates,make_temporal_split}.py` |

| Needs an API key | Notes |
|---|---|
| guard ablation cells | `core/ablation_scripts/run_d1_phase3.sh`; ~250 LLM calls per cell × 126 cells |
| catalog mining | `core/run_miner.py` with `SDC_PAPER_ALIGN=1` |

| Needs GPUs | Notes |
|---|---|
| Real-SE replays | framework checkouts at the manifest's commits, plus `collector/vtimeline` |
| collector overhead | 1.2B harness on an H20-class GPU |
