# Trace schema and how the guard is actually enforced

Established by running the recovered SQL against the published traces and reading the code
path the recorded ablation actually took. Three findings, in order of how much they matter.

## 1. The §4.1 running example cannot be reproduced from this artifact

The paper threads one example through the whole system — the SwitchMLP router-weight bug,
where an unguarded cross-rank checksum check "flags 57 correctly sharded
`query_key_value.weight` tensors on a clean $\mathrm{TP}=2$ run". `benchmark/eval/CASE2_WALKTHROUGH_PI_TOPO.md`
makes it concrete: 492 parameter records emitted at `build.snapshot`, of which 57 are
TP-sharded.

Neither anchor exists in any of the 43 published traces:

| Probe | Result across all 43 traces |
|---|---|
| parameters named `*query_key_value*` | **0** — the traces use Megatron-Core naming (`linear_qkv.weight`) |
| rows at a `build.snapshot` stage | **0** — no trace has that stage, or any stage matching `%snap%`/`%build%` |

On the clean TP=2 trace that *is* shipped (`normal_db/tp_normal`, 370,270 rows, genuine TP
ranks 0 and 1) the unguarded cross-rank check yields 8,034 violating `(param, step, stage)`
groups over 198 distinct parameters — 72 of them `linear_qkv`, 96 sharded-family. No
counting of it produces 57 or 492; the run is a 12-layer model, not the one in the example.

**So the paper's central worked example is not in the supplemental material.** Everything
else checks out against the shipped data (33 numbers, 12 figure-rendered numbers), which
makes this the conspicuous hole: the one artifact a reviewer is most likely to want to
follow end to end.

## 2. There are two trace-schema generations, and the library targets both

The shipped traces use 25 stage names from the Megatron collector:

```
after-get-batch, model-before-backward, model-after-backward, main-grad-in-backward,
main-grad-after-backward, model-after-forward-mbs-0..3, main-param-after-forward-mbs-0..3,
model-before-optimizer-step, model-after-optimizer-step, main-param-before/after-optimizer-step,
optimizer-state-before/after-backward, optimizer-state-before/after-step, …
```

The paper's §4.4 hookpoints and `benchmark/eval/p9_p16_deployment/runtime_integration/hookpoint_matrix.csv`
use a different vocabulary: `build.snapshot`, `module.forward.post`, `module.fwd.pre`,
`optim.step.post`, `checkpoint.load`, `checkpoint.save`, `distributed.all_reduce`,
`loss.compute.post`.

Counting the stage names the rule guards reference, with `%`/`*` treated as wildcards:

| | count |
|---|---:|
| distinct stage names referenced by guards | 28 |
| matching some stage in the shipped traces | 10 |
| matching none | **18** |
| rule-gate references pointing at a stage no trace has | **121 of 297 (41%)** |

The unmatched names are mostly coarse phase labels (`optimizer-step` 34 rules, `forward` 27,
`backward` 22, `model-after-forward` 14, `checkpoint` 5) plus initialization and
checkpoint-restore stages. This is the same mismatch that makes 13 of the 228 recovered
queries fail with `Referenced column "data" not found` — they are written against the other
schema.

## 3. The topology gate exists in the code but was not engaged in the recorded runs

§4.5 says the verifier, before the first training step, "extracts the active topology τ
from the launch script—DP/TP/PP/EP group sizes, `zero_stage`, and mixed-precision
dtype—and disables rules whose $\pi_{\text{topo}}$ cannot hold under it."

That mechanism is implemented. `core/sdccheck/script_config_injector.py` parses the launch
script, computes `dp = world_size / (tp·pp·ep)`, evaluates each rule's
`applicable_conditions`, and stamps `_skip = "= true"` on the ones that cannot hold;
`_check_applicable_conditions` honours `_skip` and drops them.

**It only runs when `main.py` is given `--inject-script-config`.** The ablation's
invocation does not pass it:

```
python3 -u -m sdccheck "$merged" --dp "$dp" --tp "$tp" --pp "$pp" \
    --constraints-file "$lib_path" --provider deepseek --reports-out "${out_base}.json"
```

Without it `dynamic_constraints_path` stays `None`, no `_skip` is injected, and
`_evaluate_single_condition` returns `True` for every `dp`/`tp`/`pp`/`ep` condition and for
every `stage` condition — the code says so in as many words ("stage 条件需要在运行时验证，
这里先返回 True").

So in the runs behind the paper's numbers, **no rule was gated by its topology guard.** The
guard still took effect, by a different route: it is part of the constraint specification
handed to the SQL-generating LLM, which writes it into the query. That is measurable —
**54 of the 55 tp-guarded rules with recovered SQL produce SQL that references the `tp`
field**.

### What this does and does not mean

- The measured effect is real. `lib_full` 342 false positives against `lib_no_topo` 598 is
  a genuine difference, produced by stripping the guard from the specification so the
  generated SQL omits the filter.
- The *enforcement path* differs from §4.5's description. The paper describes a
  deterministic pre-step gate; these numbers came from the guard being expressed in
  LLM-generated SQL. Both are defensible designs, but a reviewer reading §4.5 and then
  running `run_d1_phase3.sh` sees a different mechanism.
- It compounds O23: because the guard reaches the query through generation, the
  reproducibility of the guard's effect inherits the generation's nondeterminism (193 of
  228 constraints have more than one recorded SQL variant).

Suggested resolution: either pass `--inject-script-config` and re-run so the numbers come
from the gate the paper describes, or describe the implemented path — guard in the
specification, compiled into the query, executed deterministically thereafter.

## Reproduce

```bash
bash scripts/fetch_trace_dbs.sh --dest /path/for/traces
python3 core/validate_generated_sql.py --db /path/for/traces/normal_db/tp_normal/Collector/merged_coredump.db
grep -rn "inject-script-config" core/ablation_scripts/    # no hits
```
