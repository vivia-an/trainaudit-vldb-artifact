# The `trainaudit` package

This is the implementation the paper describes — the DSL, the mining layers, the rule
implementations, the diagnosis chain, and the verifier. It was missing from earlier versions
of this artifact, which shipped only `sdccheck`.

**The two are different systems, and both were used.**

| | `core/sdccheck/` | `core/trainaudit_pkg/` (here) |
|---|---|---|
| what a rule is | a specification whose SQL is generated per run by an LLM | an executable Python predicate |
| where the logic lives | nowhere on disk — recovered into `core/config/generated_sql.json` (GAP_AUDIT O23) | `trainaudit/rules/*.py`, 34 modules |
| guard ablation arms | `config/ablation_libraries/lib_no_{topo,precond}.json` | `trainaudit/rules_no_topo/`, `rules_no_precond/`, 34 modules each |
| used for | the 126-cell ablation in `experiments/guard_ablation/` | the DSL/mining pipeline and diagnosis |

Each rule declares its identity and integration tier, e.g.

```python
@rule(
    rule_id="T0-attention-head-uniformity",
    min_tier=Tier.T0_PYTORCH,
    families=["F-NEW"],
```

`tiers.py` defines **integration** tiers, which are not the S0–S6 *schema* tiers of
`tab:trace-schema`:

```
T0_PYTORCH      torch.distributed + nn.Module + optim base + build snapshot
T1_FW_METADATA  + framework attrs (param.tensor_model_parallel, …)
T2_FW_PRIMITIVE + framework primitives (parallel_state, engine props)
T3_FW_SPECIFIC  + framework-specific methods (finish_grad_sync, …)
T4_INSTANCE     + per-bug instance detectors
```

T0/T1 correspond to the "A0 (PyTorch layer) / A1 (framework layer)" columns of
`fig:portability_matrix`.

## State

```bash
cd core/trainaudit_pkg && PYTHONPATH=. python3 -m pytest tests -q
```

**114 of 124 tests pass.** All 10 failures are environmental, not logic: six are
`FileNotFoundError` on paths that assume the original workspace layout, and four are
`ModuleNotFoundError: No module named 'gen_driver'` — `gen_driver.py` lives in
`benchmark/injection/` here. All 33 rule modules import cleanly.
