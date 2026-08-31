# D1' Consolidation Set

> Implements [docs/v2_semantic_guided/30_d1_consolidation_brief.md](../../../docs/v2_semantic_guided/30_d1_consolidation_brief.md).
> **Last update**: 2026-05-10

## What is this

Three new bug surrogates that extend D1 from 14 → 17 bugs, completing 12/13 class coverage:

| ID  | Class         | Blueprint    | Detection invariant                                       |
|-----|---------------|--------------|-----------------------------------------------------------|
| CF1 | control_flow  | M-010        | `tracker.calls_per_step == 1` at end of each step         |
| CM1 | communication | O-014        | `metric_log_per_rank[0] == metric_log_per_rank[r]` ∀ r    |
| OF1 | offload       | D-029        | `gnorm.dtype == gnorm_after_offload.dtype`                |

Plus runner + aggregator scaffolding for the three-way comparison (TrainAudit / TrainCheck / Naïve).

## File layout

```
d1_prime/
├── README.md                              # This file
├── source_bug_mapping.json                # CF1/CM1/OF1 → blueprint bug + invariant + adaptability notes
├── CF1_buggy.py / CF1_fixed.py            # Plain surrogate (CPU-runnable)
├── _traincheck_CF1_buggy.py / _fixed.py   # TrainCheck-instrumented variant
├── CM1_*.py / _traincheck_CM1_*.py        # Same pattern for CM1
├── OF1_*.py / _traincheck_OF1_*.py        # Same pattern for OF1
├── run_d1prime_threeway.py                # Three-way runner skeleton (calls each tool)
├── aggregate_d1prime.py                   # Merges per-tool results → summary CSV + aggregate JSON
├── d1prime_summary.csv                    # ⭐ Paper-ready 17-row table
├── d1prime_aggregate.json                 # Aggregate metrics (?/17 per tool)
├── d1prime_report.md                      # Final report with §10 decision-matrix lookup
└── results/
    ├── trainaudit_d1prime.json
    ├── traincheck_d1prime.json
    └── naive_d1prime.json
```

## Sanity (CPU)

All 6 plain surrogate scripts run on CPU and produce divergent metrics:

| Surrogate | Buggy metric | Fixed metric | Diff visible to Naïve? |
|-----------|--------------|--------------|------------------------|
| CF1       | aux=−82.96 (calls/step=2) | aux=−41.48 (calls/step=1) | NO (sub-percent total loss) |
| CM1       | rank-disagreement=5.94 | rank-disagreement=0 | NO (rank 0's view unchanged) |
| OF1       | grad_norm=21.89492 | grad_norm=21.89468 | NO (~1e-5 relative) |

`_traincheck_*.py` adapters reach the instrumentation phase on CPU (3.5 MB log written) but require GPU at runtime — match paper §6 deployment.

## Running

### Naïve (works on CPU, runs immediately)
```bash
python3 benchmark/eval/d1_prime/run_d1prime_threeway.py --tool naive
```
Expected: 0/17 buggy detected, 0/17 FP — the design goal is to show Naïve cannot see these silent drifts.

### TrainCheck (requires GPU)
```bash
# 1. Set GPU environment, ensure CUDA driver + traincheck installed
# 2. For each bug, run fixed adapter to learn invariants:
python3 benchmark/eval/d1_prime/_traincheck_CF1_fixed.py
# 3. Run buggy adapter to query violations:
python3 benchmark/eval/d1_prime/_traincheck_CF1_buggy.py
# 4. TrainCheck inference pipeline (paper §6) needs to be wired into runner — see TODO in run_d1prime_threeway.py:run_traincheck
```

### TrainAudit (requires paper pipeline)
The paper TrainAudit deployment expects a framework adapter. For surrogates, the rule check can be written inline (see TODO in `run_d1prime_threeway.py:run_trainaudit`):
- CF1: P4 invocation-frequency rule on tracker.save_to_aux_losses_tracker
- CM1: P3 cross-rank-replication on metric_log_per_rank
- OF1: P1 dtype-preservation on offload round-trip

## Phase coverage (per brief §3 timeline)

| Phase | Status |
|-------|--------|
| 0 environmental survey | ✅ |
| 1 write 3 surrogates (plain + traincheck × buggy/fixed = 12 files) | ✅ |
| 1.5 sanity (CPU run + metric-divergence check) | ✅ |
| 2 three-way runner skeleton + Naïve real run | ✅ |
| 2.5 TrainAudit / TrainCheck wiring into paper pipeline | ⏸️ requires GPU box + paper team |
| 3 aggregate + paper-ready CSV/JSON | ✅ |

## Design notes

The three new surrogates are deliberately calibrated so that Naïve monitoring cannot detect the bug:

- **CF1**: aux_loss is weighted ~0.01 in the total loss, so 2x aux contributes <1% to total loss → loss-spike threshold doesn't fire.
- **CM1**: rank 0's metric is unchanged in buggy; only non-rank-0 metrics drift. Standard logging sees rank 0's view.
- **OF1**: fp16 round-trip introduces ~1e-5 relative drift in grad_norm; clip threshold drift is below NaN/Inf check sensitivity, but compounds over many steps.

This calibration is per brief §3.2: "ensure TrainCheck can learn invariants (healthy trace available); ensure Naïve cannot detect (sub-threshold drift)."
