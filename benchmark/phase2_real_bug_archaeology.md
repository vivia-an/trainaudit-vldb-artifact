# Phase 2: Real-Bug Archaeology Plan

## Goal

Demonstrate trainaudit detects bugs that **really existed in production
training framework code** — not synthetic. Provide paper §4.2 evidence.

## Approach

For each target framework, mine its closed PR/issue history for
silent-error fixes. Each fix commit's parent is by definition a buggy state.
Checkout the buggy commit, run trainaudit, record whether expected rule fires.

## Targets (priority order)

### 1. OLMo-core (highest confidence)
- Repo: https://github.com/allenai/OLMo-core
- Search labels: `bug`, `silent`, `fix`, `regression`
- Closed PR keywords: "fix:", "incorrect", "wrong", "leak", "not propagated"
- Estimate: ~30-50 candidate fix commits in last 12 months

### 2. DeepSpeed
- Repo: https://github.com/microsoft/DeepSpeed
- Active project, many silent-error fixes (B11/B12 already in our 13 candidates)
- Estimate: ~50-80 candidates

### 3. Megatron-LM
- Repo: https://github.com/NVIDIA/Megatron-LM
- Many silent-error fixes (B1-B10 from our existing benchmark)
- Estimate: ~40-60 candidates

### 4. OLMo (original)
- Repo: https://github.com/allenai/OLMo
- B13/B14 from existing benchmark
- Estimate: ~20-30 candidates

## Per-bug procedure

```python
# Pseudocode for each bug
def evaluate_bug(repo, fix_commit_sha):
    parent = git_show(repo, f"{fix_commit_sha}^")
    git_checkout(repo, parent)
    trace = run_trainaudit(repo, training_config)
    fired_rules = trainaudit.run_rules(trace)
    expected_rule = classify_bug(fix_commit_diff)  # may need LLM here
    return {
        "bug_id": fix_commit_sha,
        "expected_rule": expected_rule,
        "actual_fired": [r.rule_id for r in fired_rules],
        "detected": expected_rule in actual_fired,
    }
```

## Filter criteria for "silent" vs "loud"

Many bug fixes are for crashes/asserts (loud) — exclude those. Keep:
- Numerical bugs (NaN/Inf/precision drift)
- Wrong gradient computation
- Optimizer state corruption
- Comm/sync errors that don't crash
- Init scheme bugs
- Schedule/LR bugs

Use commit message heuristics:
- INCLUDE: "silent", "incorrect", "wrong", "miscompute", "off by", "leak", "drift"
- EXCLUDE: "crash", "raise", "assert", "TypeError", "ImportError"

## Estimated workload

| Step | Time |
|---|---|
| Scrape closed PRs from 4 repos via gh CLI | 2 days |
| Filter to 30 highest-confidence "silent" bugs | 1 day |
| Per-bug: checkout, run trainaudit, score | 30min × 30 = 15 hours |
| Analyze + write up | 2 days |

Total: 1.5-2 weeks for Phase 2.

## Expected output

`benchmark/phase2/real_bug_results.json`:
```json
[
  {"bug_id": "olmo-core#1234", "expected_rule": "T0-no-nan-inf",
   "actual_fired": ["T0-no-nan-inf"], "detected": true},
  ...
]
```

Confusion matrix similar to Phase 1, but on REAL bugs. If detection rate
> 70% on this benchmark, paper §4.2 has strong evidence.

## Dependencies

- gh CLI (probably already installed)
- Trainaudit driver per framework (have hunting_driver_<fw>.py)
- ssh GPU access (have eval-gpu-0)
- Time (the constraint)
