# The clean-run FP audit has multi-rank data that the summary omits

## What §6.2 rests on

`benchmark/eval/clean_run_fp/e_clean_baseline_summary.csv` has four rows —
`megatron_clean`, `megatron_moe`, `olmo_core_baseline`, `olmo_core_moe_hybrid` —
totalling **111,308 events and 0 fires**. All four are **single-rank**, and the two
OLMo-core rows only reach step 0 ("build snapshot + init events only").

The authors' own `e_status.md` is explicit that this is not the intended evidence base.
It lists the parallel configurations §6.2's runbook wants — DP=8, TP=2/DP=4,
TP=2/PP=2/DP=2, EP=2/DP=4 (MoE), FSDP zero3 — and marks every one **"需要 GPU 补跑"**,
needing a GPU re-run, with the driver scripts 待写, yet to be written. Its stated
expectation for all of them is **0 FP**.

## What the re-runs actually recorded

Those re-runs were done, at least in part. `_runs/` holds three 200-step multi-rank
clean runs — 2.76M events — each with the authors' own `rule_results.json`. None of
them are in the summary CSV. Reproduce with:

```
python3 benchmark/eval/clean_run_fp_multirank.py --check
```

| run | world | dp | steps | rules fired |
|---|---|---|---|---|
| `dp2_fsdp_long_r1` | 2 | fsdp | 200 | 4 |
| `dp8_ddp_r1` | 8 | ddp | 200 | 4 |
| `dp8_fsdp_r1` | 8 | fsdp | 200 | 3 |
| `smoke_r0` | 2 | fsdp | 5 | 4 |

Three rules fire on **every** run — a systematic signature, not sampling noise:

- `T0-loss-reduction-mode-correct` — *"201 loss calls with non-'mean' reduction"*
- `T1-fwd-output-block-uniformity` — 3 outliers, the same count in all three runs
- `T1-grad-flow-block-uniformity` — 1 outlier, likewise

plus `T0-norm-output-unit-rms` (2 runs) and `T0-param-update-applied` (dp8_ddp only,
*"200 optim.step events with frozen/un-updated params"*).

These runs carry no injected fault and the authors expected 0 FP from them, so every
firing here is a false positive. The dominant one looks like a framework-convention
mismatch rather than a detection failure: a rule that assumes `reduction='mean'` will
fire on any framework that reduces with `sum` and divides afterwards, on every step.
The two uniformity rules firing at an identical 3 and 1 across runs of different world
size, parallelism, and model size points the same way — a fixed structural feature,
most plausibly the first/last block legitimately differing from the middle ones.

## Denominator caveat

`n_violations` **saturates at exactly 50** in every file (checked: no value exceeds it),
while the message text carries the true count — `n_violations=50` next to "201 loss
calls". Any FP rate computed from `n_violations` is therefore a lower bound. This is why
the script reports rules-fired as the primary figure. It also means the 2.76M events here
cannot be turned into a defensible FP/1M figure without re-running the rules uncapped.

## Why this matters for the submission

Section 6.2 generalises a 0-FP result from four single-rank runs, two of which observe
only initialisation. The multi-rank evidence that exists in the authors' own workspace
does not show 0 FP. This does not mean the system is unsound — the firings look like
convention mismatches and heuristic outliers, which is a normal thing to report and
discuss — but the paper currently reports neither the runs nor the firings.

Deciding what to do is the authors' call and needs their records: whether these runs were
considered final, whether the three universal firings were already known and suppressed
by configuration, and whether an uncapped re-run is feasible before the deadline. What
this artifact can do is ship the recorded evidence rather than leave it in a workspace
directory, which it now does.

Related: [`O21_S2_ABLATION_RECOVERY.md`](O21_S2_ABLATION_RECOVERY.md) covers the separate
missing clean-FP source behind the 25.8/83.3 figures.
