"""Naïve baseline subprocess driver for real-bug benchmark.

Wraps `benchmark/bugs/<id>/trainaudit_run.sh` (which runs the framework's real
training script for both buggy + fixed commits), captures the framework's
own stdout, and applies the Naïve 4-signal detector to whatever loss /
grad-norm log lines we can grep out.

Why reuse trainaudit_run.sh: the existing driver already (a) checks out the
right buggy/fixed commits, (b) sets framework-specific env (MEGATRON_DIR,
DS_DIR, OLMO_DIR), (c) launches torchrun with framework-correct args. We
discard trainaudit's contract lines and run our own metric grep over the
framework's training log (Megatron's `lm loss: X | grad norm: Y`,
DeepSpeed's `loss: X | grad norm: Y`, OLMo's per-step JSON, etc.).

Per-bug emit:
  [<bug_id>] BUG DETECTED: naive:<signal>:<step>
  [<bug_id>] CLEAN: naive:no anomaly in <N> log lines
  [<bug_id>] FAIL: naive:<reason>

Run from beijing-dev-0 over SSH (GPU machines have the framework checkouts):
  ssh eval-gpu-0 "MEGATRON_DIR=... bash benchmark/eval/naive_subprocess.sh M-014"
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

K_SPIKE = 10.0
WINDOW = 20

# Per-framework log-line patterns (Megatron / DeepSpeed / OLMo cover all 23
# real bugs in doc 26 §3.2). All extract (loss, grad_norm) per step.
PATTERNS = [
    # Megatron: "iteration X | ... | lm loss: 1.234E+00 | grad norm: 5.67E+00"
    re.compile(
        r"lm loss:\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)\s*\|"
        r".*?grad norm:\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)"
    ),
    # DeepSpeed: "loss=1.234, grad_norm=5.67" or "loss: 1.234 | grad_norm: 5.67"
    re.compile(
        r"loss[=:]\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)"
        r".*?grad[_ ]norm[=:]\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)",
        re.IGNORECASE,
    ),
    # OLMo train.py emits per-step: "step=N loss=L grad_norm=G"
    re.compile(
        r"step[=:]\s*\d+.*?loss[=:]\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)"
        r".*?grad[_ ]norm[=:]\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)",
        re.IGNORECASE,
    ),
]


def _parse_value(s: str) -> float:
    s = s.strip().lower()
    if s in ("nan",):
        return math.nan
    if s in ("inf", "+inf"):
        return math.inf
    if s in ("-inf",):
        return -math.inf
    return float(s)


def parse_metrics(stdout: str) -> List[Tuple[float, float]]:
    seen: List[Tuple[float, float]] = []
    for line in stdout.splitlines():
        for pat in PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    seen.append((_parse_value(m.group(1)),
                                 _parse_value(m.group(2))))
                except ValueError:
                    pass
                break
    return seen


def naive_detect(metrics: List[Tuple[float, float]]) -> Tuple[str, str, int]:
    losses = [m[0] for m in metrics]
    grad_norms = [m[1] for m in metrics]
    for t, lv in enumerate(losses):
        if not math.isfinite(lv):
            return "DETECTED", "loss_nan", t
    for t, gn in enumerate(grad_norms):
        if not math.isfinite(gn):
            return "DETECTED", "grad_nan", t
    for t in range(WINDOW, len(losses)):
        win = losses[t - WINDOW:t]
        med = statistics.median(win) if win else 0.0
        if med > 0 and losses[t] > K_SPIKE * med:
            return "DETECTED", "loss_spike", t
    for t in range(WINDOW, len(grad_norms)):
        win = grad_norms[t - WINDOW:t]
        ref = max(win) if win else 0.0
        if ref > 0 and grad_norms[t] > K_SPIKE * ref:
            return "DETECTED", "gradnorm_spike", t
    return "CLEAN", "", -1


def run_one(bug_dir: Path, bug_id: str, timeout_s: int) -> int:
    script = bug_dir / "trainaudit_run.sh"
    if not script.exists():
        print(f"[{bug_id}] FAIL: naive:trainaudit_run.sh not found")
        return 0
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=str(bug_dir), capture_output=True,
            text=True, timeout=timeout_s,
        )
        stdout = proc.stdout + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        print(f"[{bug_id}] FAIL: naive:timeout after {timeout_s}s")
        return 0
    elapsed = time.time() - t0

    metrics = parse_metrics(stdout)
    if not metrics:
        print(f"[{bug_id}] FAIL: naive:no loss/grad_norm log lines parsed "
              f"(framework log format mismatch); stdout had "
              f"{len(stdout.splitlines())} lines")
        return 0

    verdict, signal, step = naive_detect(metrics)
    if verdict == "DETECTED":
        print(f"[{bug_id}] BUG DETECTED: naive:{signal}:step={step} "
              f"({len(metrics)} metric points, {elapsed:.1f}s)")
    else:
        print(f"[{bug_id}] CLEAN: naive:no anomaly in {len(metrics)} "
              f"metric points ({elapsed:.1f}s)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug-id", required=True)
    ap.add_argument("--bugs-root",
                    default=str(Path(__file__).resolve().parents[1] / "bugs"))
    ap.add_argument("--timeout-s", type=int, default=900)
    args = ap.parse_args()

    bug_dir = Path(args.bugs_root) / args.bug_id
    sys.exit(run_one(bug_dir, args.bug_id, args.timeout_s))


if __name__ == "__main__":
    main()
