"""Apply Naïve 4-signal detector to whatever stdout comes via stdin.

Usage: <some-driver> 2>&1 | python naive_stdin_check.py <bug_id>

Emits a single contract line:
  [<id>] BUG DETECTED: naive:<signal>:step=<t> (<n> metric points)
  [<id>] CLEAN: naive:no anomaly in <n> metric points
  [<id>] FAIL: naive:no metric lines parsed
"""
import math
import re
import statistics
import sys

K_SPIKE = 10.0
WINDOW = 20

PATTERNS = [
    re.compile(
        r"lm loss:\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)\s*\|"
        r".*?grad norm:\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)"
    ),
    re.compile(
        r"loss[=:]\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)"
        r".*?grad[_ ]norm[=:]\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)",
        re.IGNORECASE,
    ),
    re.compile(
        r"step[=:]\s*\d+.*?loss[=:]\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)"
        r".*?grad[_ ]norm[=:]\s*([\d.eE+-]+|nan|NaN|inf|Inf|-inf|-Inf)",
        re.IGNORECASE,
    ),
]


def parse_value(s):
    s = s.strip().lower()
    if s == "nan":
        return math.nan
    if s in ("inf", "+inf"):
        return math.inf
    if s == "-inf":
        return -math.inf
    return float(s)


def main():
    if len(sys.argv) < 2:
        print("usage: naive_stdin_check.py <bug_id>", file=sys.stderr)
        sys.exit(2)
    bug_id = sys.argv[1]
    stdout = sys.stdin.read()
    metrics = []
    for line in stdout.splitlines():
        for pat in PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    metrics.append((parse_value(m.group(1)),
                                    parse_value(m.group(2))))
                except ValueError:
                    pass
                break
    if not metrics:
        print(f"[{bug_id}] FAIL: naive:no metric lines parsed "
              f"({len(stdout.splitlines())} stdout lines)")
        return
    losses = [m[0] for m in metrics]
    grad_norms = [m[1] for m in metrics]
    for t, lv in enumerate(losses):
        if not math.isfinite(lv):
            print(f"[{bug_id}] BUG DETECTED: naive:loss_nan:step={t} "
                  f"({len(metrics)} pts)")
            return
    for t, gn in enumerate(grad_norms):
        if not math.isfinite(gn):
            print(f"[{bug_id}] BUG DETECTED: naive:grad_nan:step={t} "
                  f"({len(metrics)} pts)")
            return
    for t in range(WINDOW, len(losses)):
        win = losses[t - WINDOW:t]
        med = statistics.median(win) if win else 0.0
        if med > 0 and losses[t] > K_SPIKE * med:
            print(f"[{bug_id}] BUG DETECTED: naive:loss_spike:step={t} "
                  f"({len(metrics)} pts)")
            return
    for t in range(WINDOW, len(grad_norms)):
        win = grad_norms[t - WINDOW:t]
        ref = max(win) if win else 0.0
        if ref > 0 and grad_norms[t] > K_SPIKE * ref:
            print(f"[{bug_id}] BUG DETECTED: naive:gradnorm_spike:step={t} "
                  f"({len(metrics)} pts)")
            return
    print(f"[{bug_id}] CLEAN: naive:no anomaly in {len(metrics)} pts")


if __name__ == "__main__":
    main()
