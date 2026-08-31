"""commit_regression_check — run trainaudit at two commits, diff violations.

Usage:
  python commit_regression_check.py \
    --framework-dir /path/to/Megatron-LM \
    --commit-a main \
    --commit-b $PR_HEAD_SHA \
    --driver benchmark/bugs/B11/trainaudit_driver.py \
    --env DS_DIR=/path/to/DeepSpeed

Workflow:
  1. cd framework_dir, git checkout commit_a, torchrun driver, capture stdout
  2. Parse `[BUG_ID/trainaudit] BUG DETECTED via N rule(s)` blocks → set of
     (rule_id, count) per commit
  3. Same for commit_b
  4. Diff:
       new_at_b      = rules fired at commit_b but not at commit_a   ← REGRESSION
       fixed_at_b    = rules fired at commit_a but not at commit_b   ← bug fixed
       persistent    = rules fired at both
  5. Exit code 1 if new_at_b non-empty.

Exit codes:
  0  no regression (commit_b violations ⊆ commit_a violations)
  1  regression detected (new violations on commit_b)
  2  driver run failed at one or both commits
  3  parse error / no contract line found
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Matches the contract line the trainaudit drivers emit, e.g.:
#   [B11/trainaudit] BUG DETECTED via 1 rule(s):
#      - T0-clip-grad-bounded: 2 clip_grad calls left grad norm > max_norm
_BUG_DETECTED_HDR = re.compile(
    r"^\[(?P<id>[\w-]+)(?:/[\w-]+)?\]\s*BUG DETECTED")
_RULE_LINE = re.compile(
    r"^\s*-\s*(?P<rule>T[0-9]-[\w-]+):\s*(?P<msg>.+)$")
_PHASE_HDR = re.compile(
    r"^=====\s*TrainAudit(?:\s+T\d)?\s+on\s+(BUGGY|FIXED|COMMIT_[AB])\s*\(.+\)\s*=====")


@dataclass
class CommitRun:
    commit: str
    rc: int
    rules_fired: Dict[str, int] = field(default_factory=dict)
    """rule_id → number of distinct events that violated."""
    raw_log_path: Optional[str] = None
    error: str = ""


_DEFAULT_SHARED_TMP = os.environ.get(
    "TRAINAUDIT_SHARED_TMP",
    str(Path(tempfile.gettempdir()) / "trainaudit_regr"),
)


def run_driver_on_commit(framework_dir: Path, commit: str,
                          driver_path: Path, *,
                          extra_env: Optional[Dict[str, str]] = None,
                          via_ssh: Optional[str] = None,
                          repo_root: Optional[Path] = None,
                          torchrun_nproc: int = 1,
                          timeout_s: int = 600,
                          log_dir: Optional[Path] = None) -> CommitRun:
    """Checkout `commit`, torchrun the driver, capture log.

    Stages a self-contained bash script to a configurable temporary path and runs it via
    `ssh <host> 'bash -l <script>'`. This avoids local shell quoting of
    `$RANDOM`, `${PYTHONPATH:-}`, etc. Set `TRAINAUDIT_SHARED_TMP` when the
    remote host requires a shared filesystem."""
    # Default to shared tmp if running over SSH (so remote sees the same path)
    if log_dir is None:
        log_dir = Path(_DEFAULT_SHARED_TMP if via_ssh
                        else tempfile.mkdtemp(prefix="trainaudit_regr_"))
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_commit = commit.replace("/", "_").replace("~", "_")
    log_path = log_dir / f"{safe_commit}.log"
    script_path = log_dir / f"run_{safe_commit}.sh"

    repo_root_str = str(repo_root) if repo_root else os.environ.get(
        "TRAINAUDIT_REPO",
        str(Path(__file__).resolve().parents[2] / "core" / "trainaudit_pkg"),
    )

    env_lines = "\n".join(f"export {k}={v}"
                           for k, v in (extra_env or {}).items())

    script = f"""#!/bin/bash
set -uo pipefail
source /etc/shinit_v2 2>/dev/null || true
export PATH="/opt/venv/bin:/usr/local/bin:$PATH"
{env_lines}
cd "{framework_dir}"
ORIG=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)
git checkout -q "{commit}" || {{ echo "checkout failed" >&2; exit 65; }}
trap 'cd "{framework_dir}" && git checkout -q "$ORIG" 2>/dev/null || true' EXIT
export PYTHONPATH="{repo_root_str}:{framework_dir}:${{PYTHONPATH:-}}"
PORT=$((29700 + RANDOM % 200))
torchrun --nproc_per_node={torchrun_nproc} --master_port=$PORT \\
    "{driver_path}" 2>&1 | tee "{log_path}"
"""
    script_path.write_text(script)
    script_path.chmod(0o755)

    if via_ssh:
        cmd = ["ssh", via_ssh, f"bash -l '{script_path}'"]
    else:
        cmd = ["bash", "-l", str(script_path)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout_s)
        rc = proc.returncode
        if rc != 0 and not log_path.exists():
            # Capture stderr into log so parsing still works
            log_path.write_text(proc.stdout + "\n----STDERR----\n" + proc.stderr)
    except subprocess.TimeoutExpired:
        return CommitRun(commit=commit, rc=124,
                          error=f"timeout after {timeout_s}s",
                          raw_log_path=str(log_path))
    except Exception as e:  # noqa: BLE001
        return CommitRun(commit=commit, rc=2,
                          error=f"{type(e).__name__}: {e}",
                          raw_log_path=str(log_path))

    rules = parse_violations(log_path)
    return CommitRun(commit=commit, rc=rc, rules_fired=rules,
                      raw_log_path=str(log_path))


def parse_violations(log_path: Path) -> Dict[str, int]:
    """Extract rule_id → count from a trainaudit driver log."""
    if not log_path.exists():
        return {}
    rules: Dict[str, int] = {}
    in_detect_block = False
    with log_path.open(errors="ignore") as f:
        for line in f:
            if _BUG_DETECTED_HDR.match(line.strip()):
                in_detect_block = True
                continue
            if in_detect_block:
                m = _RULE_LINE.match(line)
                if m:
                    rule_id = m.group("rule")
                    msg = m.group("msg")
                    # Try to extract violation count from message; fall back to 1
                    count_match = re.search(r"^(\d+)\s+", msg)
                    n = int(count_match.group(1)) if count_match else 1
                    rules[rule_id] = rules.get(rule_id, 0) + n
                elif line.strip() == "" or "/trainaudit]" in line:
                    in_detect_block = False
    return rules


@dataclass
class RegressionReport:
    commit_a: str
    commit_b: str
    rules_a: Dict[str, int]
    rules_b: Dict[str, int]
    new_at_b: Dict[str, int]      # rules in B but not A — NEW silent error
    fixed_at_b: Dict[str, int]    # rules in A but not B — bug fixed
    persistent: Dict[str, Tuple[int, int]]  # rule → (count_a, count_b)
    rc_a: int
    rc_b: int

    def has_regression(self) -> bool:
        return bool(self.new_at_b)

    def print_human(self):
        print()
        print(f"=== TrainAudit commit regression check ===")
        print(f"commit_a: {self.commit_a}  (rc={self.rc_a}, "
              f"violations={sum(self.rules_a.values())})")
        print(f"commit_b: {self.commit_b}  (rc={self.rc_b}, "
              f"violations={sum(self.rules_b.values())})")
        print()
        if self.new_at_b:
            print("🚨 REGRESSION — new rule(s) fire on commit_b that did NOT "
                  "fire on commit_a:")
            for r, n in sorted(self.new_at_b.items()):
                print(f"  + {r}  ({n} event{'' if n==1 else 's'})")
            print()
        if self.fixed_at_b:
            print("✅ FIXED — rule(s) fired on commit_a but no longer on "
                  "commit_b:")
            for r, n in sorted(self.fixed_at_b.items()):
                print(f"  - {r}  ({n} event{'' if n==1 else 's'} on a)")
            print()
        if self.persistent:
            print("➖ PERSISTENT — rule(s) fire on both commits:")
            for r, (a, b) in sorted(self.persistent.items()):
                print(f"    {r}  (a={a}, b={b})")
            print()
        if not self.new_at_b and not self.fixed_at_b and not self.persistent:
            print("✓ no rule fires on either commit — both clean")


def diff_runs(a: CommitRun, b: CommitRun) -> RegressionReport:
    rules_a = a.rules_fired
    rules_b = b.rules_fired
    keys_a = set(rules_a)
    keys_b = set(rules_b)
    new_at_b = {r: rules_b[r] for r in (keys_b - keys_a)}
    fixed_at_b = {r: rules_a[r] for r in (keys_a - keys_b)}
    persistent = {r: (rules_a[r], rules_b[r]) for r in (keys_a & keys_b)}
    return RegressionReport(
        commit_a=a.commit, commit_b=b.commit,
        rules_a=rules_a, rules_b=rules_b,
        new_at_b=new_at_b, fixed_at_b=fixed_at_b,
        persistent=persistent, rc_a=a.rc, rc_b=b.rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework-dir", required=True,
                    help="git repo root of the framework under test")
    ap.add_argument("--commit-a", required=True, help="baseline commit (e.g. main)")
    ap.add_argument("--commit-b", required=True, help="candidate commit (e.g. PR head)")
    ap.add_argument("--driver", required=True,
                    help="path to trainaudit_driver.py for this framework")
    ap.add_argument("--env", action="append", default=[],
                    help="env var, e.g. --env DS_DIR=/x/y. May repeat.")
    ap.add_argument("--via-ssh", default=None,
                    help="run on remote host via SSH (e.g. eval-gpu-0)")
    ap.add_argument("--nproc-per-node", type=int, default=1)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--report-out", default=None,
                    help="optional path to write JSON report")
    args = ap.parse_args()

    extra_env: Dict[str, str] = {}
    for kv in args.env:
        if "=" not in kv:
            sys.exit(f"--env expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        extra_env[k] = v

    framework_dir = Path(args.framework_dir).resolve()
    driver = Path(args.driver).resolve()
    if args.log_dir:
        log_dir = Path(args.log_dir)
    elif args.via_ssh:
        # Remote needs to see the same path → use shared filesystem
        log_dir = Path(_DEFAULT_SHARED_TMP)
    else:
        log_dir = Path(tempfile.mkdtemp(prefix="trainaudit_regr_"))
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[regr] commit_a: {args.commit_a}")
    a = run_driver_on_commit(framework_dir, args.commit_a, driver,
                              extra_env=extra_env, via_ssh=args.via_ssh,
                              torchrun_nproc=args.nproc_per_node,
                              timeout_s=args.timeout_s, log_dir=log_dir)
    print(f"  rc={a.rc} rules_fired={a.rules_fired}")

    print(f"[regr] commit_b: {args.commit_b}")
    b = run_driver_on_commit(framework_dir, args.commit_b, driver,
                              extra_env=extra_env, via_ssh=args.via_ssh,
                              torchrun_nproc=args.nproc_per_node,
                              timeout_s=args.timeout_s, log_dir=log_dir)
    print(f"  rc={b.rc} rules_fired={b.rules_fired}")

    report = diff_runs(a, b)
    report.print_human()

    if args.report_out:
        Path(args.report_out).write_text(json.dumps({
            "commit_a": report.commit_a,
            "commit_b": report.commit_b,
            "rc_a": report.rc_a, "rc_b": report.rc_b,
            "rules_a": report.rules_a, "rules_b": report.rules_b,
            "new_at_b": report.new_at_b,
            "fixed_at_b": report.fixed_at_b,
            "persistent": {r: list(v) for r, v in report.persistent.items()},
            "regression": report.has_regression(),
        }, indent=2))

    if a.rc != 0 or b.rc != 0:
        return 2
    if report.has_regression():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
