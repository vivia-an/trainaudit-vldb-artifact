#!/usr/bin/env python3
"""Same-harness runner for the Real-SDC main set (E3).

Reads `real_sdc_manifest.json`, dispatches each case to the right SSH GPU host
(`eval-gpu-0` for megatron/deepspeed; `beijing-olmo-gpu` for olmo/olmo-core),
runs three phases per case (`buggy` / `reference_fixed` / `heldout_fixed`),
parses verdicts for TrainAudit / TrainCheck / Naive, and appends rows to
`real_sdc_same_harness.csv`.

This is the runner SKELETON.
- Tool invocation paths (trainaudit / traincheck / naive) are left as TODO
  hooks because each bug currently uses its own driver script. The simplest
  practical approach: each `<case>.buggy_script` is expected to itself print a
  contract line `[<case_id>] BUG DETECTED: <rule>: <msg>` or `CLEAN` or `FAIL`.
- For now this script only knows how to launch the case's own `*_run.sh` or
  `reproduce.sh` via SSH and capture the log. Three-way (TrainCheck / Naive)
  invocation is stubbed.

Usage:
    python3 run_same_harness.py --dry-run                # show plan
    python3 run_same_harness.py --case M-010             # run a single case
    python3 run_same_harness.py --all                    # run every confirmed real case
    python3 run_same_harness.py --tools trainaudit       # only TrainAudit (default)
    python3 run_same_harness.py --tools trainaudit,traincheck,naive

Exit codes:
    0 = all requested rows ran (verdict may still be FAIL)
    1 = at least one infrastructure failure (ssh / no driver)
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
REAL_SDC = ROOT / "benchmark/eval/real_sdc"
MANIFEST = REAL_SDC / "real_sdc_manifest.json"
RESULT_CSV = REAL_SDC / "real_sdc_same_harness.csv"
LOG_ROOT = REAL_SDC / "logs/same_harness"

FRAMEWORK_TO_HOST = {
    "megatron-lm": "eval-gpu-0",
    "deepspeed":   "eval-gpu-0",
    "olmo":        "beijing-olmo-gpu",
    "olmo-core":   "beijing-olmo-gpu",
}

# Phases: per docs, phase in {buggy, reference_fixed, heldout_fixed}
PHASES = ("buggy", "reference_fixed", "heldout_fixed")
ALLOWED_VERDICTS = {"DETECTED", "MISS", "CLEAN", "FAIL", "NOT_RUN"}
ALLOWED_TOOLS = {"trainaudit", "traincheck", "naive"}
CSV_FIELDS = (
    "case_id", "original_bug_id", "tool", "phase", "verdict", "detected",
    "violation_count", "total_checks", "fail_kind", "seed", "gpu_count",
    "command", "log_path",
)

def _ssh_run(host: str, cmd: str, log_path: Path, timeout: int = 1800) -> tuple[int, str]:
    """Run command on remote host with `bash -l -c`, tee output to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            host, "bash", "-l", "-c", cmd]
    print(f"  ssh {host} :: {cmd[:80]}{'...' if len(cmd) > 80 else ''}")
    with log_path.open("w") as f:
        try:
            r = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            return r.returncode, str(log_path)
        except subprocess.TimeoutExpired:
            f.write(f"\n=== TIMEOUT after {timeout}s ===\n")
            return 124, str(log_path)

def _parse_contract(log_path: Path, case_id: str, tool: str) -> dict:
    """Parse the contract line printed by the driver."""
    verdict = "FAIL"
    fail_kind = ""
    detected = False
    violation_count = ""
    total_checks = ""
    try:
        text = log_path.read_text(errors="replace")
    except FileNotFoundError:
        return dict(verdict="NOT_RUN", detected=False, fail_kind="no_log",
                    violation_count="", total_checks="")
    tag = f"[{case_id}]"
    for line in text.splitlines():
        if tag not in line:
            continue
        if "BUG DETECTED" in line:
            verdict = "DETECTED"; detected = True
            # try to capture violation_count/total_checks if present:
            # format e.g. `[X] BUG DETECTED: rule (2/433 violations)`
            if "/" in line and "violation" in line.lower():
                try:
                    seg = line.split(":")[-1]
                    parts = seg.split("(")[-1].split(")")[0].split("/")
                    if len(parts) == 2:
                        violation_count = parts[0].strip()
                        total_checks = parts[1].split()[0]
                except Exception:
                    pass
            break
        if "CLEAN" in line:
            verdict = "CLEAN"; detected = False; break
        if "FAIL" in line:
            verdict = "FAIL"; fail_kind = line.split("FAIL:")[-1].strip(); break
    return dict(verdict=verdict, detected=detected, fail_kind=fail_kind,
                violation_count=violation_count, total_checks=total_checks)

def _command_for(case: dict, tool: str, phase: str) -> tuple[str, str]:
    """Return (command_to_run_on_remote, expected_log_filename).

    Skeleton implementation:
    - For `trainaudit`, we currently run the case's `*_run.sh` (which itself
      checks out the right commit). Phase is encoded by env var so the driver
      can pick buggy vs fixed.
    - For `traincheck` / `naive`, this is a stub — fill in once the per-case
      adapters are wired.
    """
    cid = case["case_id"]
    fw  = case["framework"]
    script_field = "buggy_script" if phase == "buggy" else "fixed_script"
    script = case.get(script_field)
    if not script:
        return "", ""
    full_script = ROOT / script
    abs_script = str(full_script)

    # Per-framework env vars expected by the existing reproduce.sh / trainaudit_run.sh
    fw_env = {
        "megatron-lm": f"MEGATRON_DIR={ROOT}/exp/frameworks/Megatron-LM",
        "deepspeed":   f"DS_DIR={ROOT}/exp/frameworks/DeepSpeed",
        "olmo":        f"OLMO_DIR={ROOT}/exp/frameworks/OLMo",
        "olmo-core":   f"OLMO_CORE_DIR={ROOT}/exp/frameworks/OLMo-core",
    }.get(fw, "")

    # Phase encoding: existing reproduce.sh always runs buggy+fixed. We can't
    # cleanly split into buggy/reference_fixed/heldout_fixed without per-case
    # modifications. For now, we re-run with different seeds for heldout.
    seed = {"buggy": 0, "reference_fixed": 1, "heldout_fixed": 2}[phase]
    tool_env = {
        "trainaudit": "TRAINAUDIT=1",
        "traincheck": "TRAINCHECK=1",
        "naive":      "NAIVE=1",
    }[tool]
    cmd = (
        f"source /etc/shinit_v2 2>/dev/null || true; "
        f"export PATH=/opt/venv/bin:/usr/local/bin:$PATH; "
        f"export {fw_env} {tool_env} SAME_HARNESS_PHASE={phase} SAME_HARNESS_SEED={seed}; "
        f"bash {abs_script}"
    )
    log_name = f"{cid}__{tool}__{phase}.log"
    return cmd, log_name

def _row_for(case: dict, tool: str, phase: str, verdict_info: dict,
             cmd: str, log_path: str, gpu_count: int = 1, seed: int = 0) -> dict:
    return {
        "case_id": case["case_id"],
        "original_bug_id": case.get("original_bug_id") or case["case_id"],
        "tool": tool,
        "phase": phase,
        "verdict": verdict_info["verdict"],
        "detected": verdict_info["detected"],
        "violation_count": verdict_info["violation_count"],
        "total_checks": verdict_info["total_checks"],
        "fail_kind": verdict_info["fail_kind"],
        "seed": seed,
        "gpu_count": gpu_count,
        "command": cmd,
        "log_path": log_path,
    }

def load_manifest() -> list[dict]:
    with MANIFEST.open() as f:
        m = json.load(f)
    # Only run the cases that are committed to be real today.
    return list(m.get("cases_confirmed_real", []))

def write_rows(rows: list[dict]) -> None:
    new = not RESULT_CSV.exists()
    with RESULT_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", help="single case_id; default all confirmed_real")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tools", default="trainaudit",
                   help="comma-separated subset of trainaudit,traincheck,naive")
    p.add_argument("--phases", default=",".join(PHASES))
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    phases = [ph.strip() for ph in args.phases.split(",") if ph.strip()]
    bad = (set(tools) - ALLOWED_TOOLS) | (set(phases) - set(PHASES))
    if bad:
        print(f"ERROR: unknown tool/phase: {bad}", file=sys.stderr)
        sys.exit(2)

    cases = load_manifest()
    if args.case:
        cases = [c for c in cases if c["case_id"] == args.case]
        if not cases:
            print(f"ERROR: case_id {args.case} not in cases_confirmed_real", file=sys.stderr)
            sys.exit(2)
    if not args.all and not args.case:
        print("Tip: pass --all or --case <ID>. Listing plan only:\n")

    print(f"== plan: {len(cases)} cases × {len(tools)} tools × {len(phases)} phases ==")
    rows: list[dict] = []
    issues = 0
    for case in cases:
        host = FRAMEWORK_TO_HOST.get(case["framework"])
        if host is None:
            print(f"!! no host for framework={case['framework']} (case {case['case_id']})")
            issues += 1
            continue
        for tool in tools:
            for phase in phases:
                cmd, log_name = _command_for(case, tool, phase)
                if not cmd:
                    print(f"  - skip {case['case_id']}/{tool}/{phase}: no script wired")
                    continue
                log_path = LOG_ROOT / case["case_id"] / log_name
                if args.dry_run or (not args.all and not args.case):
                    print(f"  DRY: {case['case_id']:>10s}  {tool:>10s} {phase:>16s}  host={host}")
                    continue
                rc, lp = _ssh_run(host, cmd, log_path, timeout=args.timeout)
                if rc != 0:
                    issues += 1
                vinfo = _parse_contract(Path(lp), case["case_id"], tool)
                rows.append(_row_for(case, tool, phase, vinfo, cmd, lp))
    if rows:
        write_rows(rows)
        print(f"\nappended {len(rows)} rows to {RESULT_CSV}")
    if issues and not args.dry_run:
        print(f"\nfinished with {issues} infrastructure issue(s).", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
