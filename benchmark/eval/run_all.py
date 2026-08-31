"""D1: run trainaudit on each bug in a subset and collect results.csv.

Two execution modes:

  --mode subprocess   invoke `benchmark/bugs/<id>/trainaudit_run.sh` per bug.
                      Each script runs the real framework on the buggy
                      and fixed commits and prints a contract line:
                        [BUG_ID] BUG DETECTED: <rule_id>: <message>
                        [BUG_ID] CLEAN: <message>
                        [BUG_ID] FAIL: <stage>: <error>
                      This mode requires per-framework env vars (DS_DIR,
                      MEGATRON_DIR, etc.) and typically GPU.

  --mode synthetic    Use in-process bug surrogate runners (under
                      benchmark/eval/synthetic_runners.py) that build a
                      real PyTorch model + inject the bug pattern + run
                      trainaudit + diagnose. Reproducible, CPU-only.
                      Used for paper-table-coverage demos and harness
                      self-test on machines without framework checkouts.

Output:
  --out          results.csv   per-bug rows (one line per buggy/fixed run)
  --failures-out failures.csv  per-failure with fail_stage + stderr tail
  --table-out    paper_table_coverage.md   per-framework summary aligned with paper §4.1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---- Result row schema ---------------------------------------------------

FIELDS = [
    "bug_id", "framework", "category", "phase",  # phase ∈ {buggy, fixed}
    "verdict",   # DETECTED | CLEAN | FAIL
    "rule_id",   # which rule fired (DETECTED only)
    "duration_s",
    "fail_stage",
    "message",
]


def _row(**kwargs) -> Dict[str, Any]:
    out = {k: "" for k in FIELDS}
    out.update(kwargs)
    return out


# ---- subprocess mode -----------------------------------------------------

_LINE_DETECT = re.compile(r"^\[(?P<id>[^\]]+)\]\s*(?P<verdict>BUG DETECTED|CLEAN|FAIL)"
                          r"(?:\s*:\s*(?P<rest>.*))?$")


def _parse_run_output(stdout: str, default_id: str) -> List[Tuple[str, str, str]]:
    """Extract every well-formed contract line; tolerate noise around it."""
    out: List[Tuple[str, str, str]] = []
    for line in stdout.splitlines():
        m = _LINE_DETECT.match(line.strip())
        if m:
            out.append((m.group("id"), m.group("verdict"),
                        m.group("rest") or ""))
    if not out:
        # Driver didn't produce a contract line — synthesize a FAIL
        out.append((default_id, "FAIL",
                    "no contract line ([ID] BUG DETECTED|CLEAN|FAIL)"))
    return out


def _run_subprocess(bug_dir: Path, bug_id: str,
                    timeout_s: int = 600) -> Dict[str, Any]:
    script = bug_dir / "trainaudit_run.sh"
    if not script.exists():
        return {"bug_id": bug_id, "duration_s": 0,
                "outputs": [(bug_id, "FAIL", f"trainaudit_run.sh not found")]}
    t0 = time.time()
    try:
        proc = subprocess.run(["bash", str(script)],
                               cwd=str(bug_dir), capture_output=True,
                               timeout=timeout_s, text=True)
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        out = proc.stdout + ("\n" + proc.stderr if not ok else "")
        outputs = _parse_run_output(out, bug_id)
        return {"bug_id": bug_id, "duration_s": elapsed,
                "outputs": outputs, "stderr_tail": proc.stderr[-1000:]}
    except subprocess.TimeoutExpired:
        return {"bug_id": bug_id, "duration_s": timeout_s,
                "outputs": [(bug_id, "FAIL", f"timeout after {timeout_s}s")],
                "stderr_tail": ""}
    except Exception as e:  # noqa: BLE001
        return {"bug_id": bug_id, "duration_s": time.time() - t0,
                "outputs": [(bug_id, "FAIL", f"{type(e).__name__}: {e}")],
                "stderr_tail": ""}


# ---- synthetic mode ------------------------------------------------------


def _run_synthetic(bug_id: str) -> Dict[str, Any]:
    """In-process surrogate runner for a known bug pattern."""
    # synthetic_runners lives next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from synthetic_runners import run_synthetic_bug  # type: ignore[import-not-found]
    t0 = time.time()
    outputs = run_synthetic_bug(bug_id)
    return {"bug_id": bug_id, "duration_s": time.time() - t0,
            "outputs": outputs, "stderr_tail": ""}


# ---- main loop -----------------------------------------------------------


def run_subset(subset: List[str], manifest: Dict[str, Dict[str, Any]],
               mode: str, bugs_root: Path,
               timeout_s: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bug_id in subset:
        meta = manifest.get(bug_id, {})
        framework = meta.get("framework", "?")
        category = meta.get("category", "?")
        if mode == "subprocess":
            res = _run_subprocess(bugs_root / bug_id, bug_id,
                                   timeout_s=timeout_s)
        else:
            res = _run_synthetic(bug_id)
        for parsed_id, verdict, rest in res["outputs"]:
            phase = "buggy"
            # Drivers may print TWO contract lines: first for buggy commit,
            # second for fixed. Distinguish by ordering: 1st BUG-DETECTED-ish
            # → buggy; 2nd CLEAN-ish → fixed. Fallback: treat all as buggy.
            verdict_norm = verdict.replace(" ", "_").upper()
            if verdict_norm == "BUG_DETECTED":
                row = _row(bug_id=parsed_id, framework=framework,
                           category=category, phase=phase,
                           verdict="DETECTED",
                           rule_id=_extract_rule_id(rest),
                           duration_s=f"{res['duration_s']:.2f}",
                           message=rest[:200])
            elif verdict_norm == "CLEAN":
                row = _row(bug_id=parsed_id, framework=framework,
                           category=category,
                           phase="fixed" if any(r["bug_id"] == parsed_id
                                                 and r["phase"] == "buggy"
                                                 for r in rows) else "buggy",
                           verdict="CLEAN",
                           duration_s=f"{res['duration_s']:.2f}",
                           message=rest[:200])
            else:
                row = _row(bug_id=parsed_id, framework=framework,
                           category=category, phase=phase,
                           verdict="FAIL",
                           duration_s=f"{res['duration_s']:.2f}",
                           fail_stage=_classify_fail_stage(rest),
                           message=rest[:200])
            rows.append(row)
    return rows


def _extract_rule_id(message: str) -> str:
    """If the BUG DETECTED message starts with 'T0-...:', pull out the rule."""
    m = re.match(r"^([A-Z]\d-[\w\-]+):", message)
    return m.group(1) if m else ""


def _classify_fail_stage(message: str) -> str:
    msg = message.lower()
    if "checkout" in msg or "git" in msg:
        return "checkout"
    if "import" in msg or "modulenotfound" in msg:
        return "install"
    if "timeout" in msg:
        return "timeout"
    if "trainaudit_run.sh not found" in msg:
        return "missing_driver"
    if "no contract line" in msg:
        return "no_contract_line"
    if "cuda" in msg or "gpu" in msg:
        return "no_gpu"
    return "runtime"


# ---- output writers ------------------------------------------------------


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def write_paper_table(path: Path, rows: List[Dict[str, Any]],
                      subset_name: str) -> None:
    by_fw_cat: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    by_fw: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["phase"] != "buggy":
            continue
        by_fw_cat[(r["framework"], r["category"])][r["verdict"]] += 1
        by_fw[r["framework"]][r["verdict"]] += 1

    out: List[str] = []
    out.append(f"# Paper coverage table — {subset_name}\n\n")
    n_buggy = sum(1 for r in rows if r["phase"] == "buggy")
    n_det = sum(1 for r in rows if r["phase"] == "buggy"
                and r["verdict"] == "DETECTED")
    n_fail = sum(1 for r in rows if r["phase"] == "buggy"
                  and r["verdict"] == "FAIL")
    n_fixed_clean = sum(1 for r in rows if r["phase"] == "fixed"
                         and r["verdict"] == "CLEAN")
    n_fixed_total = sum(1 for r in rows if r["phase"] == "fixed")
    fp_rate = (1 - n_fixed_clean / n_fixed_total) if n_fixed_total else 0.0
    out.append(f"- Buggy runs: **{n_buggy}**\n")
    out.append(f"- Buggy detected: **{n_det}** ({n_det / max(n_buggy,1):.1%})\n")
    out.append(f"- Buggy failed (driver/install/etc): {n_fail}\n")
    out.append(f"- Fixed-commit FP rate: {fp_rate:.1%} "
               f"({n_fixed_total - n_fixed_clean}/{n_fixed_total})\n\n")

    out.append("## By framework\n\n")
    out.append("| framework | buggy_runs | detected | failed | det_rate |\n")
    out.append("|---|---:|---:|---:|---:|\n")
    for fw in sorted(by_fw):
        c = by_fw[fw]
        b = c["DETECTED"] + c["CLEAN"] + c["FAIL"]
        d = c["DETECTED"]
        out.append(f"| {fw} | {b} | {d} | {c['FAIL']} | "
                   f"{d/max(b,1):.1%} |\n")

    out.append("\n## By framework × category\n\n")
    out.append("| framework | category | buggy_runs | detected | det_rate |\n")
    out.append("|---|---|---:|---:|---:|\n")
    for (fw, cat) in sorted(by_fw_cat):
        c = by_fw_cat[(fw, cat)]
        b = c["DETECTED"] + c["CLEAN"] + c["FAIL"]
        d = c["DETECTED"]
        out.append(f"| {fw} | {cat} | {b} | {d} | {d/max(b,1):.1%} |\n")
    path.write_text("".join(out))


def write_failures(path: Path, rows: List[Dict[str, Any]]) -> None:
    fail_rows = [r for r in rows if r["verdict"] == "FAIL"]
    write_csv(path, fail_rows)


# ---- CLI -----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True,
                    help="path to selected_32.json or selected_48.json")
    ap.add_argument("--manifest", default="benchmark/eval/manifest.json")
    ap.add_argument("--bugs-root", default="benchmark/bugs")
    ap.add_argument("--mode", choices=["subprocess", "synthetic"],
                    default="synthetic")
    ap.add_argument("--out", default="benchmark/eval/results.csv")
    ap.add_argument("--failures-out", default="benchmark/eval/failures.csv")
    ap.add_argument("--table-out",
                    default="benchmark/eval/paper_table_coverage.md")
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    subset = json.loads(Path(args.subset).read_text())
    bug_ids = subset["bugs"]
    manifest_rows = json.loads(Path(args.manifest).read_text())
    manifest = {r["bug_id"]: r for r in manifest_rows}

    rows = run_subset(bug_ids, manifest, args.mode,
                       Path(args.bugs_root), args.timeout_s)
    write_csv(Path(args.out), rows)
    write_failures(Path(args.failures_out), rows)
    write_paper_table(Path(args.table_out), rows, subset.get("name", "unnamed"))

    n_det = sum(1 for r in rows if r["phase"] == "buggy"
                and r["verdict"] == "DETECTED")
    n_buggy = sum(1 for r in rows if r["phase"] == "buggy")
    print(f"results: {len(rows)} rows -> {args.out}")
    print(f"detected: {n_det}/{n_buggy} buggy runs")
    print(f"paper table: {args.table_out}")


if __name__ == "__main__":
    main()
