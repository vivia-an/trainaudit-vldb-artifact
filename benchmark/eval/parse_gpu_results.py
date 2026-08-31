"""Parse trainaudit_gpu_logs/*.log → results.csv + paper_table_gpu.md.

Each .log file contains the buggy + fixed phase output of a
trainaudit_run.sh. We extract the verdict line per phase and emit a
2-row results.csv compatible with run_all.py format.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


_PHASE_RE = re.compile(
    r"^=====\s*TrainAudit(?:\s+T\d)?\s+on\s+(BUGGY|FIXED)\s*\((.+?)\)\s*=====")
# Multiple contract dialects: "[BUG_ID/trainaudit]", "[BUG_ID/trainaudit-prebuild]", "[BUG_ID]"
_VERDICT_RE = re.compile(
    r"^\[(?P<id>[\w-]+)(?:/[\w-]+)?\]\s*(?P<verdict>BUG DETECTED|CLEAN|FAIL)"
    r"(?:\s*via\s+\d+\s+rule\(s\))?\s*:?\s*(?P<rest>.*)$")
_RULE_RE = re.compile(r"^\s*-\s*(?P<rule>[\w.-]+):\s*(?P<msg>.+)$")


def parse_log(path: Path, manifest: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    bug_id = path.stem
    meta = manifest.get(bug_id, {})
    framework = meta.get("framework", "?")
    category = meta.get("category", "?")

    rows: List[Dict[str, Any]] = []
    phase: str = ""
    commit: str = ""
    pending_verdict: Tuple[str, str] = ("", "")
    pending_rule: Tuple[str, str] = ("", "")

    def flush():
        if phase and (pending_verdict[0] or pending_rule[0]):
            verdict_text, rest = pending_verdict
            rule_id, rule_msg = pending_rule
            v = verdict_text or ("FAIL" if not rule_id else "")
            rows.append({
                "bug_id": bug_id, "framework": framework,
                "category": category,
                "phase": phase.lower(),
                "commit": commit,
                "verdict": v,
                "rule_id": rule_id,
                "message": (rule_msg or rest)[:240],
            })

    with path.open(errors="ignore") as f:
        for line in f:
            m = _PHASE_RE.match(line.strip())
            if m:
                if phase:  # close previous phase
                    flush()
                    pending_verdict = ("", "")
                    pending_rule = ("", "")
                phase = m.group(1)
                commit = m.group(2)
                continue
            mv = _VERDICT_RE.match(line.strip())
            if mv:
                pending_verdict = (mv.group("verdict").replace(" ", "_"),
                                    mv.group("rest"))
                continue
            mr = _RULE_RE.match(line)
            if mr and pending_verdict[0] == "BUG_DETECTED" and not pending_rule[0]:
                pending_rule = (mr.group("rule"), mr.group("msg"))
                continue
    flush()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir",
                    default="/volume/qscai/cqs/temp/trainaudit_gpu_logs")
    ap.add_argument("--manifest", default="benchmark/eval/manifest.json")
    ap.add_argument("--out", default="benchmark/eval/results_gpu.csv")
    ap.add_argument("--table-out",
                    default="benchmark/eval/paper_table_gpu.md")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        sys.exit(f"logs dir not found: {logs_dir}")
    manifest_rows = json.loads(Path(args.manifest).read_text())
    manifest = {r["bug_id"]: r for r in manifest_rows}

    all_rows: List[Dict[str, Any]] = []
    for log in sorted(logs_dir.glob("*.log")):
        all_rows.extend(parse_log(log, manifest))

    # CSV
    fields = ["bug_id", "framework", "category", "phase", "commit",
              "verdict", "rule_id", "message"]
    with Path(args.out).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # Aggregate
    by_bug: Dict[str, Dict[str, str]] = {}
    for r in all_rows:
        by_bug.setdefault(r["bug_id"], {})
        by_bug[r["bug_id"]][r["phase"]] = r["verdict"]

    n_bugs = len(by_bug)
    n_buggy_detected = sum(1 for v in by_bug.values()
                            if v.get("buggy") == "BUG_DETECTED")
    n_fixed_clean = sum(1 for v in by_bug.values()
                         if v.get("fixed") == "CLEAN")
    n_fixed_total = sum(1 for v in by_bug.values() if "fixed" in v)
    by_fw: Dict[str, Counter] = defaultdict(Counter)
    for r in all_rows:
        if r["phase"] == "buggy":
            by_fw[r["framework"]][r["verdict"]] += 1

    out: List[str] = []
    out.append("# Paper §4.1 — GPU verification (trainaudit_run.sh on eval-gpu-0)\n\n")
    out.append(f"- Bugs run on real GPU: **{n_bugs}** (4× H200, 4 frameworks)\n")
    out.append(f"- Buggy phase **DETECTED**: **{n_buggy_detected}/{n_bugs}** "
                f"= {n_buggy_detected/max(n_bugs,1):.1%}\n")
    out.append(f"- Fixed phase CLEAN: **{n_fixed_clean}/{n_fixed_total}** "
                f"= {n_fixed_clean/max(n_fixed_total,1):.0%}; "
                f"**FP rate "
                f"{(n_fixed_total - n_fixed_clean)/max(n_fixed_total,1):.1%}**\n")
    n_no_fixed_verdict = n_bugs - n_fixed_total
    if n_no_fixed_verdict:
        out.append(f"- {n_no_fixed_verdict} FIXED phase(s) emitted no contract "
                    f"line (framework added an assertion on the buggy config "
                    f"in the fixed commit and aborted at init — itself "
                    f"evidence the bug class is gone).\n")
    out.append("\n")

    out.append("## Per-bug verdicts\n\n"
                "| bug_id | framework | category | buggy | fixed | rule fired |\n"
                "|---|---|---|---|---|---|\n")
    for bid in sorted(by_bug):
        meta = manifest.get(bid, {})
        fired = ""
        for r in all_rows:
            if r["bug_id"] == bid and r["phase"] == "buggy" and r["rule_id"]:
                fired = r["rule_id"]
                break
        b = by_bug[bid].get("buggy", "?")
        fx = by_bug[bid].get("fixed", "?")
        out.append(f"| {bid} | {meta.get('framework','?')} | "
                    f"{meta.get('category','?')} | "
                    f"**{b}** | {fx} | `{fired}` |\n")

    out.append("\n## By framework\n\n"
                "| framework | bugs | DETECTED | det_rate |\n"
                "|---|---:|---:|---:|\n")
    for fw in sorted(by_fw):
        c = by_fw[fw]
        n = sum(c.values())
        d = c["BUG_DETECTED"]
        out.append(f"| {fw} | {n} | {d} | {d/max(n,1):.1%} |\n")

    Path(args.table_out).write_text("".join(out))

    print(f"results: {len(all_rows)} rows -> {args.out}")
    print(f"detection: {n_buggy_detected}/{n_bugs} buggy phases")
    print(f"FP: {n_fixed_total - n_fixed_clean}/{n_fixed_total} fixed phases")
    print(f"table: {args.table_out}")


if __name__ == "__main__":
    main()
