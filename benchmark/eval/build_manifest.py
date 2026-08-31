"""Scan benchmark/bugs/* and emit manifest.json + manifest_summary.md.

Read-only. Source of truth for D1 selected_32/48/80 subset selection.

Manifest row schema:
  bug_id, framework, category, reproduction_status, has_detect_py,
  has_reproduce_sh, has_trainaudit_driver, gpu_needed, buggy_commit,
  fixed_commit, expected_output, detection_method, root_cause,
  source_config_path, missing_fields[]

Run:
  python benchmark/eval/build_manifest.py \
      --bugs-dir benchmark/bugs \
      --out      benchmark/eval/manifest.json \
      --summary  benchmark/eval/manifest_summary.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED = ["bug_id", "framework", "category"]
OPTIONAL = [
    "reproduction_status", "buggy_commit", "fixed_commit", "expected_output",
    "detection_method", "root_cause", "gpu_needed", "trigger_conditions",
    "invariant", "severity", "title", "issue_url", "pr_url",
    "parallel_dimensions", "diff_summary", "silent_check",
]


def _load_one(bug_dir: Path) -> Optional[Dict[str, Any]]:
    cfg = bug_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        with cfg.open() as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {
            "bug_id": bug_dir.name,
            "source_config_path": str(cfg),
            "missing_fields": REQUIRED + ["<load_error>"],
            "_error": f"{type(e).__name__}: {e}",
        }

    row: Dict[str, Any] = {
        "bug_id": data.get("bug_id", bug_dir.name),
        "source_config_path": str(cfg),
    }
    missing: List[str] = []
    for k in REQUIRED:
        v = data.get(k)
        if v in (None, ""):
            missing.append(k)
        else:
            row[k] = v
    for k in OPTIONAL:
        if k in data:
            row[k] = data[k]

    row["has_detect_py"] = (bug_dir / "detect.py").exists()
    row["has_reproduce_sh"] = (bug_dir / "reproduce.sh").exists()
    row["has_trainaudit_driver"] = (bug_dir / "trainaudit_run.sh").exists()
    row["missing_fields"] = missing
    return row


def build_manifest(bugs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in sorted(bugs_root.iterdir()):
        if not entry.is_dir():
            continue
        row = _load_one(entry)
        if row is None:
            continue
        rows.append(row)
    return rows


def _summarize(rows: List[Dict[str, Any]]) -> str:
    n = len(rows)
    fw = Counter(r.get("framework", "?") for r in rows)
    status = Counter(r.get("reproduction_status", "<unset>") for r in rows)
    cat = Counter(r.get("category", "?") for r in rows)
    n_detect = sum(1 for r in rows if r.get("has_detect_py"))
    n_reproduce = sum(1 for r in rows if r.get("has_reproduce_sh"))
    n_driver = sum(1 for r in rows if r.get("has_trainaudit_driver"))
    n_repro = status.get("reproduced", 0)

    fw_x_status: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        fw_x_status[r.get("framework", "?")][r.get("reproduction_status", "<unset>")] += 1

    out: List[str] = []
    out.append("# Benchmark manifest summary\n\n")
    out.append(f"- Total bug directories with `config.json`: **{n}**\n")
    out.append(f"- `reproduction_status == reproduced`: **{n_repro}**\n")
    out.append(f"- Has `detect.py`: **{n_detect}**\n")
    out.append(f"- Has `reproduce.sh`: **{n_reproduce}**\n")
    out.append(f"- Has `trainaudit_run.sh`: **{n_driver}**\n\n")
    out.append("> ⚠️ `reproduction_status` is **incomplete**: ~10 driverable "
               "bugs verified working in `docs/v2_semantic_guided/22_paper_evidence_index.md` "
               "are still labelled `<unset>` in their `config.json`. "
               "Use `has_trainaudit_driver` as the primary 'is this bug actually runnable' signal, "
               "and treat `reproduction_status` as a hint only.\n\n")

    out.append("## By framework\n\n")
    out.append("| framework | total | reproduced | has_driver | has_detect |\n")
    out.append("|---|---:|---:|---:|---:|\n")
    for f in sorted(fw):
        t = fw[f]
        rprd = fw_x_status[f].get("reproduced", 0)
        hd = sum(1 for r in rows if r.get("framework") == f and r.get("has_trainaudit_driver"))
        hde = sum(1 for r in rows if r.get("framework") == f and r.get("has_detect_py"))
        out.append(f"| {f} | {t} | {rprd} | {hd} | {hde} |\n")

    out.append("\n## By reproduction_status\n\n")
    for s, c in sorted(status.items(), key=lambda kv: -kv[1]):
        out.append(f"- `{s}`: {c}\n")

    out.append("\n## Top 15 categories\n\n")
    for c, k in cat.most_common(15):
        out.append(f"- `{c}`: {k}\n")

    out.append("\n## Bugs with trainaudit_run.sh (D1 driver pool)\n\n")
    for r in rows:
        if r.get("has_trainaudit_driver"):
            out.append(f"- `{r['bug_id']}` ({r.get('framework', '?')}, "
                       f"status={r.get('reproduction_status', '<unset>')}, "
                       f"category={r.get('category', '?')})\n")

    return "".join(out)


def _selected_subset(rows: List[Dict[str, Any]], n: int,
                     mode: str) -> List[str]:
    """Pick a balanced subset by framework.

    Selection caveat: `reproduction_status` is unreliable — many driverable
    bugs verified working in doc 22 carry no `reproduced` annotation in
    `config.json`. So we treat `has_trainaudit_driver` as the strongest
    signal (someone wrote a driver = bug is at least drivable),
    `reproduction_status == reproduced` as secondary.

    mode:
      - "driver_only": existing 15 drivers, balanced by framework (selected_32)
      - "driver_plus_repro": drivers + reproduced-without-driver
        (selected_48 — gen_driver.py targets)
    """
    if mode == "driver_only":
        pool = [r for r in rows if r.get("has_trainaudit_driver")]
    elif mode == "driver_plus_repro":
        pool = [r for r in rows
                if r.get("has_trainaudit_driver")
                or r.get("reproduction_status") == "reproduced"]
    else:
        raise ValueError(f"unknown mode: {mode}")

    by_fw: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool:
        # Drivers come first within each framework
        by_fw[r.get("framework", "?")].append(r)
    for f in by_fw:
        by_fw[f].sort(key=lambda r: (not r.get("has_trainaudit_driver"),
                                     r.get("reproduction_status") != "reproduced",
                                     r["bug_id"]))

    picked: List[Dict[str, Any]] = []
    while len(picked) < n and any(by_fw.values()):
        for f in list(by_fw.keys()):
            if not by_fw[f]:
                continue
            picked.append(by_fw[f].pop(0))
            if len(picked) >= n:
                break
    return [r["bug_id"] for r in picked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs-dir", default="benchmark/bugs")
    ap.add_argument("--out", default="benchmark/eval/manifest.json")
    ap.add_argument("--summary", default="benchmark/eval/manifest_summary.md")
    ap.add_argument("--subset-32", default="benchmark/eval/selected_32.json")
    ap.add_argument("--subset-48", default="benchmark/eval/selected_48.json")
    args = ap.parse_args()

    bugs_root = Path(args.bugs_dir)
    rows = build_manifest(bugs_root)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    Path(args.summary).write_text(_summarize(rows))

    s32 = _selected_subset(rows, 32, mode="driver_only")
    s48 = _selected_subset(rows, 48, mode="driver_plus_repro")
    Path(args.subset_32).write_text(json.dumps({
        "name": "selected_32",
        "policy": "drivers only, balanced by framework (15 currently exist; "
                  "expands as gen_driver.py adds more)",
        "n": len(s32), "bugs": s32,
    }, indent=2))
    Path(args.subset_48).write_text(json.dumps({
        "name": "selected_48",
        "policy": "drivers + reproduced-without-driver (gen_driver.py targets)",
        "n": len(s48), "bugs": s48,
    }, indent=2))

    print(f"manifest: {len(rows)} rows -> {args.out}")
    print(f"summary:  {args.summary}")
    print(f"selected_32: {len(s32)} -> {args.subset_32}")
    print(f"selected_48: {len(s48)} -> {args.subset_48}")


if __name__ == "__main__":
    main()
