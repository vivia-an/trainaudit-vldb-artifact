"""Generate paper §4.1 three-way comparison table from the per-baseline CSVs.

Inputs:
  benchmark/eval/baseline_traincheck_results.csv   (has TrainAudit + TrainCheck)
  benchmark/eval/baseline_naive_results.csv        (has Naïve)

Output:
  benchmark/eval/paper_table_baseline_3way.md      (per doc 26 D5)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_ROOT = REPO_ROOT / "benchmark" / "eval"


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc-csv",
                    default=str(HARNESS_ROOT / "baseline_traincheck_results.csv"))
    ap.add_argument("--naive-csv",
                    default=str(HARNESS_ROOT / "baseline_naive_results.csv"))
    ap.add_argument("--out",
                    default=str(HARNESS_ROOT / "paper_table_baseline_3way.md"))
    args = ap.parse_args()

    tc_rows = _read(Path(args.tc_csv))
    naive_rows = _read(Path(args.naive_csv))

    # Index by (bug_id, phase)
    tc_by = {(r["bug_id"], r["phase"]): r for r in tc_rows}
    naive_by = {(r["bug_id"], r["phase"]): r for r in naive_rows}

    bug_ids = sorted({r["bug_id"] for r in tc_rows})

    out: List[str] = []
    out.append("# Paper §4.1 — Three-way baseline comparison (D1 same-集合)\n\n")
    out.append("## Per-bug verdict (buggy phase)\n\n")
    out.append("| bug_id | framework | TrainAudit | TrainCheck | Naïve |\n")
    out.append("|---|---|---|---|---|\n")
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fail_counts: Dict[str, int] = defaultdict(int)
    for bug in bug_ids:
        tc = tc_by.get((bug, "buggy"), {})
        nv = naive_by.get((bug, "buggy"), {})
        ta = tc.get("trainaudit_verdict", "?")
        tcv = tc.get("traincheck_verdict", "?")
        nvv = nv.get("verdict", "?")
        for col, v in (("TrainAudit", ta), ("TrainCheck", tcv), ("Naïve", nvv)):
            counts[col][v] += 1
            if v == "FAIL":
                fail_counts[col] += 1
        out.append(f"| {bug} | {tc.get('framework', '?')} | "
                    f"{ta} | {tcv} | {nvv} |\n")

    n_total = len(bug_ids)
    out.append(f"\n## Summary (buggy phase, n={n_total} D1 surrogates)\n\n")
    out.append("| 方法 | DETECTED | CLEAN (miss) | FAIL | det_rate |\n")
    out.append("|---|---:|---:|---:|---:|\n")
    for col in ("TrainAudit", "TrainCheck", "Naïve"):
        c = counts[col]
        d = c.get("DETECTED", 0)
        cl = c.get("CLEAN", 0)
        fl = c.get("FAIL", 0)
        out.append(f"| {col} | {d} | {cl} | {fl} | "
                    f"{d / n_total:.1%} |\n")

    # FP rate (fixed phase)
    out.append(f"\n## Fixed-phase FP rate (n={n_total} fixed runs)\n\n")
    out.append("For paper FP evidence, TrainCheck fixed-phase rows must come "
                "from held-out clean reruns checked with invariants learned "
                "from a separate clean reference run. Do not report a "
                "reference-trace re-check as an independent FP measurement.\n\n")
    out.append("| 方法 | fixed CLEAN | FP rate |\n")
    out.append("|---|---:|---:|\n")
    for col, key, src in (("TrainAudit", "trainaudit_verdict", tc_by),
                            ("TrainCheck", "traincheck_verdict", tc_by),
                            ("Naïve", "verdict", naive_by)):
        n_clean = sum(1 for bug in bug_ids
                      if src.get((bug, "fixed"), {}).get(key) == "CLEAN")
        out.append(f"| {col} | {n_clean}/{n_total} | "
                    f"{(n_total - n_clean) / n_total:.1%} |\n")

    out.append("\n## Source\n\n")
    out.append(f"- TrainCheck per-bug results: `{Path(args.tc_csv).name}`\n")
    out.append(f"- Naïve per-bug results: `{Path(args.naive_csv).name}`\n")
    out.append("- TrainCheck pipeline (per bug): "
                "benchmark/eval/traincheck_surrogates/run_one.sh\n")
    out.append("- Naïve detector: 4 signals (loss_spike, loss_nan, grad_nan, "
                "gradnorm_spike) over 25 training steps\n")

    Path(args.out).write_text("".join(out))
    print(f"-> {args.out}")
    for col in ("TrainAudit", "TrainCheck", "Naïve"):
        c = counts[col]
        print(f"  {col}: DETECTED={c.get('DETECTED', 0)}/{n_total}, "
              f"CLEAN={c.get('CLEAN', 0)}, FAIL={c.get('FAIL', 0)}")


if __name__ == "__main__":
    main()
