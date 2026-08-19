"""Offline π_topo ablation against B1/B2 multi-rank GPU traces.

Reads 8 duckdb files at benchmark/eval/_runs_gpu_ablation/B{1,2}_{BUGGY,FIXED}_rank{0,1}.duckdb
produced by `benchmark/eval/run_gpu_ablation_b1_b2.sh`. For each trace, runs
`full` and `no_topo` rule sets and prints the Δ per replica-cksum rule.

This is the supplementary E3-buggy run that v2 SPEC §0 marks optional.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "trainaudit"))
sys.path.insert(0, str(_REPO / "benchmark" / "eval"))

from run_ablation_s2 import replay_one  # reuses existing harness

TRACE_DIR = _REPO / "benchmark" / "eval" / "_runs_gpu_ablation"
EXPECT = [(bug, phase, rank)
          for bug in ("B1", "B2")
          for phase in ("BUGGY", "FIXED")
          for rank in (0, 1)]
REPLICA_RULES = ("T1-replica-cksum-equal",
                 "T1-grad-replica-cksum-equal",
                 "T1-buffer-replica-cksum-equal",
                 "T1-process-group-size-correct")
NO_TOPO_DIR = _REPO / "trainaudit" / "trainaudit" / "rules_no_topo"


def main():
    missing = [t for t in EXPECT
               if not (TRACE_DIR / f"{t[0]}_{t[1]}_rank{t[2]}.duckdb").exists()]
    if missing:
        print(f"[ERROR] {len(missing)} expected trace(s) missing:")
        for bug, phase, rank in missing:
            print(f"   {TRACE_DIR / f'{bug}_{phase}_rank{rank}.duckdb'}")
        print(f"\nRun: bash {_REPO}/benchmark/eval/run_gpu_ablation_b1_b2.sh "
              f"on a 2-GPU pod first.")
        return 1

    if not NO_TOPO_DIR.exists():
        print(f"[ERROR] {NO_TOPO_DIR} missing — run "
              f"`python scripts/build_ablation_rules.py --mode no_topo` first.")
        return 1

    print(f"[trace inventory] {len(EXPECT)} traces at {TRACE_DIR}")
    totals = defaultdict(lambda: {"full": 0, "no_topo": 0})
    per_trace = []
    for bug, phase, rank in EXPECT:
        p = TRACE_DIR / f"{bug}_{phase}_rank{rank}.duckdb"
        row = {"trace": f"{bug}_{phase}_rank{rank}", "rules": {}}
        for setting, rdir in (("full", None), ("no_topo", NO_TOPO_DIR)):
            for rid, viol, n_v, _dt in replay_one(p, setting, rdir):
                if rid not in REPLICA_RULES:
                    continue
                row["rules"].setdefault(rid, {})[setting] = n_v
                totals[rid][setting] += n_v
        per_trace.append(row)

    print("\n=== Per-trace π_topo Δ (replica-cksum rules only) ===")
    hdr = f"{'trace':<24} | " + " | ".join(
        f"{rid.replace('T1-', ''):>30}" for rid in REPLICA_RULES)
    print(hdr)
    print("-" * len(hdr))
    for row in per_trace:
        cells = []
        for rid in REPLICA_RULES:
            d = row["rules"].get(rid, {})
            full = d.get("full", 0)
            no_topo = d.get("no_topo", 0)
            delta = no_topo - full
            cells.append(f"{full}→{no_topo} (Δ={delta:+d})".rjust(30))
        print(f"{row['trace']:<24} | " + " | ".join(cells))

    print("\n=== Aggregate ===")
    print(f"{'rule_id':<40} | {'full':>10} | {'no_topo':>10} | {'Δ':>10}")
    print("-" * 80)
    for rid in REPLICA_RULES:
        f, nt = totals[rid]["full"], totals[rid]["no_topo"]
        print(f"{rid:<40} | {f:>10} | {nt:>10} | {nt - f:>+10}")

    # Markdown snippet to paste into paper_table_ablation_s2.md
    snippet = _REPO / "benchmark" / "eval" / "paper_table_ablation_s2_b1b2.md"
    with open(snippet, "w") as f:
        f.write("## E3 supplementary — B1/B2 buggy multi-rank (TP=2)\n\n")
        f.write("8 traces (2 bugs × 2 phases × 2 ranks). Driver: "
                "`benchmark/eval/run_gpu_ablation_b1_b2.sh`.\n\n")
        f.write("| trace | rule | full | no_topo | Δ |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in per_trace:
            for rid in REPLICA_RULES:
                d = row["rules"].get(rid, {})
                full = d.get("full", 0)
                no_topo = d.get("no_topo", 0)
                if full == 0 and no_topo == 0:
                    continue
                f.write(f"| {row['trace']} | {rid} | {full} | {no_topo} "
                        f"| {no_topo - full:+d} |\n")
        f.write("\n### Aggregate\n\n")
        f.write("| rule_id | full | no_topo | Δ |\n")
        f.write("|---|---:|---:|---:|\n")
        for rid in REPLICA_RULES:
            tf, tnt = totals[rid]["full"], totals[rid]["no_topo"]
            f.write(f"| {rid} | {tf} | {tnt} | {tnt - tf:+d} |\n")
    print(f"\n[output] markdown snippet → {snippet.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
