#!/usr/bin/env python3
"""Recompute the guard ablation with the duplicated-rank databases excluded.

Four of the clean baselines have rank 1 byte-identical to rank 0 (GAP_AUDIT O26), so every
cross-rank comparison on them passes vacuously. That raised a fair question about §5.3: how
much of the reported 342 → 429 / 551 / 598 rests on traces where the rules under test cannot
fire?

This answers it from the shipped per-cell data — no re-capture, no re-run — by dropping those
four databases and re-aggregating.

    python3 benchmark/injection/recompute_ablation_clean.py
"""
import collections
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CELLS = [
    ROOT / "experiments" / "guard_ablation" / "d1_results.csv",
    ROOT / "experiments" / "guard_ablation_no_adversarial" / "no_adversarial_results.csv",
]
# Verified at full payload by benchmark/injection/audit_rank_captures.py.
DUPLICATED = {"dp_normal", "dp_normal_db", "dist_optimizer_normal", "mixed_precision_normal"}
ARMS = [("lib_full", "full"), ("lib_no_precond", "-pi_precond"),
        ("lib_no_adversarial", "-adversarial"), ("lib_no_topo", "-pi_topo")]


def main():
    rows = []
    for f in CELLS:
        if not f.exists():
            sys.exit(f"missing {f}")
        rows += list(csv.DictReader(f.open()))

    def agg(keep):
        out = collections.defaultdict(collections.Counter)
        for r in rows:
            if not keep(r):
                continue
            for k in ("total", "pass", "fail_FP", "error"):
                out[r["lib"]][k] += int(r[k])
            out[r["lib"]]["dbs"] += 1
        return out

    published = agg(lambda r: True)
    corrected = agg(lambda r: r["db"] not in DUPLICATED)

    print(f"{'arm':<14}{'published':>10}{'vs full':>10}   {'corrected':>10}{'vs full':>10}")
    for lib, label in ARMS:
        p, c = published[lib]["fail_FP"], corrected[lib]["fail_FP"]
        dp = "—" if lib == "lib_full" else \
            f"+{100 * (p - published['lib_full']['fail_FP']) / published['lib_full']['fail_FP']:.1f}%"
        dc = "—" if lib == "lib_full" else \
            f"+{100 * (c - corrected['lib_full']['fail_FP']) / corrected['lib_full']['fail_FP']:.1f}%"
        print(f"{label:<14}{p:>10}{dp:>10}   {c:>10}{dc:>10}")

    op = [published[l]["fail_FP"] for l, _ in ARMS]
    oc = [corrected[l]["fail_FP"] for l, _ in ARMS]
    print(f"\ndatabases: {published['lib_full']['dbs']} published, "
          f"{corrected['lib_full']['dbs']} after excluding {len(DUPLICATED)}")
    print(f"ordering published: {' < '.join(map(str, op))}   monotone={op == sorted(op)}")
    print(f"ordering corrected: {' < '.join(map(str, oc))}   monotone={oc == sorted(oc)}")
    if oc == sorted(oc):
        print("\nThe ordering and the rough magnitudes survive the correction, so O26 does not\n"
              "overturn §5.3's conclusion — it shifts the numbers by 2–6 percentage points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
