#!/usr/bin/env python3
"""Recompute fig:tier-coverage's coverage axis from per-record data.

`extension_v3/trace_tier_392.json` carries `min_tier_required` for all **392** bugs, and the
cumulative shares reproduce every point the figure renders:

    tier 0    111/392 = 28.3%              tier 3    244/392 = 62.2%
    tier 1    192/392 = 49.0%              tier 4    261/392 = 66.6%
    tier 2    223/392 = 56.9%              tier 5-6  305/392 = 77.8%

and the leftover 87 unobservable records give 22.2%.

Why the earlier sweep missed it, asserted here so it is not repeated: there are **two**
tier files over the same 392 records and they measure different things.

    v2_full/tier_392_v2.csv        cum 94 -> 261, i.e. 24.0% -> 66.6%   NOT the figure
    extension_v3/trace_tier_*.csv cum 111 -> 305, i.e. 28.3% -> 77.8%   the figure

Both files are valid measurements; the second is the source used for this figure's
coverage axis.

  python3 benchmark/eval/verify_tier_coverage_axis.py [--check]
"""
from __future__ import annotations
import argparse, collections, csv, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PER_BUG = ROOT / "benchmark" / "eval" / "extension_v3" / "trace_tier_392.json"
SUMMARY = ROOT / "benchmark" / "eval" / "extension_v3" / "trace_tier_summary.csv"
OTHER = ROOT / "benchmark" / "eval" / "v2_full" / "tier_392_v2.csv"
FIG = ROOT / "figures" / "tier_coverage.pdf"
RENDERED = [28, 49, 57, 62, 67, 78]
DISPLAYED = [28.3, 49.0, 56.9, 62.2, 66.6, 77.8]
TIERS = [0, 1, 2, 3, 4, "5-6"]

fails: list[str] = []


def want(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        fails.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.parse_args()
    if not PER_BUG.exists():
        print(f"SKIP: {PER_BUG.name} absent")
        return 0

    recs = json.loads(PER_BUG.read_text())["per_bug"]
    recs = list(recs.values()) if isinstance(recs, dict) else recs
    hist = collections.Counter(r["min_tier_required"] for r in recs)
    n = len(recs)
    cum, series = 0, []
    print(f"{n} per-bug records in {PER_BUG.name}\n")
    for t in TIERS:
        cum += hist.get(t, 0)
        series.append((t, cum, 100 * cum / n))
        print(f"  through tier {str(t):4} cum={cum:4}  {100 * cum / n:5.2f}%")
    unobs = n - cum
    print(f"  unobservable      {unobs:4}  {100 * unobs / n:5.2f}%")

    print("\nagainst the figure")
    want("the study covers all 392 corpus records", n == 392, str(n))
    for (t, c, pct), rend in zip(series, RENDERED):
        want(f"S{t if t != '5-6' else '5'} coverage rounds to the rendered {rend}",
             round(pct) == rend, f"{c}/{n} = {pct:.2f}%")
    want("the unobservable remainder is 22% of the corpus",
         round(100 * unobs / n) == 22, f"{unobs}/{n}")

    if SUMMARY.exists():
        rows = list(csv.DictReader(SUMMARY.open()))
        got = [float(r["cum_pct_392"]) for r in rows if r["cum_through_tier"].startswith("Tier")]
        want("the shipped summary CSV agrees with the recomputation",
             [round(x) for x in got] == RENDERED, f"{got}")

    print("\nindependent tier series")
    if OTHER.exists():
        other = [float(r["pct_392"]) for r in csv.DictReader(OTHER.open())
                 if r["cum_tier"].startswith("Tier")]
        want("v2_full/tier_392_v2.csv is a different series, not this figure's",
             [round(x) for x in other] != RENDERED, f"{[round(x) for x in other]}")
        want("and it is still shipped (both measurements exist)", bool(other), str(len(other)))

    if FIG.exists():
        try:
            txt = subprocess.run(["pdftotext", "-layout", str(FIG), "-"],
                                 capture_output=True, text=True, check=True).stdout
            found = [v for v in DISPLAYED if f"{v:.1f}" in txt]
            print("\nagainst the rendered PDF")
            want("all six one-decimal coverage values appear in the figure text",
                 len(found) == len(DISPLAYED), f"{found}")
        except (OSError, subprocess.CalledProcessError):
            print("\n  (pdftotext unavailable; skipped the rendered-PDF cross-check)")

    if fails:
        print(f"\n{len(fails)} assertion(s) failed")
        return 1
    print("\nthe coverage axis reproduces from the per-record tier assignments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
