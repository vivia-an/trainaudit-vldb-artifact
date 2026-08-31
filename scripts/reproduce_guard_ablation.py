#!/usr/bin/env python3
"""Regenerate d1_results.csv from the shipped per-cell reports, offline.  (R5)

§5.3's leave-one-database-out guard ablation reports aggregate false positives of
342 (full), 429 (without pi_precond) and 598 (without pi_topo).  The
*measurement* step cannot be re-run offline: sdccheck generates each
check's SQL with an LLM, so regenerating the raw reports needs network access and
API credentials.

Everything downstream of that IS reproducible, and this proves it for BOTH arms.
126 per-cell report JSONs ship in experiments/guard_ablation/ (42 databases x 3
libraries) and 42 more in experiments/guard_ablation_no_adversarial/, and the
shipped aggregator core/ablation_scripts/aggregate_fp.py rebuilds each committed
CSV from them.  This runs that aggregation into a temporary file and compares row
by row, then checks all four published totals -- 342 full, 429 without
pi_precond, 551 without adversarial verification, 598 without pi_topo -- and the
guard ranking those imply.

So the reproduction boundary is precise: the LLM-driven SQL generation is not
re-runnable from the artifact, but the published numbers follow from the shipped
raw reports by a shipped script, and that derivation is checked on every run.

  python3 scripts/reproduce_guard_ablation.py [--check]
"""
from __future__ import annotations
import argparse, csv, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "core" / "ablation_scripts" / "aggregate_fp.py"
LIBS = ROOT / "core" / "config" / "ablation_libraries"
EXP = ROOT / "experiments"

# Both arms of §5.3, each: reports dir, committed CSV, library json, expected report count.
ARMS = [
    (EXP / "guard_ablation", "d1_results.csv", LIBS / "lib_full.json", 126),
    (EXP / "guard_ablation_no_adversarial", "no_adversarial_results.csv",
     LIBS / "lib_no_adversarial.json", 42),
]
# "removing pi_precond, adversarial verification, and pi_topo raises aggregate false
# positives from 342 to 429, 551, and 598, respectively" -- main.tex §5.3.
PUBLISHED = {"lib_full": 342, "lib_no_precond": 429,
             "lib_no_adversarial": 551, "lib_no_topo": 598}


def rows(p: Path):
    return sorted(tuple(sorted(r.items())) for r in csv.DictReader(p.open()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    fails = []

    def want(label, cond):
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
        if not cond:
            fails.append(label)

    tot: dict[str, int] = {}
    for reports, csv_name, lib, n_want in ARMS:
        shipped_csv = reports / csv_name
        found = sorted(reports.glob("lib_*__*.json"))
        rel = reports.relative_to(ROOT)
        print(f"\n{rel}: {len(found)} per-cell report JSON(s)")
        if not found or not shipped_csv.exists():
            print(f"  SKIP: reports or {csv_name} absent")
            continue
        stray = [p.name for p in reports.glob("*.json") if p not in found]
        if stray:
            print(f"  note: {len(stray)} file(s) not matching lib_*__*.json: "
                  f"{stray[:3]} -- scanned by the aggregator but contributing no row")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "regen.csv"
            r = subprocess.run([sys.executable, str(AGG), "--reports-dir", str(reports),
                                "--library-json", str(lib), "--csv-out", str(out)],
                               capture_output=True, text=True, cwd=ROOT)
            if r.returncode != 0:
                print(r.stdout[-1500:], r.stderr[-1500:])
                fails.append(f"{rel}: aggregator failed")
                continue
            regen, shipped = rows(out), rows(shipped_csv)
        print(f"  regenerated {len(regen)} row(s); {csv_name} has {len(shipped)}")
        want(f"{csv_name} regenerates exactly from its reports", regen == shipped)
        if regen != shipped:
            for x in [q for q in regen if q not in shipped][:2]:
                print(f"    only in regenerated: {dict(x)}")
            for x in [q for q in shipped if q not in regen][:2]:
                print(f"    only in shipped:     {dict(x)}")
        want(f"{rel} still ships {n_want} per-cell report(s)", len(found) == n_want)
        for q in csv.DictReader(shipped_csv.open()):
            tot[q["lib"]] = tot.get(q["lib"], 0) + int(q["fail_FP"])

    print("\naggregate false positives per arm:")
    for lib, want_v in PUBLISHED.items():
        got = tot.get(lib)
        print(f"  {'ok  ' if got == want_v else 'FAIL'}  {lib:18} = {got} (paper: {want_v})")
        if got != want_v:
            fails.append(f"{lib} total")
    want("the published guard ranking holds: topo > adversarial > precond > full",
         tot.get("lib_no_topo", 0) > tot.get("lib_no_adversarial", 0)
         > tot.get("lib_no_precond", 0) > tot.get("lib_full", 0))

    if not a.check:
        return 0
    if fails:
        print(f"\n{len(fails)} assertion(s) failed")
        return 1
    print("\nall four of §5.3's arms follow from the shipped reports by a shipped script")
    return 0


if __name__ == "__main__":
    sys.exit(main())
