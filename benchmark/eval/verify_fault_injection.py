#!/usr/bin/env python3
"""A fully reproducible injection experiment that the paper does not use.

`paper_table_fault_injection.md` is headed "Paper §4.1" and reports 34 injected faults,
31/31 detection on the severe and moderate ones, and 3/3 true negatives on subthreshold
boundary cases. Every figure reproduces exactly from `fault_injection.csv`:

    34 rows                     31 DETECTED + 3 TRUE_NEGATIVE
    severity                    25 severe, 6 moderate, 3 subthreshold
    severe+moderate detection   31/31
    boundary                    3/3 TRUE_NEGATIVE -- no firing at the sensitivity floor
    by tier                     T0 22/22, T1 9/9

**The paper does not report any of it.** Neither `main.tex` nor `appendix.tex` mentions
injection in this sense, and §4.1 is the architecture section, so that heading is stale from
an earlier draft. The experiment is complete, its data ships, and its result is positive.

That is worth flagging as unused evidence rather than as a defect. A 100% detection rate on
31 injected faults *paired with* zero firing on the three subthreshold cases speaks to
sensitivity and precision together, which is the pairing the paper argues for throughout --
and it is a different experimental modality from the commit-pair replays, so it does not
inherit their selection bias toward bugs with localizable witnesses (§3's own caveat).

This check asserts the table against its data. It takes no view on whether the paper should
include it.

  python3 benchmark/eval/verify_fault_injection.py [--check]
"""
from __future__ import annotations
import argparse, collections, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "fault_injection.csv"
SEVERE = {"severe", "moderate"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    fail = []
    rows = list(csv.DictReader(CSV.open()))

    def want(label, got, expect):
        ok = got == expect
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:44} {got}  stated {expect}")
        if not ok:
            fail.append(f"{label}: {got} != {expect}")

    sev = [r for r in rows if r["severity"] in SEVERE]
    det = [r for r in sev if r["verdict"] == "DETECTED"]
    bd = [r for r in rows if r["severity"] not in SEVERE]

    want("injected faults", len(rows), 34)
    want("severe + moderate", len(sev), 31)
    want("of those, detected", len(det), 31)
    want("boundary cases", len(bd), 3)
    want("boundary cases that stayed silent",
         sum(1 for r in bd if r["verdict"] == "TRUE_NEGATIVE"), 3)
    for tier, expect in (("T0", 22), ("T1", 9)):
        s = [r for r in sev if r["tier"] == tier]
        want(f"{tier} detected", sum(1 for r in s if r["verdict"] == "DETECTED"), expect)

    print(f"\n  severity mix: {dict(collections.Counter(r['severity'] for r in rows))}")
    print(f"  every fault names the rule it expects; {sum(1 for r in rows if r['expected_rule'])}"
          f"/{len(rows)} populated")
    print("\n  none of this appears in main.tex or appendix.tex -- unused evidence, not a defect")

    if a.check:
        if fail:
            print(f"\nFAIL: {fail}", file=sys.stderr)
            return 1
        print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
