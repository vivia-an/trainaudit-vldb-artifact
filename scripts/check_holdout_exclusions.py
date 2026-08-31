#!/usr/bin/env python3
"""The re-mining holdout's contamination control covers every evaluated case.

`experiments/holdout_mining/EXCLUSION_LIST.json` is a **pre-registered** withholding list --
timestamped 2026-07-12, sourced from the appendix detection table -- naming 19 Real-SE cases
whose fix commits, issues and bug-fix history entries must be kept out of every mining run.
Its two rules are worth quoting because they draw the line in the right place:

  1. any mining prompt, catalog entry or bug-fix history record citing one of these fix
     commits or case ids must be withheld;
  2. constraint *names* are not withheld -- the templates predate the cases -- only the
     case-specific evidence.

Nothing in the artifact asserted that this list still covers what is actually evaluated, and
a holdout that has drifted out from under its evaluation set is contaminated without anyone
noticing.

It has not drifted, and it errs the safe way. Every one of the 18 rows in the current
`tab:detection-results` is on the list; the list additionally withholds `OLMoC-O-022`, a case
since dropped from the table. Over-withholding is conservative, under-withholding would be
contamination, so a strict superset is the direction to be in.

This check fails if a case ever appears in the table without being on the list.

  python3 scripts/check_holdout_exclusions.py
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "experiments" / "holdout_mining" / "EXCLUSION_LIST.json"
APX = ROOT / "paper" / "appendix.tex"


def main() -> int:
    if not LIST.exists():
        print(f"FAIL: {LIST.relative_to(ROOT)} is missing", file=sys.stderr)
        return 1
    ex = json.loads(LIST.read_text())
    ids = {c["case_id"] for c in ex["cases"]}

    tex = APX.read_text()
    i = tex.index("label{tab:detection-results}")
    rows = re.findall(r"^\s*([A-Za-z]+)-([0-9a-f]{7,10})\s*&",
                      tex[i:tex.index(r"\end{table}", i)], re.M)
    cur = {f"{a}-{b}" for a, b in rows}

    print(f"pre-registered {ex['registered_at'][:10]}: {len(ids)} withheld case(s)")
    print(f"current tab:detection-results: {len(cur)} row(s)")
    uncovered = sorted(cur - ids)
    extra = sorted(ids - cur)
    print(f"  {'ok  ' if not uncovered else 'FAIL'}  every evaluated case is withheld"
          f"  {uncovered or 'yes'}")
    print(f"  note  withheld but no longer evaluated: {extra or 'none'} "
          f"(over-withholding is the safe direction)")
    print(f"  note  rule 2 keeps constraint names in the template space; only "
          f"case-specific evidence is withheld")

    if uncovered:
        print(f"\nFAIL: evaluated cases missing from the holdout list: {uncovered}",
              file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
