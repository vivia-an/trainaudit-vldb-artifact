#!/usr/bin/env python3
"""Audit the corrected matched Catalog/free-form ablation printed in the paper."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def rows(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    aggregate = {row["arm"]: row for row in rows("catalog_direct_ablation.csv")}
    pairing = {
        row["outcome"]: int(row["count"])
        for row in rows("catalog_endpoint_pairing.csv")
    }
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}" +
              (f"  [{detail}]" if detail else ""))
        if not condition:
            failures.append(label)

    expected = {
        "Catalog": {
            "l1": 420, "l2": 5334, "l3": 3436, "l4": 357,
            "deploy_entries": 45, "realse_detected": 17,
            "fixed_false_positives": 0,
        },
        "Free-form": {
            "l1": 780, "l2": 10500, "l3": 3600, "l4": 210,
            "deploy_entries": 29, "realse_detected": 11,
            "fixed_false_positives": 2,
        },
    }
    for arm, values in expected.items():
        for field, value in values.items():
            check(f"{arm} {field} is {value}", int(aggregate[arm][field]) == value)
        check(f"{arm} Real-SE denominator is 18", int(aggregate[arm]["realse_total"]) == 18)
        check(f"{arm} fixed-side denominator is 17", int(aggregate[arm]["fixed_total"]) == 17)

    catalog_yield = int(aggregate["Catalog"]["l4"]) / int(aggregate["Catalog"]["l2"])
    free_yield = int(aggregate["Free-form"]["l4"]) / int(aggregate["Free-form"]["l2"])
    ratio = catalog_yield / free_yield
    check("Catalog yield rounds to 6.69%", round(100 * catalog_yield, 2) == 6.69)
    check("free-form yield is 2.00%", round(100 * free_yield, 2) == 2.00)
    check("yield ratio rounds to 3.35x", round(ratio, 2) == 3.35)

    expected_pairing = {
        "both_detected": 11,
        "catalog_only": 6,
        "free_form_only": 0,
        "neither_detected": 1,
    }
    check("paired outcome counts match", pairing == expected_pairing, str(pairing))
    check("paired outcomes total 18", sum(pairing.values()) == 18)
    check("pairing reproduces Catalog 17/18",
          pairing["both_detected"] + pairing["catalog_only"] == 17)
    check("pairing reproduces free-form 11/18",
          pairing["both_detected"] + pairing["free_form_only"] == 11)

    discordant = pairing["catalog_only"] + pairing["free_form_only"]
    smaller = min(pairing["catalog_only"], pairing["free_form_only"])
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
    mcnemar_p = min(1.0, 2 * tail)
    check("two-sided exact McNemar p is 0.03125", math.isclose(mcnemar_p, 0.03125))

    paper = (ROOT / "paper" / "main.tex").read_text() + "\n" + \
            (ROOT / "paper" / "appendix.tex").read_text()
    for literal in ("6.69\\%", "2.00\\%", "3.35$\\times$", "17/18", "11/18",
                    "0/17", "2/17", "p{=}0.031"):
        check(f"paper prints {literal}", literal in paper)
    for stale in ("1.52$\\times$", "6.70$\\pm$0.22\\%", "4.40$\\pm$0.22\\%"):
        check(f"paper omits stale {stale}", stale not in paper)

    if failures:
        print(f"\nFAIL: {len(failures)} assertion(s)", file=sys.stderr)
        return 1
    print(f"\nCorrected Catalog ablation verified; exact McNemar p={mcnemar_p:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
