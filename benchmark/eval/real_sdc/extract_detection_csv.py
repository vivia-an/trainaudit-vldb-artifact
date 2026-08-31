#!/usr/bin/env python3
"""Validate the released per-case Real-SE detection record.

The 17 confirmed replay cases are defined by `real_sdc_manifest.json`; the
observability-boundary case is tracked separately by the manifest and appendix.
This command is read-only and never rewrites the released CSV.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "real_sdc_manifest.json"
RECORD = HERE / "real_se_detection.csv"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    expected = {row["case_id"] for row in manifest["cases_confirmed_real"]}
    rows = list(csv.DictReader(RECORD.open()))
    observed = {row["case_id"] for row in rows}

    failures = []
    if observed != expected:
        failures.append(
            f"case set mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    if len(rows) != len(observed):
        failures.append("duplicate case_id in real_se_detection.csv")

    trainaudit = sum(row["trainaudit"] == "DETECTED" for row in rows)
    traincheck = sum(row["traincheck"] == "DETECTED" for row in rows)
    naive = sum(row["naive"] == "DETECTED" for row in rows)
    tool_failures = sum(row["traincheck"] == "TOOL_FAILURE" for row in rows)

    expected_totals = {
        "TrainAudit": (trainaudit, 17),
        "TrainCheck": (traincheck, 5),
        "Naive": (naive, 0),
        "TrainCheck tool failures": (tool_failures, 1),
    }
    for label, (got, want) in expected_totals.items():
        if got != want:
            failures.append(f"{label}: {got} != {want}")

    print(f"confirmed Real-SE records: {len(rows)}")
    print(f"  TrainAudit detected {trainaudit}/{len(rows)}")
    print(f"  TrainCheck detected {traincheck}/{len(rows)}")
    print(f"  Naive detected {naive}/{len(rows)}")
    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
