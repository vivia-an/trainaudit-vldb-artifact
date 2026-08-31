#!/usr/bin/env python3
"""Turn the per-case verdict table in SMOKE_REPORT.md into a machine-readable CSV.

SMOKE_REPORT.md is the record of the *current* Real-SE case set — the one the paper's
appendix table `tab:detection-results` and the `numbers.tex` macros are computed from.
It is prose, so this script lifts its table into real_se_detection.csv without retyping
anything: the CSV is derived, the report stays the source of truth.

Two older detection files ship alongside and score *different* case sets; see
../DETECTION_FILES_NOTE.md before using them.
"""
import csv
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPORT = HERE / "SMOKE_REPORT.md"
OUT = HERE / "real_se_detection.csv"

VERDICT = {"✓": "DETECTED", "✗": "MISSED", "F": "TOOL_FAILURE"}


def parse_cell(cell):
    """'✓ (1/316)' -> ('DETECTED', '1/316'); '✗ (0/199, via CF1)' -> ('MISSED', '0/199, via CF1')."""
    cell = cell.strip()
    if not cell:
        return "", ""
    verdict = VERDICT.get(cell[0], cell[0])
    m = re.search(r"\(([^)]*)\)", cell)
    return verdict, m.group(1) if m else ""


def main():
    if not REPORT.exists():
        sys.exit(f"missing {REPORT}")
    block = re.search(r"## Per-case verdict.*?\n\n(.*?)\n\n", REPORT.read_text(), re.S)
    if not block:
        sys.exit("could not locate the '## Per-case verdict' table")

    out = []
    for line in block.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] == "Case" or set(cells[0]) <= {"-"}:
            continue
        ta, ta_d = parse_cell(cells[1])
        tc, tc_d = parse_cell(cells[2])
        nv, nv_d = parse_cell(cells[3])
        out.append({
            "case_id": cells[0],
            "trainaudit": ta,
            "traincheck": tc, "traincheck_detail": tc_d,
            "naive": nv,
            "pattern": cells[4], "tier": cells[5],
        })

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    n = len(out)
    ta = sum(r["trainaudit"] == "DETECTED" for r in out)
    tc = sum(r["traincheck"] == "DETECTED" for r in out)
    tcf = sum(r["traincheck"] == "TOOL_FAILURE" for r in out)
    nv = sum(r["naive"] == "DETECTED" for r in out)
    print(f"wrote {OUT.name}: {n} cases")
    print(f"  TrainAudit  {ta}/{n}")
    print(f"  TrainCheck  {tc}/{n}  ({tcf} tool failure)")
    print(f"  Naive       {nv}/{n}")


if __name__ == "__main__":
    main()
