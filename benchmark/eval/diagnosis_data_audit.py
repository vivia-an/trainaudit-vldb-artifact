#!/usr/bin/env python3
"""What the shipped diagnosis files do and do not contain.

app:diagnosis reports, on the 17 detected Real-SE cases:
  * 9/17 (53%) with a candidate set of size 1 at rule alert
  * a median C1 expansion to 6 (peak 12), then 5 of 8 converging at C1 and 2 at C2
  * the trajectory 1 -> 6 -> 1 drawn in fig:diagnosis-mechanism

Those are candidate-set cardinalities. This checks the shipped files for them.

    python3 benchmark/eval/diagnosis_data_audit.py
"""
import collections
import csv
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SIZE_FIELD = re.compile(r"set|size|cand|s_rule|c1|c2|count", re.I)


def main():
    inp = HERE / "diagnosis_s3_input.json"
    if not inp.exists():
        sys.exit(f"missing {inp.name}")
    d = json.loads(inp.read_text())
    cases = d["cases"]
    fields = sorted({k for c in cases for k in c})
    size_fields = [k for k in fields if SIZE_FIELD.search(k)]
    chains = collections.Counter(len(c.get("rca_chain") or []) for c in cases)

    print(f"diagnosis_s3_input.json: {len(cases)} cases")
    print(f"  fields                : {', '.join(fields)}")
    print(f"  candidate-set fields  : {size_fields or 'NONE'}")
    print(f"  rca_chain lengths     : {dict(chains)}")
    print(f"  built from            : {d.get('meta', {}).get('source', '?')}")

    print("\nrater files:")
    for tag in ("A", "B"):
        f = HERE / f"diagnosis_s3_rater_{tag}.csv"
        if not f.exists():
            print(f"  rater {tag}: missing")
            continue
        rows = list(csv.DictReader(f.open()))
        cols = [c for c in rows[0] if c != "bug_id"] if rows else []
        filled = {c: sum(1 for r in rows if (r[c] or "").strip()) for c in cols}
        state = "unfilled" if all(v == 0 for v in filled.values()) else "filled"
        print(f"  rater {tag}: {len(rows)} rows, {state}  {filled}")
        if state == "filled":
            l1 = collections.Counter((r.get("L1_correct") or "").strip() for r in rows)
            l2 = collections.Counter((r.get("L2_correct") or "").strip() for r in rows)
            print(f"           L1 {dict(l1)}   L2 {dict(l2)}")

    print("""
So the shipped files support a different statement than the appendix makes.

  * No candidate-set cardinality is recorded anywhere, and `rca_chain` is empty for all 17
    cases, so 53%, the median of 6, the peak of 12 and the 1->6->1 trajectory cannot be
    recomputed here.
  * The study is single-rater: rater A's rubric is entirely unfilled, so there is no
    inter-rater agreement for this study even though the design has two raters.
  * What rater B does record is a yes/no judgement on whether the rule name identifies the
    fault: L1 yes on 17 of 17, with L2 marked N/A throughout. That is a *stronger* claim than
    53% but a different one -- "the name identifies the fault" is not "the candidate set has
    size 1".
  * The input was assembled from paper_table_baseline_3way.md, which scores the superseded
    intermediate case set (see DETECTION_FILES_NOTE.md), so its 17 case ids may not be the
    current 17 either.

Either ship the candidate-set measurements behind fig:diagnosis-mechanism, or restate the
appendix around what rater B actually recorded.""")


if __name__ == "__main__":
    main()
