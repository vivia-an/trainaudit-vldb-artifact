#!/usr/bin/env python3
"""Check appendix tab:detection-results row by row against the authoritative CSV.

`main.tex` sends the reader here for per-case outcomes ("per-case outcomes in
Appendix~\\ref{app:extended_data}"), so this 18-row table is where the 17/18 headline is
actually itemised. Nothing verified it.

Verifying it needs a join the table does not make easy: its rows are keyed by **fixed
commit** (`Megatron-3c637fc0d`, `DS-d56268f3`) while `real_se_detection.csv` is keyed by
case id (`B1`, `B3`). `real_sdc_manifest.json` carries `fixed_commit` per case, so the two
can be matched through it -- and the mapping is checked here rather than assumed.

Seventeen rows resolve that way. The eighteenth, `OLMo-d7994c86`, is deliberately absent
from the manifest: it is LC1/O-003, the observability-boundary case that makes the rate
94.4% rather than 100%. TrainAudit records the expected miss; the two baseline cells are
unavailable and print em dashes.

Checked per row: the TrainAudit verdict, the TrainCheck verdict, and TrainCheck's `n/m`
violation fraction where the table prints one.

  python3 benchmark/eval/verify_appendix_detection_table.py [--check]
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
APX = ROOT / "paper" / "appendix.tex"
SDC = ROOT / "benchmark" / "eval" / "real_sdc"
BOUNDARY_COMMIT = "d7994c86"
# TrainCheck failed to infer on one case; the table can only render a cross for it.
NOT_DETECTED = {"MISSED", "TOOL_FAILURE", "NOT_RUN"}

ROW = re.compile(
    r"^\s*([A-Za-z]+)-([0-9a-f]{7,10})\s*&\s*([^&]+?)\s*&\s*(\S+)\s*&\s*([^&]+?)\s*&"
    r"\s*(\$\\checkmark\$|\$\\times\$|---)\s*&\s*(\$\\checkmark\$|\$\\times\$|---)"
    r"(?:\\,\((\d+)/(\d+)\))?\s*&", re.M)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    fail = []

    body = APX.read_text()
    block = body[body.index("label{tab:detection-results}"):]
    block = block[:block.index("\\end{table}")]
    rows = ROW.findall(block)

    man = json.loads((SDC / "real_sdc_manifest.json").read_text())
    cases = man if isinstance(man, list) else (
        man.get("cases") or next(v for v in man.values() if isinstance(v, list)))
    # The table abbreviates commits to 8 or 9 characters and the manifest holds full
    # hashes, so match on whichever is the shorter prefix rather than a fixed width.
    manifest_commits = [((c.get("fixed_commit") or ""), c["case_id"]) for c in cases
                        if c.get("fixed_commit")]

    def resolve(abbrev):
        hits = {cid for full, cid in manifest_commits if full.startswith(abbrev)}
        return hits.pop() if len(hits) == 1 else None
    det = {r["case_id"]: r for r in csv.DictReader(
        (SDC / "real_se_detection.csv").open())}

    print(f"appendix table rows parsed: {len(rows)}")
    if len(rows) != 18:
        fail.append(f"expected 18 rows, parsed {len(rows)}")

    resolved = 0
    for fw, commit, _fwname, _tier, constraint, ta, tc, n, m in rows:
        tag = f"{fw}-{commit}"
        if commit == BOUNDARY_COMMIT:
            ok = ta == "$\\times$" and tc == "---"
            print(f"  {tag:20} boundary case (LC1/O-003), expected TA miss / baselines unavailable "
                  f"-> {'ok' if ok else 'MISMATCH'}")
            if not ok:
                fail.append(f"{tag}: boundary row should be TA miss / baseline unavailable")
            continue
        cid = resolve(commit)
        if cid is None:
            print(f"  {tag:20} NOT IN MANIFEST")
            fail.append(f"{tag}: no manifest entry")
            continue
        row = det.get(cid)
        if row is None:
            print(f"  {tag:20} -> {cid}: not in real_se_detection.csv")
            fail.append(f"{cid}: absent from the detection CSV")
            continue
        resolved += 1
        bad, notes = [], []
        for tool, cell in (("trainaudit", ta), ("traincheck", tc)):
            got = row[tool]
            shown_detected = cell == "$\\checkmark$"
            if shown_detected != (got == "DETECTED"):
                bad.append(f"{tool} {got} vs table "
                           f"{'checkmark' if shown_detected else 'cross'}")
            elif got in NOT_DETECTED and got != "MISSED":
                notes.append(f"{tool}={got} rendered as a cross")
        # The CSV appends the surrogate id -- "0/199, via CF1" -- so compare the fraction.
        detail = (row.get("traincheck_detail") or "").split(",")[0].strip()
        if n and detail and detail != f"{n}/{m}":
            bad.append(f"detail {detail} vs table {n}/{m}")
        status = "ok" if not bad else "MISMATCH: " + "; ".join(bad)
        if notes and not bad:
            status = "ok  (" + "; ".join(notes) + ")"
        print(f"  {tag:20} -> {cid:10} {constraint[:34]:34} {status}")
        fail.extend(f"{cid}: {b}" for b in bad)

    print(f"\n{resolved} row(s) resolved through the manifest and compared; "
          f"1 boundary row checked separately")

    # Every evaluated case should have a corresponding per-bug artifact directory.
    bugs = ROOT / "benchmark" / "bugs"
    bare = []
    kinds = {}
    for c in cases:
        d = bugs / c["case_id"]
        names = sorted(f.name for f in d.iterdir()) if d.is_dir() else []
        if not names:
            bare.append(c["case_id"])
        kinds[c["case_id"]] = names
    have = len(cases) - len(bare)
    print(f"  {'ok  ' if not bare else 'FAIL'}  per-bug artifacts present for "
          f"{have}/{len(cases)} Real-SE cases" + (f"  MISSING {bare}" if bare else ""))
    if bare:
        fail.append(f"no per-bug artifacts for {bare}")
    n_detect = sum(1 for v in kinds.values() if "detect.py" in v)
    print(f"        {n_detect} carry detect.py; {len(cases) - n_detect} are "
          f"trainaudit-driver only, matching the runnable-tier split")
    if a.check:
        if fail:
            print(f"\nFAIL: {fail}", file=sys.stderr)
            return 1
        print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
