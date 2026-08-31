#!/usr/bin/env python3
"""Reconcile the 392-record corpus with the pools it was built from.

Appendix `tab:dataset-construction-flow` states the rule in prose:

    295-record pool + 96 records unique to the 128-record pool + 1 orphan

Everything downstream rests on that 392 -- `tab:taxonomy`, the pattern distribution, the
held-out split -- and nothing checked the arithmetic or the membership. `pool_overlap.json`
carries both the counts and the actual id lists, so it can be reconciled exactly rather
than just added up.

It reconciles, and the orphan is identifiable by name:

    overlap(128, 295)         32
    only in the 128-pool      96
    only in the 295-pool     263
                            ----
    pool records             391   pairwise disjoint
    orphan                   + 1   M-NEW-MUON-MTP
                            ----
    corpus                   392   = the record count in silent_evidence_392.json

with 32 + 96 = 128 and 32 + 263 = 295 both holding. The one id present in the evidence
file and in no pool is exactly the orphan `pool_overlap.json` records as *included*
("full config + detect + reproduce"); a second orphan, `M-NEW-2`, is recorded as excluded
for having no `config.json`, and correctly does not appear.

  python3 benchmark/eval/verify_corpus_construction.py [--check]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOTAL = 392


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    fail = []

    def want(label, got, expect):
        ok = got == expect
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:46} {got}  expected {expect}")
        if not ok:
            fail.append(f"{label}: {got} != {expect}")

    d = json.loads((HERE / "pool_overlap.json").read_text())
    m = d["meta"]
    ov, o128, o295 = (set(d["overlap_ids"]), set(d["only128_ids"]),
                      set(d["only295_ids"]))

    print("pool_overlap.json")
    want("overlap ids", len(ov), m["overlap_size"])
    want("only-128 ids", len(o128), m["only128_size"])
    want("only-295 ids", len(o295), m["only295_size"])
    want("overlap + only128 == the 128-pool", len(ov) + len(o128), m["pool128_size"])
    want("overlap + only295 == the 295-pool", len(ov) + len(o295), m["pool295_size"])

    overlaps = [(a_, b_) for a_, b_, n in
                (("overlap", "only128", ov & o128), ("overlap", "only295", ov & o295),
                 ("only128", "only295", o128 & o295)) if n]
    print(f"  {'ok  ' if not overlaps else 'FAIL'}  "
          f"{'the three id sets are pairwise disjoint':46} "
          f"{'yes' if not overlaps else overlaps}")
    if overlaps:
        fail.append(f"id sets intersect: {overlaps}")

    union = ov | o128 | o295
    want("distinct pool records", len(union), TOTAL - 1)
    want("pool records + 1 orphan", len(union) + 1, m["merged_total"])
    want("the appendix's 295 + 96 + 1", m["pool295_size"] + m["only128_size"] + 1, TOTAL)

    ev = json.loads((HERE / "silent_evidence_392.json").read_text())["bugs"]
    ids = {b.get("bug_id") or b.get("id") for b in ev}
    print("\nsilent_evidence_392.json")
    want("evidence records", len(ids), TOTAL)
    print(f"  {'ok  ' if union <= ids else 'FAIL'}  "
          f"{'every pool record appears in the evidence file':46} "
          f"{'yes' if union <= ids else sorted(union - ids)[:5]}")
    if not union <= ids:
        fail.append("pool records missing from the evidence file")

    extra = sorted(ids - union)
    included = [k for k, v in m["orphan_decision"].items() if v.startswith("included")]
    print(f"  {'ok  ' if extra == included else 'FAIL'}  "
          f"{'the one non-pool record is the included orphan':46} {extra}")
    if extra != included:
        fail.append(f"non-pool records {extra} != included orphan {included}")

    excluded = [k for k, v in m["orphan_decision"].items() if v.startswith("excluded")]
    print(f"  {'ok  ' if not (set(excluded) & ids) else 'FAIL'}  "
          f"{'the excluded orphan stayed out':46} {excluded}")
    if set(excluded) & ids:
        fail.append(f"excluded orphan present: {set(excluded) & ids}")

    if a.check:
        if fail:
            print(f"\nFAIL: {fail}", file=sys.stderr)
            return 1
        print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
