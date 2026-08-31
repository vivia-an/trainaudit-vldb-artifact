#!/usr/bin/env python3
"""Cut a temporal split of the evidence corpus from resolved upstream dates.

Sec 5.4 records the temporal holdout as blocked for want of per-record dates. It is not
blocked by the data: resolve_record_dates.py recovers a date for almost every record from
its upstream issue or commit. This turns those dates into a split a holdout can be run
against, and describes what the split buys — how many records fall either side, and
whether the frameworks stay represented on both sides, which is what decides if the
result would mean anything.

    python3 make_temporal_split.py                    # median split
    python3 make_temporal_split.py --cutoff 2026-01-01
    python3 make_temporal_split.py --train-frac 0.7
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DATES = HERE / "record_dates.csv"
MANIFEST = HERE / "manifest_v2.json"
OUT = HERE / "temporal_split.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", help="ISO date; records before it are the training side")
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="if no --cutoff, pick the date putting this fraction earlier (default 0.5)")
    args = ap.parse_args()

    if not DATES.exists():
        sys.exit(f"{DATES.name} not found — run resolve_record_dates.py first")
    rows = [r for r in csv.DictReader(DATES.open()) if r["date"]]
    if not rows:
        sys.exit("no dated records yet")

    meta = {b["bug_id"]: b for b in json.loads(MANIFEST.read_text())["bugs"]}
    dated = sorted(rows, key=lambda r: r["date"])
    cutoff = args.cutoff or dated[int(len(dated) * args.train_frac)]["date"]

    train = [r for r in dated if r["date"] < cutoff]
    test = [r for r in dated if r["date"] >= cutoff]
    undated = sum(1 for r in csv.DictReader(DATES.open()) if not r["date"])

    def fw(rs):
        return collections.Counter((meta.get(r["bug_id"], {}).get("framework") or r["framework"] or "?")
                                   for r in rs)

    def cat(rs):
        return collections.Counter(meta.get(r["bug_id"], {}).get("category", "?") for r in rs)

    print(f"corpus            {len(dated)} dated, {undated} undated")
    print(f"span              {dated[0]['date']} .. {dated[-1]['date']}")
    print(f"cutoff            {cutoff}")
    print(f"train (before)    {len(train)}")
    print(f"test  (on/after)  {len(test)}\n")

    ftr, fte = fw(train), fw(test)
    print(f"  {'framework':<18}{'train':>7}{'test':>7}")
    for k in sorted(set(ftr) | set(fte)):
        print(f"  {k:<18}{ftr.get(k, 0):>7}{fte.get(k, 0):>7}")

    ctr, cte = cat(train), cat(test)
    both = sum(1 for k in set(ctr) | set(cte) if ctr.get(k) and cte.get(k))
    print(f"\n  categories        {len(set(ctr) | set(cte))} total, {both} present on both sides")
    only_test = sorted(k for k in cte if not ctr.get(k))
    if only_test:
        print(f"  test-only         {', '.join(only_test)}")
        print("                    (these are the genuinely unseen classes a temporal holdout tests)")

    payload = {
        "cutoff": cutoff, "n_train": len(train), "n_test": len(test), "n_undated": undated,
        "span": [dated[0]["date"], dated[-1]["date"]],
        "train_ids": [r["bug_id"] for r in train],
        "test_ids": [r["bug_id"] for r in test],
        "note": "Dates are upstream issue-creation dates where available, otherwise commit dates; "
                "see record_dates.csv for which applies per record.",
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {OUT.name}")
    print("\nTo run the holdout: mine the Pattern Catalog using only train_ids, then measure "
          "coverage on test_ids. A catalog built before the cutoff that still covers the "
          "test-side records is evidence the templates generalise forward in time rather than "
          "fitting the pool they were drawn from.")


if __name__ == "__main__":
    main()
