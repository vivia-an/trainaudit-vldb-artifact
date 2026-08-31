#!/usr/bin/env python3
"""Check tab:taxonomy against the corpus -- and pin which column it uses.

`benchmark/eval/v2_full/category_392_v2.csv` carries **two** count columns for the same 13
categories, `v2_count` and `manifest_count`, plus the `delta` between them. Both sum to 392,
so the disagreement is a reclassification between two labelling passes rather than a
counting error -- but it is a substantial one: **12 of the 13 categories differ**, the
largest by 13 records (numerical, 62 vs 75, a fifth of that category).

The paper's `tab:taxonomy` matches **`manifest_count`** on all 13 rows. It matches
`v2_count` on only one. That is worth pinning down, because the file is named `..._v2.csv`
and leads with the `v2_count` column, so anyone verifying the table -- or later
regenerating it -- would naturally reach for the wrong one and find twelve mismatches.

This asserts the paper's numbers against `manifest_count`, and separately that `v2_count`
still does *not* reproduce them, so that if the two passes are ever reconciled the check
notices rather than silently passing.

**The decisive test.** `manifest_v2.json` holds a per-record `category` for all 392
records. Aggregating it reproduces the paper's 13 counts **exactly, 13 of 13**, and matches
`v2_count` on one. So `tab:taxonomy` is verified against per-record data rather than
against a summary, and `manifest_count` is confirmed authoritative a third time. The file
also records 7 known label disagreements between the two pools, which is the authors
documenting the conflict rather than hiding it.

A second, independent line of evidence points the same way. `category_resolved.json` holds
a **per-record** taxonomy label for each of the 295-pool records, in exactly this 13-label
vocabulary. Those 295 are a subset of the 392, so no category's resolved count can exceed
the corpus count. Against `manifest_count` that holds for all 13. Against `v2_count` it
**fails** -- 25 records resolve to `optimizer_state` where `v2_count` records only 23 --
which is impossible for a subset, and is a second reason to treat `manifest_count` as
authoritative.

  python3 benchmark/eval/verify_taxonomy_table.py [--check]
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "v2_full" / "category_392_v2.csv"
EVID = HERE / "silent_evidence_392.json"

# tab:taxonomy as it appears in paper/main.tex, keyed by the CSV's category slug.
PAPER = {
    "numerical": 75, "control_flow": 50, "gradient_sync": 46, "checkpoint": 32,
    "moe": 28, "optimizer_state": 28, "data_loading": 28, "communication": 26,
    "dtype": 24, "loss_computation": 23, "sharding": 13, "lr_schedule": 13,
    "offload": 6,
}
TOTAL = 392


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    fail = []

    rows = {r["category"]: r for r in csv.DictReader(CSV.open())}
    print(f"{'category':18}{'paper':>7}{'manifest':>10}{'v2':>6}{'delta':>7}")
    man_ok = v2_ok = 0
    for cat, want in sorted(PAPER.items(), key=lambda kv: -kv[1]):
        if cat not in rows:
            fail.append(f"{cat} missing from the CSV")
            continue
        m, v = int(rows[cat]["manifest_count"]), int(rows[cat]["v2_count"])
        man_ok += m == want
        v2_ok += v == want
        flag = "" if m == want else "   <-- MISMATCH vs manifest"
        print(f"{cat:18}{want:>7}{m:>10}{v:>6}{rows[cat]['delta']:>7}{flag}")
        if m != want:
            fail.append(f"{cat}: paper {want} != manifest {m}")

    print(f"\nrows matching manifest_count: {man_ok}/{len(PAPER)}")
    print(f"rows matching v2_count:       {v2_ok}/{len(PAPER)}"
          f"   <- the paper does NOT use this column")
    if len(rows) != 13:
        fail.append(f"expected 13 categories, found {len(rows)}")
    for col in ("manifest_count", "v2_count"):
        s = sum(int(r[col]) for r in rows.values())
        print(f"{col} sums to {s}" + ("" if s == TOTAL else f"  <-- expected {TOTAL}"))
        if s != TOTAL:
            fail.append(f"{col} sums to {s}, not {TOTAL}")

    # category_resolved.json carries a per-record taxonomy label for the 295-pool, in
    # exactly this 13-label vocabulary. Since those 295 records are a subset of the 392,
    # each category's resolved count must not exceed the corpus count -- an independent
    # test of which column is authoritative.
    resolved = Counter(r["new_category"] for r in
                       json.loads((HERE / "category_resolved.json").read_text())["records"])
    print("\nper-record labels in category_resolved.json (the 295-pool):")
    unknown = set(resolved) - set(rows)
    if unknown:
        fail.append(f"resolved labels outside the 13: {sorted(unknown)}")
    print(f"  {len(resolved)} distinct labels, all within the 13: {not unknown}")
    if sum(resolved.values()) != 295:
        fail.append(f"resolved records sum to {sum(resolved.values())}, not 295")
    for col, expect_subset in (("manifest_count", True), ("v2_count", False)):
        over = [(c, resolved[c], int(rows[c][col])) for c in resolved
                if resolved[c] > int(rows[c][col])]
        holds = not over
        print(f"  subset of {col:15} {'holds' if holds else 'VIOLATED by ' + str(over)}")
        if expect_subset and not holds:
            fail.append(f"295-pool exceeds {col} for {over}")
        if not expect_subset and holds:
            fail.append("v2_count unexpectedly also satisfies the documented subset discriminator")

    # The strongest form: rebuild the 13 counts from the per-record labels for the whole
    # corpus, rather than comparing the paper against a summary someone typed.
    mv2 = json.loads((HERE / "manifest_v2.json").read_text())
    per = Counter(b["category"] for b in mv2["bugs"])
    print("\nrebuilt from manifest_v2.json's per-record `category` (all 392 records):")
    bad = [(c, per.get(c, 0), want) for c, want in PAPER.items() if per.get(c, 0) != want]
    print(f"  matches the paper on {len(PAPER) - len(bad)}/{len(PAPER)} categories"
          + (f"  MISMATCH {bad}" if bad else ""))
    if bad:
        fail.extend(f"per-record {c}: {got} != paper {want}" for c, got, want in bad)
    if sum(per.values()) != TOTAL:
        fail.append(f"per-record labels sum to {sum(per.values())}, not {TOTAL}")
    v2_hits = sum(1 for c in rows if per.get(c, 0) == int(rows[c]["v2_count"]))
    print(f"  the same aggregation matches v2_count on {v2_hits}/{len(rows)}"
          f"   <- third independent reason manifest_count is authoritative")
    if v2_hits == len(rows):
        fail.append("v2_count unexpectedly also matches the authoritative per-record labels")

    # The other complete labelling: annotations_392_v2.json, which reproduces v2_count.
    ann = json.loads((HERE / "v2_full" / "annotations_392_v2.json").read_text())
    ann_recs = ann if isinstance(ann, list) else (
        ann.get("records") or ann.get("bugs")
        or next(v for v in ann.values() if isinstance(v, list)))
    ann_per = Counter(r["category"] for r in ann_recs)
    hits_v2 = sum(1 for c in rows if ann_per.get(c, 0) == int(rows[c]["v2_count"]))
    hits_mf = sum(1 for c in rows if ann_per.get(c, 0) == int(rows[c]["manifest_count"]))
    print(f"\nannotations_392_v2.json ({len(ann_recs)} records) reproduces "
          f"v2_count on {hits_v2}/{len(rows)}, manifest_count on {hits_mf}/{len(rows)}")
    print("  so both columns have a per-record source; they are two annotation passes,")
    print("  and coding_uncertainty.py reports rates over this one, not the paper's.")
    if hits_v2 != len(rows):
        fail.append(f"annotations no longer reproduce v2_count ({hits_v2}/{len(rows)})")
    if len(ann_recs) != TOTAL:
        fail.append(f"annotations hold {len(ann_recs)} records, not {TOTAL}")

    dis = mv2.get("category_disagreements") or []
    print(f"  the file records {len(dis)} known label disagreement(s) between the two "
          f"pools, e.g. {dis[0]['bug_id']}: {dis[0].get('cat_128_pool')} vs "
          f"{dis[0].get('cat_295_resolved')}" if dis else "")

    n_bugs = len(json.loads(EVID.read_text())["bugs"])
    print(f"silent_evidence_392.json holds {n_bugs} bug records"
          + ("" if n_bugs == TOTAL else f"  <-- expected {TOTAL}"))
    if n_bugs != TOTAL:
        fail.append(f"evidence file holds {n_bugs}, not {TOTAL}")

    if v2_ok == len(PAPER):
        fail.append("v2_count now also reproduces the table -- the two passes were "
                    "reconciled; re-read which column is authoritative")

    if a.check:
        if fail:
            print(f"\nFAIL: {fail}", file=sys.stderr)
            return 1
        print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
