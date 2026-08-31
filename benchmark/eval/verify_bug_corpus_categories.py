#!/usr/bin/env python3
"""Every per-bug config reconciles with the manifest, by one of two mechanisms.

benchmark/bugs/<ID>/config.json is what a reviewer sampling the corpus reads, and
its `category` disagrees with manifest_v2.json's on 161 of 296 records. That looks
alarming until the two reconciliation mechanisms are applied, and it takes both:

  agree outright                                        132
  resolved by that record's own alias_map entry in
  category_resolved.json (e.g. optimizer -> optimizer_state) 154
  resolved by manifest_v2.json's category_origin =
  "128_pool_override" (e.g. D-029 communication -> offload)    7
  one side has no category                                3
  UNEXPLAINED                                              0

The second mechanism is the non-obvious one. For those 7 the resolved record says
the category is UNCHANGED -- `communication -> communication`, "already_in_13_class"
-- while the manifest carries something else, so consulting category_resolved.json
alone makes them look like a three-way contradiction between config, resolution
record and manifest. The manifest records the override in `category_origin`, and
that field is what closes it.

This is a clean negative: the corpus is internally consistent. It is worth pinning
because the reconciliation is not discoverable from either file alone, and because
D-029 and O-005 are Real-SE evaluation cases whose coarse label feeds
tab:realse-class-coverage -- D-029 is the Offload representative, and its config
says `communication`.

  python3 benchmark/eval/verify_bug_corpus_categories.py [--check]
"""
from __future__ import annotations
import argparse, collections, glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "benchmark" / "eval" / "manifest_v2.json"
RES = ROOT / "benchmark" / "eval" / "category_resolved.json"
BUGS = ROOT / "benchmark" / "bugs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not MAN.exists() or not RES.exists() or not BUGS.is_dir():
        print("SKIP: manifest, resolution record or bug corpus absent")
        return 0

    man = {b["bug_id"]: b for b in json.loads(MAN.read_text())["bugs"]}
    recs = json.loads(RES.read_text())["records"]
    per = {r["bug_id"]: (r.get("old_category"), r.get("new_category")) for r in recs}

    agree = by_alias = by_override = missing = 0
    unexplained: list = []
    field_mismatch = collections.Counter()
    cfgs = sorted(glob.glob(str(BUGS / "*" / "config.json")))
    for c in cfgs:
        bid = Path(c).parent.name
        d = json.loads(Path(c).read_text())
        m = man.get(bid)
        if not m:
            unexplained.append((bid, "not in manifest", ""))
            continue
        for f in ("framework", "fixed_commit", "buggy_commit"):
            x, y = str(d.get(f) or "").strip(), str(m.get(f) or "").strip()
            if x and y and x.lower() != y.lower():
                field_mismatch[f] += 1
        cc, mc = (d.get("category") or "").strip(), (m.get("category") or "").strip()
        if not cc or not mc:
            missing += 1
        elif cc == mc:
            agree += 1
        elif per.get(bid) == (cc, mc):
            by_alias += 1
        elif "override" in str(m.get("category_origin") or ""):
            by_override += 1
        else:
            unexplained.append((bid, cc, mc))

    print(f"per-bug configs: {len(cfgs)}")
    print(f"  category agrees outright                {agree:>4}")
    print(f"  resolved by the record's alias_map entry {by_alias:>4}")
    print(f"  resolved by manifest category_origin     {by_override:>4}")
    print(f"  one side has no category                 {missing:>4}")
    print(f"  UNEXPLAINED                              {len(unexplained):>4}")
    for bid, cc, mc in unexplained[:8]:
        print(f"      {bid}: config={cc!r} manifest={mc!r}")
    print(f"\n  framework/commit disagreements: {dict(field_mismatch) or 'none'}")

    if not a.check:
        return 0
    fails = []

    def want(label, cond):
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
        if not cond:
            fails.append(label)

    print()
    want("the corpus still ships 296 per-bug configs", len(cfgs) == 296)
    want("every config's bug_id is in the manifest",
         not any(u[1] == "not in manifest" for u in unexplained))
    want("no framework or commit field disagrees with the manifest", not field_mismatch)
    want("every category disagreement is explained", not unexplained)
    want("both reconciliation mechanisms are still needed",
         by_alias > 0 and by_override > 0)
    want("the alias map still carries the bulk of them", by_alias >= 150)
    want("the 128-pool override still accounts for the rest", by_override == 7)

    if fails:
        print(f"\n{len(fails)} assertion(s) failed")
        return 1
    print("\nthe per-bug corpus reconciles with the manifest; both mechanisms are needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
