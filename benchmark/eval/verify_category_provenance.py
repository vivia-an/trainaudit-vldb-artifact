#!/usr/bin/env python3
"""Every record states how its coarse category was derived, and it reconciles.

The corpus carries several category vocabularies. The per-record provenance fields map
between them and make the transformation checkable end to end.

The chain, for each of the 392 records:

    benchmark/bugs/<ID>/config.json  ->  `category`, the fine-grained original label
                                         (70 distinct values across 296 shipped configs)
    manifest_v2.json                 ->  `category`, the coarse 13-class label
    manifest_v2.json                 ->  `category_origin`, **the derivation itself**

`category_origin` is not another label -- it is a provenance string naming the rule that
was applied, e.g. `alias_map (moe_router_init → moe)` or
`128_pool_override (was: already_in_13_class)`. Distribution across the 392:

    alias_map                154    the fine label and its coarse target, written out
    already_in_13_class      131    the original label was already one of the 13
    128_pool_curated          96    taken from the 128-pool's curation
    128_pool_override          7
    manual: <reason>           2    each with its reason stated inline

The 154 `alias_map` rows are the ones that can be checked from both ends, and **all 154
reconcile**: the source named in the string equals the per-bug `config.json` category, and
the target equals the coarse category in the manifest. None disagrees.

The fine-to-coarse mapping is therefore recorded per record, together with the method used
to derive it. The appendix's descriptive subsystem labels remain a separate presentation
axis rather than another corpus category vocabulary.

  python3 benchmark/eval/verify_category_provenance.py [--check]
"""
from __future__ import annotations
import argparse, collections, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
BUGS = HERE.parent / "bugs"
ALIAS = re.compile(r"alias_map \((.+?) → (.+?)\)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    fail = []

    cfgs = {}
    for f in BUGS.rglob("config.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        cfgs[d.get("bug_id") or f.parent.name] = d
    bugs = {b["bug_id"]: b for b in
            json.loads((HERE / "manifest_v2.json").read_text())["bugs"]}

    print(f"per-bug config.json: {len(cfgs)}   manifest_v2 records: {len(bugs)}")
    if len(cfgs) < 290:
        fail.append(f"only {len(cfgs)} per-bug configs found (corpus thin?)")

    # Fields that should agree exactly between the two independent sources.
    for field in ("framework", "buggy_commit", "fixed_commit"):
        shared = [b for b in cfgs if b in bugs
                  and str(cfgs[b].get(field, "")).strip()
                  and str(bugs[b].get(field, "")).strip()]
        dis = [b for b in shared
               if str(cfgs[b][field]).strip() != str(bugs[b][field]).strip()]
        print(f"  {'ok  ' if not dis else 'FAIL'}  {field:14} agrees on "
              f"{len(shared) - len(dis)}/{len(shared)} shared records"
              + (f"  DISAGREE {dis[:3]}" if dis else ""))
        if dis:
            fail.append(f"{field} disagrees on {len(dis)} records: {dis[:5]}")

    kinds = collections.Counter(
        (str(b.get("category_origin") or "").split("(")[0].strip() or "(empty)")
        for b in bugs.values())
    print(f"\n  category_origin kinds: {dict(kinds.most_common(6))}")
    if kinds.get("(empty)", 0):
        fail.append(f"{kinds['(empty)']} records have no category_origin")

    ok = bad = 0
    worst = []
    for bid, rec in bugs.items():
        m = ALIAS.match(str(rec.get("category_origin") or ""))
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        good = dst == rec["category"] and (
            bid not in cfgs or str(cfgs[bid].get("category")) == src)
        if good:
            ok += 1
        else:
            bad += 1
            if len(worst) < 3:
                worst.append((bid, src, dst, rec["category"],
                              cfgs.get(bid, {}).get("category")))
    print(f"  {'ok  ' if not bad else 'FAIL'}  alias_map rows reconcile end to end: "
          f"{ok} ok, {bad} bad" + (f"  e.g. {worst}" if worst else ""))
    if bad:
        fail.append(f"{bad} alias_map rows do not reconcile")
    if ok < 150:
        fail.append(f"only {ok} alias_map rows found, expected ~154")

    if a.check:
        if fail:
            print(f"\nFAIL: {fail}", file=sys.stderr)
            return 1
        print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
