#!/usr/bin/env python3
"""Recompute frozen-catalog generalization aggregates from per-case records.

The verifier checks the three released arms, the catalog-size curve, and the
temporal holdout split. It validates aggregation and record consistency; it does
not regenerate the recorded coverage labels.

Run with:
  python3 benchmark/eval/verify_catalog_generalization.py --check
"""
from __future__ import annotations
import argparse, collections, csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "catalog_generalization"
ARMS = {"A_catalog": "cov_A", "B1_freeform_frozen": "cov_B1",
        "B2_freeform_remine": "cov_B2"}
NON_OBSERVABLE = "exceeds_tier6"


def load(stem):
    return [json.loads(l) for l in (HERE / f"{stem}.jsonl").read_text().splitlines()
            if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    summary = {r["arm"]: r for r in csv.DictReader(
        (HERE / "generalization_summary.csv").open())
        if r.get("arm") in ARMS}
    fail = []

    print(f"{'arm':22}{'covered/n':>12}{'pct':>8}   {'observable':>12}{'pct':>8}")
    for arm, stem in ARMS.items():
        rows = load(stem)
        cov = sum(1 for r in rows if str(r.get("covered")).lower() == "true")
        obs = [r for r in rows if r.get("tier_field") != NON_OBSERVABLE]
        cov_o = sum(1 for r in obs if str(r.get("covered")).lower() == "true")
        s = summary[arm]
        print(f"{arm:22}{cov:>6}/{len(rows):<5}{cov/len(rows)*100:>7.1f}%"
              f"   {cov_o:>6}/{len(obs):<5}{cov_o/len(obs)*100:>7.1f}%")
        for label, got, want in [("cov_all", cov, int(s["cov_all"])),
                                 ("n_all", len(rows), int(s["n_all"])),
                                 ("cov_observable", cov_o, int(s["cov_observable"])),
                                 ("n_observable", len(obs), int(s["n_observable"]))]:
            if got != want:
                fail.append(f"{arm}.{label}: {got} != {want}")
        for label, got, want in [("pct_all", cov / len(rows) * 100, float(s["pct_all"])),
                                 ("pct_observable", cov_o / len(obs) * 100,
                                  float(s["pct_observable"]))]:
            if abs(got - want) > 0.05:
                fail.append(f"{arm}.{label}: {got:.1f} != {want}")

    # The coverage-versus-size curve, by ranking templates on held-out usage.
    rows = load("cov_A")
    covered = [r for r in rows if str(r.get("covered")).lower() == "true"]
    freq = collections.Counter(r["matched_id"] for r in covered)
    published = {}
    lines = (HERE / "generalization_summary.csv").read_text().splitlines()
    seen_header = False
    for ln in lines:
        if ln.startswith("top_K"):
            seen_header = True
            continue
        if seen_header:
            parts = ln.split(",")
            if len(parts) == 2 and parts[0].strip().isdigit():
                published[int(parts[0])] = float(parts[1])
            elif ln.strip() == "":
                break

    print(f"\ncoverage vs catalog size ({len(freq)} templates used)")
    print(f"{'K':>4}{'recomputed':>13}{'published':>11}")
    for K, want in sorted(published.items()):
        top = {t for t, _ in freq.most_common(K)}
        got = sum(1 for r in covered if r["matched_id"] in top) / len(rows) * 100
        ok = abs(got - want) <= 0.05
        print(f"{K:>4}{got:>12.1f}%{want:>10.1f}%   {'ok' if ok else 'MISMATCH'}")
        if not ok:
            fail.append(f"curve K={K}: {got:.1f} != {want}")

    print(f"\nnon-observable records: "
          f"{len([r for r in load('cov_A') if r.get('tier_field') == NON_OBSERVABLE])}"
          "/249")

    # --- the holdout split, against the corpus ---
    def ids(d):
        if isinstance(d, list):
            return {x if isinstance(x, str) else (x.get("bug_id") or x.get("id"))
                    for x in d}
        for k in ("bugs", "ids", "held_out", "records"):
            if k in d:
                return ids(d[k])
        return set()

    held = ids(json.loads((HERE / "heldout_bugs.json").read_text()))
    freeze = ids(json.loads((HERE / "freeze_bugs.json").read_text()))
    corpus = {b["bug_id"] for b in json.loads(
        (HERE.parent / "manifest_v2.json").read_text())["bugs"]}
    po = json.loads((HERE.parent / "pool_overlap.json").read_text())
    pool128 = set(po["overlap_ids"]) | set(po["only128_ids"])
    excluded = corpus - freeze - held

    print("\nthe holdout split")
    for label, cond, detail in [
        ("held-out set is 249", len(held) == 249, len(held)),
        ("freeze set is the 128-pool exactly", freeze == pool128,
         f"{len(freeze)} ids, identical: {freeze == pool128}"),
        ("freeze and holdout are disjoint", not (freeze & held),
         f"overlap {len(freeze & held)}"),
        ("both lie inside the 392 corpus", (freeze | held) <= corpus,
         f"outside: {sorted((freeze | held) - corpus)[:3] or 'none'}"),
        ("the 15 left out are the B1-B15 seed bugs",
         excluded == {f"B{i}" for i in range(1, 16)},
         f"{len(excluded)}: {sorted(excluded, key=lambda s: int(s[1:]))}"),
    ]:
        print(f"  {'ok  ' if cond else 'FAIL'}  {label:44} {detail}")
        if not cond:
            fail.append(label)
    print("  split accounting: 128 freeze + 249 held out + 15 development = 392")

    if a.check:
        if fail:
            print(f"\nFAIL: {fail}", file=sys.stderr)
            return 1
        print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
