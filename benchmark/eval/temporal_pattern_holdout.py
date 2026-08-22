#!/usr/bin/env python3
"""Temporal pattern-coverage holdout, from recorded data alone.

§5.4 lists the temporal holdout as blocked. The dates turned out to be recoverable
(resolve_record_dates.py) and the split is in temporal_split.json, but running the real
holdout means re-mining a catalog on the train side, which needs an LLM. This measures the
question one layer up, using only what is recorded: were the *pattern families* present in
post-cutoff bugs already present in pre-cutoff ones?

It is coverage, not detection — see the caveats it prints.

    python3 benchmark/eval/temporal_pattern_holdout.py
"""
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ann_f = HERE / "v2_full" / "annotations_392_v2.json"
    split_f = HERE / "temporal_split.json"
    for f in (ann_f, split_f):
        if not f.exists():
            sys.exit(f"missing {f.name}"
                     + ("  (run resolve_record_dates.py then make_temporal_split.py)"
                        if f is split_f else ""))

    ann = {r["bug_id"]: r for r in json.loads(ann_f.read_text())["annotations"]}
    split = json.loads(split_f.read_text())
    train, test = set(split["train_ids"]), set(split["test_ids"])

    def dist(ids, key):
        return collections.Counter(str(ann.get(b, {}).get(key)) for b in ids)

    ptr, pte = dist(train, "pattern_id"), dist(test, "pattern_id")
    seen = {p for p in ptr if p != "None"}
    covered = sum(n for p, n in pte.items() if p != "None" and p in seen)
    total = sum(n for p, n in pte.items() if p != "None")
    unseen = sorted(p for p in pte if p != "None" and p not in seen)

    print(f"cutoff {split['cutoff']}: {len(train)} train / {len(test)} test "
          f"({split['n_undated']} records undated, excluded)\n")
    print(f"{'pattern':<10}{'train':>7}{'test':>7}")
    for p in sorted(set(ptr) | set(pte)):
        print(f"{p:<10}{ptr.get(p, 0):>7}{pte.get(p, 0):>7}")

    print(f"\npattern families in train: {len(seen)}   test-only: {len(unseen)} "
          f"{unseen if unseen else ''}")
    print(f"test-period records whose family was already present pre-cutoff: "
          f"{covered}/{total} ({100 * covered / total:.1f}%)")

    ftr, fte = dist(train, "framework"), dist(test, "framework")
    print(f"\n{'framework':<14}{'train':>7}{'test':>7}")
    for k in sorted(set(ftr) | set(fte)):
        print(f"{k:<14}{ftr.get(k, 0):>7}{fte.get(k, 0):>7}")

    print("""
Read this as coverage, not detection. Three caveats, in order of how much they matter:

  1. It says the pattern family was known pre-cutoff, not that a rule mined pre-cutoff would
     fire on the later bug. The real holdout needs a re-mine.
  2. The pattern_id labels were assigned with the whole corpus in view, so a family may have
     been defined partly by the very post-cutoff bugs it is credited with covering. That
     hindsight inflates coverage by an unknown amount.
  3. The split confounds with framework (GAP_AUDIT O22): OLMo is 69/5 and OLMo-core 5/63,
     because OLMo-core is the newer project. Part of what looks temporal is cross-framework.""")


if __name__ == "__main__":
    main()
