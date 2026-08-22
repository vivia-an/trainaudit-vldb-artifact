#!/usr/bin/env python3
"""The corpus's own declared uncertainty, per record.

The paper reports inter-rater reliability as kappa = 0.566 over a 50-record subset. The
annotations also carry `borderline_flags` on every record, which the paper does not report --
a whole-corpus, per-record signal of the same thing.

The flags mix three different meanings and must not be summed. Doing so gives 56% of the
corpus "borderline", which overstates it badly:

  scope declaration  SOURCE_ONLY_DETECTION, EXCEEDS_TIER6 -- facts about detectability, not
                     doubt about the coding
  coding ambiguity   PATTERN_AMBIGUOUS_*, CATEGORY_AMBIGUOUS_* -- the annotator could not
                     decide between two labels. This is the reliability-relevant class.
  weak evidence      INFERRED_FROM_SPARSE_CONFIG

    python3 benchmark/eval/coding_uncertainty.py
"""
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def classify(flag):
    if flag.startswith(("PATTERN_AMBIGUOUS", "CATEGORY_AMBIGUOUS")):
        return "coding ambiguity"
    if flag in ("SOURCE_ONLY_DETECTION", "EXCEEDS_TIER6"):
        return "scope declaration"
    if flag.startswith("INFERRED_FROM"):
        return "weak evidence"
    return "other"


def main():
    f = HERE / "v2_full" / "annotations_392_v2.json"
    if not f.exists():
        sys.exit(f"missing {f}")
    recs = json.loads(f.read_text())["annotations"]

    per_class = collections.defaultdict(set)
    instances = collections.Counter()
    for i, r in enumerate(recs):
        flags = r.get("borderline_flags") or []
        if isinstance(flags, str):
            flags = [flags] if flags else []
        for x in flags:
            per_class[classify(x)].add(i)
            instances[classify(x)] += 1

    any_flag = set().union(*per_class.values()) if per_class else set()
    print(f"{len(recs)} records; {len(any_flag)} carry at least one flag "
          f"({100 * len(any_flag) / len(recs):.1f}%) — but the classes mean different things:\n")
    for k in ("scope declaration", "coding ambiguity", "weak evidence", "other"):
        if k in per_class:
            n = len(per_class[k])
            print(f"  {k:<20}{n:>4} records ({100 * n / len(recs):>4.1f}%)"
                  f"   {instances[k]} flag instances")

    amb = per_class.get("coding ambiguity", set())
    print(f"\nThe reliability-relevant figure is coding ambiguity: "
          f"{len(amb)}/{len(recs)} = {100 * len(amb) / len(recs):.1f}%.")
    print("The paper reports kappa = 0.566 over a 50-record subset; this is the whole corpus,")
    print("and the two are consistent — both say the coding is moderately reliable.\n")

    per_cat = collections.defaultdict(lambda: [0, 0])
    for i, r in enumerate(recs):
        c = str(r.get("category"))
        per_cat[c][1] += 1
        if i in amb:
            per_cat[c][0] += 1
    print("coding ambiguity by category (the axis of tab:taxonomy):")
    for c, (a, b) in sorted(per_cat.items(), key=lambda x: -x[1][0] / max(x[1][1], 1)):
        bar = "#" * round(20 * a / b)
        print(f"  {c:<22}{a:>3}/{b:<4}{100 * a / b:>5.0f}%  {bar}")
    print("""
Reporting this alongside kappa would be more forthcoming than kappa alone, and it pre-empts a
reviewer who opens annotations_392_v2.json and finds the flags unmentioned. Note that summing
all flags gives 56%, which is the number a hostile reading would quote.""")


if __name__ == "__main__":
    main()
