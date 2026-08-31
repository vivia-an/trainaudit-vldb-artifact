#!/usr/bin/env python3
"""How many of the three predicates each corpus record actually needs.

§4.2 argues that a deployable constraint needs all three predicates — pi_schema for the
property, pi_topo for the topology scope, pi_precond for the phase. The corpus already
records that judgement per bug, in `minimum_sufficient_layer`, and the paper does not use it.
This tabulates it.

Re-aggregation of recorded annotations; nothing is inferred.

    python3 benchmark/eval/predicate_necessity.py
"""
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
LABEL = {
    "schema_only": "pi_schema alone",
    "schema_topo": "pi_schema + pi_topo",
    "schema_topo_precond": "all three",
    "none": "not applicable / no online check",
}


def main():
    f = HERE / "v2_full" / "annotations_392_v2.json"
    if not f.exists():
        sys.exit(f"missing {f}")
    recs = json.loads(f.read_text())["annotations"]

    layer = collections.Counter(str(r.get("minimum_sufficient_layer")) for r in recs)
    applicable = sum(n for k, n in layer.items() if k != "none")
    print(f"{len(recs)} corpus records; {applicable} carry a minimum-sufficient layer\n")
    print(f"{'minimum sufficient predicate set':<34}{'records':>9}{'share':>9}")
    for k in ("schema_only", "schema_topo", "schema_topo_precond"):
        n = layer.get(k, 0)
        print(f"{LABEL[k]:<34}{n:>9}{100 * n / applicable:>8.1f}%")
    print(f"{LABEL['none']:<34}{layer.get('none', 0):>9}{'—':>9}")

    need_all = layer.get("schema_topo_precond", 0)
    print(f"\n{need_all} of {applicable} ({100 * need_all / applicable:.0f}%) need all three "
          f"predicates; {layer.get('schema_only', 0)} are expressible with pi_schema alone.")

    by_type = collections.defaultdict(collections.Counter)
    for r in recs:
        lay = str(r.get("minimum_sufficient_layer"))
        if lay == "none":
            continue
        by_type[str(r.get("invariant_type"))][lay] += 1
    print(f"\nby invariant type (the families where pi_schema alone suffices):")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1].get("schema_only", 0))[:6]:
        tot = sum(c.values())
        print(f"  {t:<32}all three {c.get('schema_topo_precond',0):>4}/{tot:<4}"
              f"  schema-only {c.get('schema_only',0)}")
    print("""
This is the corpus's own coding, so it inherits that process's judgement — it says an
annotator considered all three predicates necessary for the check to be deployable, not that
a two-predicate version was built and measured firing. It corroborates §4.2's argument at
corpus scale; the measured version of the same claim is the guard ablation in §5.3.""")


if __name__ == "__main__":
    main()
