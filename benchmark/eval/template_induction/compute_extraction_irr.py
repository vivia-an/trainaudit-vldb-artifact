"""E1 vs E2 extraction agreement over the 79-case double-coded subset.

Reports raw agreement and Cohen's kappa on expected_relation (raw and
alias-normalized), training_phase, and normalized topology_scope.
Usage: python3 compute_extraction_irr.py [catalog_json_for_alias_map]
"""
import json
import sys
from pathlib import Path

from kappa import cohen_kappa

HERE = Path(__file__).parent


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm_topo(s):
    s = (s or "").lower().replace(" ", "")
    if s in ("any", "single-rank", "singlerank", "none"):
        return "any/single"
    dims = [d for d in ("tp", "dp", "pp", "ep", "cp") if d + ">1" in s or d + ">=1" in s]
    return "+".join(sorted(dims)) if dims else s[:20]


def main():
    e1 = {}
    for f in list((HERE / "seed").glob("extract_e1_p*.jsonl")) + \
             list((HERE / "development").glob("extract_e1_p*.jsonl")):
        for r in load_jsonl(f):
            e1[r["case_id"]] = r
    e2 = {}
    for f in (HERE / "analysis").glob("extract_e2_p*.jsonl"):
        for r in load_jsonl(f):
            e2[r["case_id"]] = r

    alias = {}
    if len(sys.argv) > 1:
        alias = json.load(open(sys.argv[1])).get("operator_alias_map", {})

    common = sorted(set(e1) & set(e2))
    print(f"double-coded cases: {len(common)}")

    def report(name, key, normfn=lambda x: x):
        pairs = [(normfn(e1[c].get(key)), normfn(e2[c].get(key))) for c in common]
        po, k = cohen_kappa(pairs)
        print(f"{name:34s} raw_agreement={po:.3f} kappa={k:.3f}")
        return pairs

    report("expected_relation (raw)", "expected_relation")
    report("expected_relation (alias-normed)", "expected_relation",
           lambda x: alias.get(x, x))
    report("training_phase", "training_phase")
    report("topology_scope (normalized)", "topology_scope", norm_topo)

    disagree = [c for c in common
                if alias.get(e1[c]["expected_relation"], e1[c]["expected_relation"])
                != alias.get(e2[c]["expected_relation"], e2[c]["expected_relation"])]
    json.dump(
        [{"case_id": c, "e1": e1[c]["expected_relation"], "e2": e2[c]["expected_relation"],
          "e1_stmt": e1[c]["relation_statement"], "e2_stmt": e2[c]["relation_statement"]}
         for c in disagree],
        open(HERE / "analysis" / "extraction_irr_disagreements.json", "w"),
        indent=1, ensure_ascii=False)
    print(f"disagreements written: {len(disagree)} -> analysis/extraction_irr_disagreements.json")


if __name__ == "__main__":
    main()
