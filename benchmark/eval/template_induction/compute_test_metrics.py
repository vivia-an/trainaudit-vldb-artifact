"""Locked-test IRR + coverage metrics (protocol §7).

Run 1 (before adjudication): python3 compute_test_metrics.py
  -> analysis/test_irr.json, test/adjudication_input.json
Run 2 (after adjudication):  python3 compute_test_metrics.py --adjudicated
  -> analysis/test_metrics.json using test/adjudicated_annotations.csv
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from kappa import cohen_kappa

HERE = Path(__file__).parent
UNCOVERED = {"SINGLETON_NEW_RELATION", "UNOBSERVABLE", "REFERENCE_DEPENDENT",
             "NON_GROUNDABLE", "INSUFFICIENT_EVIDENCE"}


def load_rater(tag):
    out = {}
    for p in sorted((HERE / "test").glob(f"verdicts_rater_{tag}_p*.jsonl")):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["case_id"]] = r
    return out


def irr():
    a, b = load_rater("a"), load_rater("b")
    cases = sorted(set(a) & set(b))
    print(f"double-rated cases: {len(cases)} (a={len(a)}, b={len(b)})")

    pairs6 = [(a[c]["verdict"], b[c]["verdict"]) for c in cases]
    po6, k6 = cohen_kappa(pairs6)

    pairs2 = [("MATCH" if a[c]["verdict"] == "MATCH" else "UNCOVERED",
               "MATCH" if b[c]["verdict"] == "MATCH" else "UNCOVERED") for c in cases]
    po2, k2 = cohen_kappa(pairs2)

    both_match = [c for c in cases
                  if a[c]["verdict"] == "MATCH" and b[c]["verdict"] == "MATCH"]
    pairsT = [(a[c]["template_id"], b[c]["template_id"]) for c in both_match]
    poT, kT = cohen_kappa(pairsT)

    grounded = [c for c in both_match]
    pairsG = [(str(a[c].get("groundable", "")).startswith("yes"),
               str(b[c].get("groundable", "")).startswith("yes")) for c in grounded]
    poG, kG = cohen_kappa(pairsG)

    res = {
        "n_double_rated": len(cases),
        "verdict_6way": {"raw_agreement": round(po6, 3), "cohen_kappa": round(k6, 3)},
        "match_vs_uncovered": {"raw_agreement": round(po2, 3), "cohen_kappa": round(k2, 3)},
        "template_assignment_among_both_match": {
            "n": len(both_match), "raw_agreement": round(poT, 3),
            "cohen_kappa": round(kT, 3)},
        "groundability_among_both_match": {
            "n": len(grounded), "raw_agreement": round(poG, 3),
            "cohen_kappa": round(kG, 3)},
        "rater_a_verdicts": dict(Counter(r["verdict"] for r in a.values())),
        "rater_b_verdicts": dict(Counter(r["verdict"] for r in b.values())),
    }
    for k, v in res.items():
        print(k, ":", v)
    json.dump(res, open(HERE / "analysis" / "test_irr.json", "w"), indent=1)

    disagreements = [c for c in cases
                     if a[c]["verdict"] != b[c]["verdict"]
                     or (a[c]["verdict"] == "MATCH"
                         and a[c]["template_id"] != b[c]["template_id"])
                     or (a[c]["verdict"] == "MATCH"
                         and str(a[c].get("groundable", "")).startswith("yes")
                         != str(b[c].get("groundable", "")).startswith("yes"))]
    cases_by_id = {c["bug_id"]: c for c in
                   json.load(open(HERE / "inputs" / "test_cases.json"))}
    payload = [{"case_id": c, "case": cases_by_id[c],
                "rater_a": a[c], "rater_b": b[c]} for c in disagreements]
    json.dump(payload, open(HERE / "test" / "adjudication_input.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"\ndisagreements needing adjudication: {len(disagreements)}"
          f" -> test/adjudication_input.json")

    # rater CSV exports (deliverables)
    for tag, d in (("a", a), ("b", b)):
        with open(HERE / "test" / f"test_annotations_rater_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "verdict", "template_id", "alt_template_id",
                        "groundable", "relation_operator", "guard_note", "rationale"])
            for c in sorted(d):
                r = d[c]
                w.writerow([c, r["verdict"], r.get("template_id"), r.get("alt_template_id"),
                            r.get("groundable"), r.get("relation_operator"),
                            r.get("guard_note"), r.get("rationale")])
    print("wrote test/test_annotations_rater_{a,b}.csv")


def metrics():
    rows = list(csv.DictReader(open(HERE / "test" / "adjudicated_annotations.csv")))
    cases_by_id = {c["bug_id"]: c for c in
                   json.load(open(HERE / "inputs" / "test_cases.json"))}
    n = len(rows)
    matched = [r for r in rows if r["verdict"] == "MATCH"]
    joint = [r for r in matched if r["groundable"].strip().lower().startswith("yes")]
    novel = [r for r in rows if r["verdict"] == "SINGLETON_NEW_RELATION"]

    def by_fw(subset):
        return Counter(cases_by_id[r["case_id"]]["framework"] for r in subset)

    fw_total = Counter(cases_by_id[r["case_id"]]["framework"] for r in rows)
    fw_match, fw_joint = by_fw(matched), by_fw(joint)

    frozen = json.load(open(HERE / "frozen_template_catalog.json"))
    support = {t["template_id"]: t["support_count"] for t in frozen["templates"]}
    test_support = Counter(r["template_id"] for r in matched)

    res = {
        "n_test": n,
        "C_schema": round(len(matched) / n, 4),
        "C_joint": round(len(joint) / n, 4),
        "novelty_rate": round(len(novel) / n, 4),
        "verdict_histogram": dict(Counter(r["verdict"] for r in rows)),
        "per_framework": {fw: {"n": fw_total[fw],
                               "C_schema": round(fw_match[fw] / fw_total[fw], 3),
                               "C_joint": round(fw_joint[fw] / fw_total[fw], 3)}
                          for fw in sorted(fw_total)},
        "compression": {
            "n_templates": len(frozen["templates"]),
            "n_singletons_in_catalog": len(frozen["singletons"]),
            "distinct_templates_used_in_test": len(test_support),
            "test_support_per_template": dict(test_support.most_common()),
            "catalog_support_per_template": support,
        },
    }
    json.dump(res, open(HERE / "analysis" / "test_metrics.json", "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if "--adjudicated" in sys.argv:
        metrics()
    else:
        irr()
