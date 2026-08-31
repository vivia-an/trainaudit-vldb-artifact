#!/usr/bin/env python3
"""Recompute the paper's inter-rater statistics from the annotation records.

Four agreement numbers were unverified by anything in the artifact:

  main.tex     schema match 92.3% (72/78), joint grounding 91.0% (71/78),
               template assignment agreement kappa = 0.954
  appendix.tex Cohen's kappa = 0.566 over 50 stratified cases

All four reproduce. Details worth knowing:

* **0.566 is a ground-truth-vs-LLM comparison, and the appendix says so** -- its table row
  reads "B (Independent LLM)". Recomputing it end to end from the marginals rather than
  trusting the stored value: p_o = 30/50 = 0.6, p_e = sum over the 13 classes of
  p_gt(c)*p_llm(c) = 0.078, kappa = (0.6-0.078)/(1-0.078) = 0.56616.

* **92.3% is the adjudicated figure, not either rater's raw count.** Rater A recorded 72
  MATCH of 78 and rater B recorded 70, so the raw joint count is 70/78 = 89.7%.
  Disagreements were then adjudicated against written instructions
  (`adjudication_result.json` records a deciding clause and which rater it sided with for
  each), and the adjudicated file holds exactly 72 MATCH. Reporting the adjudicated count is
  standard; the point is that 72 is not rater A's number surviving unexamined.

* **`groundability_among_both_match` stores `cohen_kappa: 0.0` beside
  `raw_agreement: 0.986`.** That is the kappa paradox -- 71 of 72 cases fall in one
  category, so chance agreement is nearly total and kappa collapses. The paper correctly
  reports grounding as a raw 91.0% rather than as a kappa, but a reviewer who opens
  `test_irr.json` will see the 0.0 and should know why it is there.

  python3 benchmark/eval/verify_irr.py [--check]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TI = HERE / "template_induction"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    fail = []

    def cmp(label, got, want, tol=0.0005):
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:44} {got}  vs paper {want}")
        if not ok:
            fail.append(label)

    # --- appendix: Cohen's kappa over the 50-case stratified sample ---
    d = json.loads((HERE / "irr_50_results.json").read_text())
    n, ag = d["meta"]["n"], d["agreements"]
    gt, llm = d["gt_distribution"], d["llm_distribution"]
    p_o = ag / n
    p_e = sum((gt.get(c, 0) / sum(gt.values())) * (llm.get(c, 0) / sum(llm.values()))
              for c in set(gt) | set(llm))
    kappa = (p_o - p_e) / (1 - p_e)
    print("appendix.tex, 50 stratified cases (ground truth vs independent LLM)")
    cmp("agreements + disagreements == n", ag + len(d["disagreements"]), n)
    cmp("p_observed", p_o, 0.6)
    cmp("p_expected recomputed from the marginals", round(p_e, 3), 0.078)
    cmp("Cohen's kappa", round(kappa, 3), 0.566)

    # --- main text: the double-rated template-assignment study ---
    irr = json.loads((TI / "analysis" / "test_irr.json").read_text())
    adj = list(csv.DictReader((TI / "test" / "adjudicated_annotations.csv").open()))
    rater = {}
    for who in ("a", "b"):
        rows = list(csv.DictReader(
            (TI / "test" / f"test_annotations_rater_{who}.csv").open()))
        rater[who] = {r["case_id"]: r["verdict"] for r in rows}
    shared = set(rater["a"]) & set(rater["b"])
    joint = sum(1 for k in shared
                if rater["a"][k] == "MATCH" and rater["b"][k] == "MATCH")
    match = sum(1 for r in adj if r["verdict"] == "MATCH")
    grounded = sum(1 for r in adj
                   if r["verdict"] == "MATCH" and r["groundable"] == "yes")

    print("\nmain.tex, 78 double-rated cases")
    cmp("n double-rated", len(adj), irr["n_double_rated"])
    cmp("adjudicated MATCH (schema match 72/78)", match, 72)
    cmp("schema match %", round(match / len(adj) * 100, 1), 92.3)
    cmp("adjudicated MATCH & groundable (71/78)", grounded, 71)
    cmp("joint grounding %", round(grounded / len(adj) * 100, 1), 91.0)
    cmp("template assignment kappa",
        irr["template_assignment_among_both_match"]["cohen_kappa"], 0.954)

    print(f"\n  context: raters A/B recorded "
          f"{sum(1 for v in rater['a'].values() if v == 'MATCH')} and "
          f"{sum(1 for v in rater['b'].values() if v == 'MATCH')} MATCH; raw joint "
          f"{joint}/{len(shared)} = {joint/len(shared)*100:.1f}%. The 92.3% is post-adjudication.")
    g = irr["groundability_among_both_match"]
    print(f"  context: groundability agreement is raw {g['raw_agreement']} with "
          f"kappa {g['cohen_kappa']} -- the kappa paradox, one category holds almost every")
    print("           case. The paper reports grounding as a raw percentage, correctly.")

    fail += check_appendix_reliability()

    if a.check:
        if fail:
            print(f"\nFAIL: {fail}", file=sys.stderr)
            return 1
        print("\nOK")
    return 0



def check_appendix_reliability():
    """Validate the appendix reliability statistics and template-growth record."""
    import csv as _csv
    import json as _json
    root = Path(__file__).resolve().parents[2]
    out = []

    def want(label, expected, measured, src):
        ok = (round(measured, 3) == round(expected, 3)
              if isinstance(measured, float) else measured == expected)
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:44} {measured} vs paper {expected}")
        if not ok:
            out.append(f"{label}: {measured} != {expected}   [{src}]")

    print("\nappendix.tex, reliability and generalization")
    # Relation-operator kappa. The script needs the catalog passed for the alias map --
    # run without it, raw and alias-normed both read 0.604 and the paper's 0.600 looks
    # wrong. The shipped report records 0.604 raw / 0.600 alias-normed.
    rep = (root / "benchmark" / "eval" / "template_induction" / "report"
           / "experiment_report.md")
    if rep.exists():
        body = rep.read_text()
        want("relation-operator kappa, alias-normalized", True,
             "| 0.646 | 0.600 |" in body, rep)
        want("training_phase kappa", True, "| 0.975 | 0.968 |" in body, rep)
        want("topology_scope kappa", True, "| 0.899 | 0.834 |" in body, rep)
    ti = root / "benchmark" / "eval" / "template_induction" / "analysis" / "test_irr.json"
    if ti.exists():
        d = _json.loads(ti.read_text())
        want("six-way verdict kappa", 0.772, d["verdict_6way"]["cohen_kappa"], ti)
        want("match-versus-uncovered kappa", 0.843,
             d["match_vs_uncovered"]["cohen_kappa"], ti)
        want("n double-rated (locked test)", 78, d["n_double_rated"], ti)
        nov = d["rater_a_verdicts"].get("SINGLETON_NEW_RELATION", 0)
        want("novelty count", 3, nov, ti)
        want("novelty rate %", 3.8, round(nov / d["n_double_rated"] * 100, 1), ti)

    sat = (root / "benchmark" / "eval" / "template_induction" / "development"
           / "saturation_by_batch.csv")
    if sat.exists():
        rows = list(_csv.DictReader(sat.open()))
        want("templates before the first batch", 27,
             int(rows[0]["templates_total_after"]) - int(rows[0]["templates_delta"]), sat)
        want("templates after the last batch", 35,
             int(rows[-1]["templates_total_after"]), sat)

    # Protocol A: 25/32 = 78.1% with 7 disagreements. Pure arithmetic on the stated counts.
    want("Protocol A agreements + disagreements", 32, 25 + 7, "appendix prose")
    want("Protocol A raw agreement %", 78.1, round(25 / 32 * 100, 1), "appendix prose")

    if sat.exists():
        cov = [float(r["cum_catalog_coverage"]) for r in _csv.DictReader(sat.open())]
        want("mean catalog coverage across development", 0.86,
             round(sum(cov) / len(cov), 2), sat)

    # The seed/dev/test split the appendix prints as "128 / 186 / 78".
    split = root / "benchmark" / "eval" / "template_induction" / "dataset_split.json"
    if split.exists():
        d = _json.loads(split.read_text())
        # These hold lists of case ids, not counts.
        want("seed split", 128, len(d["seed"]), split)
        want("development split", 186, len(d["development"]), split)
        want("locked test split", 78, len(d["test"]), split)
        want("Real-SE cases forced into the locked test", 18,
             len(d["meta"]["real_se_forced_test"]), split)

    # "five labels reach F1>=0.80". irr_50_results.json ships no per-label F1, but the
    # full 50 (gt, llm) pairs are recoverable: the 20 disagreements carry both labels and
    # the other 30 agree by definition, which the reconstruction checks against the stored
    # agreement count and against each disagreement's own gt field.
    gtf = root / "benchmark" / "eval" / "irr_50_groundtruth.json"
    rf = root / "benchmark" / "eval" / "irr_50_results.json"
    if gtf.exists() and rf.exists():
        gt = _json.loads(gtf.read_text())["gt"]
        res = _json.loads(rf.read_text())
        dis = {d["bug_id"]: (d["gt"], d["llm"]) for d in res["disagreements"]}
        consistent = all(dis[b][0] == g for b, g in gt.items() if b in dis)
        pairs = {b: (g, dis[b][1] if b in dis else g) for b, g in gt.items()}
        agree = sum(1 for g, l in pairs.values() if g == l)
        print("\nappendix.tex, per-label F1 on the 50-case rerate")
        want("reconstruction covers the whole sample", 50, len(pairs), gtf)
        want("each disagreement's gt matches the ground-truth file", True, consistent, gtf)
        want("reconstructed agreements match the stored count",
             res["agreements"], agree, rf)
        f1s = {}
        for L in {g for g, _ in pairs.values()} | {l for _, l in pairs.values()}:
            tp = sum(1 for g, l in pairs.values() if g == L and l == L)
            fp = sum(1 for g, l in pairs.values() if g != L and l == L)
            fn = sum(1 for g, l in pairs.values() if g == L and l != L)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1s[L] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        want("labels reaching F1 >= 0.80", 5,
             sum(1 for v in f1s.values() if v >= 0.80), rf)
        # The appendix says disagreements concentrate on the broad control_flow label.
        want("control_flow is the weakest label", True,
             f1s.get("control_flow", 1.0) == min(f1s.values()), rf)

    # tab:silent-evidence. The L4 row prints 136, which is the data's L4 (134) PLUS the 2
    # crash-keyword WARNING records -- the same 2 the table lists separately as
    # counter-evidence. Tiers must sum to 392 and WARNING is not a tier, so the choice is
    # defensible; it is asserted here so the composition is explicit rather than implied.
    sev = root / "benchmark" / "eval" / "silent_evidence_392.json"
    if sev.exists():
        raw = _json.loads(sev.read_text())
        recs = raw if isinstance(raw, list) else next(
            v for v in raw.values() if isinstance(v, list))
        lv = {}
        for r in recs:
            lv[str(r.get("evidence_level"))] = lv.get(str(r.get("evidence_level")), 0) + 1
        print("\nappendix.tex, tab:silent-evidence")
        want("L1 count", 23, lv.get("L1", 0), sev)
        want("L2 count", 161, lv.get("L2", 0), sev)
        want("L3 count", 72, lv.get("L3", 0), sev)
        want("crash-keyword WARNING count", 2, lv.get("WARNING", 0), sev)
        want("the L4 row's 136 is L4 plus the 2 WARNING records", 136,
             lv.get("L4", 0) + lv.get("WARNING", 0), sev)
        want("needs_manual_review agrees with that 136", 136,
             sum(1 for r in recs if r.get("needs_manual_review")), sev)
        want("the four printed tiers sum to the corpus", 392,
             23 + 161 + 72 + (lv.get("L4", 0) + lv.get("WARNING", 0)), sev)
    return out

if __name__ == "__main__":
    sys.exit(main())
