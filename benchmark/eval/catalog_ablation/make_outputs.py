"""S5 — emit the SPEC §7 deliverables from per_cell_results.json."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ("L1_raw_items", "L1_hypothesis", "L2_enumerated", "L3_pass",
          "L4_keep", "hyp_empty_entities", "hyp_l2_zero")
REPS = 5


def main():
    cells = json.loads((HERE / "per_cell_results.json").read_text())

    # ---- per-arm result files -------------------------------------------
    for arm in ("A", "B"):
        rows = [c for c in cells if c["arm"] == arm]
        per_rep = []
        for rep in range(REPS):
            rr = [c for c in rows if c["rep"] == rep]
            tot = {f: sum(c[f] for c in rr) for f in FIELDS}
            surv = sorted({s for c in rr for s in c["survivors"]})
            drops: Counter = Counter()
            for c in rr:
                drops += Counter(c["drops"])
            per_rep.append({
                "rep": rep, **tot,
                "yield_L4_over_L2": (tot["L4_keep"] / tot["L2_enumerated"]
                                     if tot["L2_enumerated"] else 0.0),
                "distinct_survivors": len(surv),
                "survivors": surv,
                "drops": dict(drops),
            })
        (HERE / f"arm_{arm.lower()}_results.json").write_text(json.dumps({
            "arm": arm,
            "use_catalog": arm == "A",
            "model": "deepseek-v4-flash",
            "temperature": 1.0,
            "max_tokens": 32768,
            "reps": REPS,
            "seed_file_entries_per_rep": 112,
            "per_rep": per_rep,
            "per_cell": rows,
        }, indent=2))

    # ---- summary csv -----------------------------------------------------
    agg = defaultdict(lambda: defaultdict(int))
    for c in cells:
        for f in FIELDS:
            agg[(c["arm"], c["rep"])][f] += c[f]

    def vals(arm, f):
        return [agg[(arm, r)][f] for r in range(REPS)]

    def yields(arm):
        return [agg[(arm, r)]["L4_keep"] / agg[(arm, r)]["L2_enumerated"]
                for r in range(REPS)]

    with (HERE / "catalog_ablation_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "rep", "L1_raw_items", "L1_hypothesis",
                    "L2_enumerated", "L3_pass", "L4_keep", "yield_L4_over_L2",
                    "L3_reject_rate", "distinct_survivors"])
        for arm in ("A", "B"):
            for rep in range(REPS):
                a = agg[(arm, rep)]
                surv = {s for c in cells
                        if c["arm"] == arm and c["rep"] == rep
                        for s in c["survivors"]}
                w.writerow([arm, rep, a["L1_raw_items"], a["L1_hypothesis"],
                            a["L2_enumerated"], a["L3_pass"], a["L4_keep"],
                            f'{a["L4_keep"]/a["L2_enumerated"]:.4f}',
                            f'{1-a["L3_pass"]/a["L2_enumerated"]:.4f}',
                            len(surv)])
        # median rows
        for arm in ("A", "B"):
            w.writerow([arm, "median"]
                       + [f"{st.median(vals(arm, f)):.0f}" for f in
                          ("L1_raw_items", "L1_hypothesis", "L2_enumerated",
                           "L3_pass", "L4_keep")]
                       + [f"{st.median(yields(arm)):.4f}",
                          f"{st.median([1-agg[(arm,r)]['L3_pass']/agg[(arm,r)]['L2_enumerated'] for r in range(REPS)]):.4f}",
                          len({s for c in cells if c["arm"] == arm
                               for s in c["survivors"]})])
        w.writerow(["ratio_A_over_B", "median", "", "", "", "", "",
                    f"{st.median(yields('A'))/st.median(yields('B')):.3f}",
                    "", ""])

    # ---- detection comparison (Step 4: blocked) --------------------------
    with (HERE / "detection_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "arm_a_detected", "arm_b_detected",
                    "fixed_side_fp", "status", "note"])
        manifest = json.loads(
            (HERE.parent / "real_sdc/real_sdc_manifest.json").read_text())
        for case in manifest["cases_confirmed_real"]:
            w.writerow([case["case_id"], "NOT_RUN", "NOT_RUN", "NOT_RUN",
                        "BLOCKED",
                        "requires GPU replay; eval-gpu-0 / beijing-olmo-gpu "
                        "unresolvable from this cluster and no cached "
                        "buggy/fixed duckdb trace exists under benchmark/bugs/"])

    print("wrote arm_a_results.json, arm_b_results.json, "
          "catalog_ablation_summary.csv, detection_comparison.csv")
    print(f"\nyield A median {st.median(yields('A')):.4f} "
          f"range [{min(yields('A')):.4f},{max(yields('A')):.4f}]")
    print(f"yield B median {st.median(yields('B')):.4f} "
          f"range [{min(yields('B')):.4f},{max(yields('B')):.4f}]")
    print(f"ratio A/B = {st.median(yields('A'))/st.median(yields('B')):.3f}")


if __name__ == "__main__":
    main()
