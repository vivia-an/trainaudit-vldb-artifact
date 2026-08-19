"""Score S3 diagnosis-accuracy ratings.

Inputs:
  benchmark/eval/diagnosis_s3_rater_A.csv
  benchmark/eval/diagnosis_s3_rater_B.csv

Outputs (stdout + markdown table written to paper_table_diagnosis_s3.md):
  - L1 hit rate per rater (X/17)
  - L2 hit rate per rater (X/N where N excludes N/A)
  - Final L1/L2 verdict per case (agreement only; disagreement listed)
  - Cohen's kappa for L1 and (if available) L2
  - List of disagreement cases
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RATER_A = ROOT / "diagnosis_s3_rater_A.csv"
RATER_B = ROOT / "diagnosis_s3_rater_B.csv"
OUT_MD = ROOT / "paper_table_diagnosis_s3.md"


def load_csv(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bug_id = row["bug_id"].strip()
            if not bug_id:
                continue
            rows[bug_id] = {
                "L1": row["L1_correct"].strip(),
                "L2": row["L2_correct"].strip(),
                "confidence": row.get("confidence", "").strip(),
                "justification": row.get("justification", "").strip(),
            }
    return rows


def kappa_binary(labels_a, labels_b):
    """Cohen's kappa for two raters on a binary classification."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return float("nan")
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n
    cats = sorted({*labels_a, *labels_b})
    p_e = 0.0
    for c in cats:
        pa = sum(1 for x in labels_a if x == c) / n
        pb = sum(1 for x in labels_b if x == c) / n
        p_e += pa * pb
    if abs(1 - p_e) < 1e-12:
        return 1.0 if p_o == 1.0 else float("nan")
    return (p_o - p_e) / (1 - p_e)


def fmt_kappa(k):
    if k != k:  # NaN
        return "n/a"
    if k >= 0.81:
        tag = "almost perfect"
    elif k >= 0.61:
        tag = "substantial"
    elif k >= 0.41:
        tag = "moderate"
    elif k >= 0.21:
        tag = "fair"
    else:
        tag = "slight"
    return f"{k:.3f} ({tag})"


def main():
    a = load_csv(RATER_A)
    b = load_csv(RATER_B)
    if not b:
        print("Rater B file empty", file=sys.stderr)
        sys.exit(1)

    bug_ids = sorted(b.keys())
    if not a or all(not v["L1"] for v in a.values()):
        print("Rater A not filled yet — Rater B summary only:\n")
        l1 = [b[bid]["L1"] for bid in bug_ids]
        l2 = [b[bid]["L2"] for bid in bug_ids]
        l1_yes = l1.count("yes")
        l2_eligible = [v for v in l2 if v in ("yes", "no")]
        l2_yes = l2_eligible.count("yes")
        print(f"Rater B L1: {l1_yes}/{len(l1)}")
        print(
            f"Rater B L2: {l2_yes}/{len(l2_eligible)} "
            f"(N/A excluded: {l2.count('N/A')})"
        )
        return

    common = [bid for bid in bug_ids if bid in a]
    a_l1 = [a[bid]["L1"] for bid in common]
    b_l1 = [b[bid]["L1"] for bid in common]
    a_l2 = [a[bid]["L2"] for bid in common]
    b_l2 = [b[bid]["L2"] for bid in common]

    a_l1_hit = a_l1.count("yes")
    b_l1_hit = b_l1.count("yes")

    a_l2_elig = [v for v in a_l2 if v in ("yes", "no")]
    b_l2_elig = [v for v in b_l2 if v in ("yes", "no")]
    a_l2_hit = a_l2_elig.count("yes")
    b_l2_hit = b_l2_elig.count("yes")

    # Final: count cases where both raters say yes (conservative)
    final_l1 = sum(1 for x, y in zip(a_l1, b_l1) if x == "yes" and y == "yes")
    l2_pairs = [
        (x, y) for x, y in zip(a_l2, b_l2) if x in ("yes", "no") and y in ("yes", "no")
    ]
    final_l2_yes = sum(1 for x, y in l2_pairs if x == "yes" and y == "yes")
    final_l2_n = len(l2_pairs)

    kappa_l1 = kappa_binary(a_l1, b_l1)
    if l2_pairs:
        ka, kb = zip(*l2_pairs)
        kappa_l2 = kappa_binary(list(ka), list(kb))
    else:
        kappa_l2 = float("nan")

    print(f"n cases: {len(common)}")
    print(
        f"L1 hit: A={a_l1_hit}/{len(a_l1)}, B={b_l1_hit}/{len(b_l1)}, "
        f"final(both-yes)={final_l1}/{len(common)}"
    )
    print(
        f"L2 hit: A={a_l2_hit}/{len(a_l2_elig)}, B={b_l2_hit}/{len(b_l2_elig)}, "
        f"final(both-yes)={final_l2_yes}/{final_l2_n}"
    )
    print(f"Cohen's kappa L1: {fmt_kappa(kappa_l1)}")
    print(f"Cohen's kappa L2: {fmt_kappa(kappa_l2)}")

    disagree = [bid for bid in common if a[bid]["L1"] != b[bid]["L1"]]
    if disagree:
        print("\nL1 disagreements:")
        for bid in disagree:
            print(
                f"  {bid}: A={a[bid]['L1']} ({a[bid]['justification']}) | "
                f"B={b[bid]['L1']} ({b[bid]['justification']})"
            )
    else:
        print("\nL1 disagreements: none")

    l2_disagree = [
        bid
        for bid in common
        if a[bid]["L2"] in ("yes", "no")
        and b[bid]["L2"] in ("yes", "no")
        and a[bid]["L2"] != b[bid]["L2"]
    ]
    if l2_disagree:
        print("\nL2 disagreements:")
        for bid in l2_disagree:
            print(f"  {bid}: A={a[bid]['L2']} | B={b[bid]['L2']}")

    # Write markdown summary table
    lines = [
        "# Paper S3 — Diagnosis accuracy on 17 TrainAudit DETECTED cases",
        "",
        "| Tier | Definition | Rater A | Rater B | Final (both yes) | Cohen's kappa |",
        "|---|---|---:|---:|---:|---|",
        f"| L1 rule name -> fault class | Rule name + message correctly identifies "
        f"the fault class | {a_l1_hit}/{len(a_l1)} | {b_l1_hit}/{len(b_l1)} | "
        f"{final_l1}/{len(common)} | {fmt_kappa(kappa_l1)} |",
        f"| L2 RCA chain -> leaf root cause | RCA chain converges to the leaf "
        f"root cause | {a_l2_hit}/{len(a_l2_elig)} | {b_l2_hit}/{len(b_l2_elig)} | "
        f"{final_l2_yes}/{final_l2_n} | {fmt_kappa(kappa_l2)} |",
        "",
        "## Disagreement cases",
        "",
    ]
    if disagree:
        for bid in disagree:
            lines.append(
                f"- **{bid}** L1: A={a[bid]['L1']} vs B={b[bid]['L1']}"
            )
    else:
        lines.append("- None for L1")
    if l2_disagree:
        for bid in l2_disagree:
            lines.append(f"- **{bid}** L2: A={a[bid]['L2']} vs B={b[bid]['L2']}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
