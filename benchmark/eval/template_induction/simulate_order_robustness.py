"""Order-robustness of the saturation curves (protocol §6).

Replays the FINAL coding under many random orderings of the development
stream. Labels come from the FROZEN catalog (not the live batch decisions), so
that a case which was a singleton when first seen but was later promoted into a
template still counts as support for that template -- protocol §6 prescribes
simulating orders "利用最终人工编码结果", i.e. under the final coding.
  - case in some template's positive_examples -> supports that template
  - case in the catalog's singletons list     -> supports no template
Under a simulated order, a template is "discovered" at the position where its
2nd independent supporting case appears (protocol §4.1 admission rule), seeded
by the 27 templates already present from the seed stage. A case counts as
"covered" if its supporting template has been discovered at or before its own
position -- i.e. coverage as it would have been observed live under that order.

Outputs: analysis/saturation_orders.csv (median + 5-95% band per position)
         analysis/saturation_curve.pdf
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DEV = HERE / "development"
N_ORDERS = 200
SEED_TEMPLATES = 27


def load_labels():
    """dev case_id -> final template_id, or None if it stayed a singleton."""
    dev_cases = set(json.load(open(HERE / "dataset_split.json"))["development"])
    cat = json.load(open(HERE / "frozen_template_catalog.json"))
    labels = {c: None for c in dev_cases}
    for t in cat["templates"]:
        for cid in t["positive_examples"]:
            if cid in dev_cases:
                labels[cid] = t["template_id"]
    return labels


def simulate(order, labels, seed_template_ids, seed_support):
    """Returns (templates_curve, coverage_curve) over positions 1..len(order).

    seed_support pre-loads each template's supporting cases that came from the
    seed stage, so a template promoted from a seed singleton plus one dev case
    is discovered on that dev case (support reaches 2), matching the live run.
    """
    support = defaultdict(int, seed_support)
    discovered = set(seed_template_ids)
    covered = 0
    t_curve, c_curve = [], []
    for i, cid in enumerate(order, 1):
        tid = labels[cid]
        if tid is not None:
            support[tid] += 1
            if tid not in discovered and support[tid] >= 2:
                discovered.add(tid)
            if tid in discovered:
                covered += 1
        t_curve.append(len(discovered))
        c_curve.append(covered / i)
    return t_curve, c_curve


def main():
    labels = load_labels()
    seed_cat = json.load(open(HERE / "seed" / "initial_catalog.json"))
    seed_ids = {t["template_id"] for t in seed_cat["templates"]}
    assert len(seed_ids) == SEED_TEMPLATES
    seed_case_ids = set(json.load(open(HERE / "dataset_split.json"))["seed"])
    frozen = json.load(open(HERE / "frozen_template_catalog.json"))
    seed_support = {t["template_id"]: sum(1 for c in t["positive_examples"]
                                          if c in seed_case_ids)
                    for t in frozen["templates"]}
    cases = sorted(labels)
    rng = random.Random(20260716)

    t_runs, c_runs = [], []
    for _ in range(N_ORDERS):
        order = cases[:]
        rng.shuffle(order)
        t, c = simulate(order, labels, seed_ids, seed_support)
        t_runs.append(t)
        c_runs.append(c)

    n = len(cases)

    def band(runs, i):
        vals = sorted(r[i] for r in runs)
        return (vals[len(vals) // 2], vals[int(0.05 * len(vals))],
                vals[min(int(0.95 * len(vals)), len(vals) - 1)])

    rows = []
    for i in range(n):
        tm, tlo, thi = band(t_runs, i)
        cm, clo, chi = band(c_runs, i)
        rows.append({"cases_processed": i + 1,
                     "templates_median": tm, "templates_p05": tlo, "templates_p95": thi,
                     "coverage_median": round(cm, 4), "coverage_p05": round(clo, 4),
                     "coverage_p95": round(chi, 4)})
    with open(HERE / "analysis" / "saturation_orders.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    x = [r["cases_processed"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55))
    ax = axes[0]
    ax.fill_between(x, [r["templates_p05"] for r in rows], [r["templates_p95"] for r in rows],
                    alpha=0.25, color="#4C72B0", linewidth=0)
    ax.plot(x, [r["templates_median"] for r in rows], color="#1F3F73", lw=1.5)
    ax.axhline(SEED_TEMPLATES, ls="--", lw=1, color="gray")
    ax.text(2, SEED_TEMPLATES + 0.2, f"seed={SEED_TEMPLATES}", fontsize=5.5, color="gray")
    ax.set_ylabel("# templates", fontsize=7)
    ax.set_title("(a) Catalog growth", fontsize=7.5)

    ax = axes[1]
    ax.fill_between(x, [r["coverage_p05"] for r in rows], [r["coverage_p95"] for r in rows],
                    alpha=0.25, color="#C44E52", linewidth=0)
    ax.plot(x, [r["coverage_median"] for r in rows], color="#8C2E33", lw=1.5)
    ax.set_ylim(0.6, 1)
    ax.set_ylabel("coverage", fontsize=7)
    ax.set_title("(b) Coverage", fontsize=7.5)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.25, lw=0.5)
        a.set_xlabel("dev cases", fontsize=7)
        a.tick_params(labelsize=6)
        a.set_xticks([0, 100, 186])
    fig.tight_layout(pad=0.25, w_pad=0.5)
    fig.savefig(HERE / "analysis" / "saturation_curve.pdf")

    print(f"orders={N_ORDERS} dev_cases={n}")
    print(f"final templates: median={rows[-1]['templates_median']} "
          f"[{rows[-1]['templates_p05']},{rows[-1]['templates_p95']}]")
    print(f"final coverage: median={rows[-1]['coverage_median']} "
          f"[{rows[-1]['coverage_p05']},{rows[-1]['coverage_p95']}]")
    half = rows[n // 2]
    print(f"at 50% of stream: templates median={half['templates_median']}, "
          f"coverage median={half['coverage_median']}")


if __name__ == "__main__":
    main()
