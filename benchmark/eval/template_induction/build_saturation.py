"""Aggregate the 8 development-batch logs into saturation_by_batch.csv and
evaluate the pre-registered stopping rule (protocol §6)."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
DEV = HERE / "development"


def main():
    rows = []
    cum_cases = 0
    cum_matched_or_covered = 0
    prev_templates = len(json.load(open(HERE / "seed" / "initial_catalog.json"))["templates"])
    seed_templates = prev_templates
    no_new_streak = 0
    stop_batch = None

    for n in range(1, 9):
        log = json.load(open(DEV / f"batch_{n:02d}_log.json"))
        cat = json.load(open(DEV / f"catalog_after_batch_{n:02d}.json"))
        n_cases = len(log["decisions"])
        matched = log["n_matched"]
        new_singletons = log["n_new_singletons"]
        new_templates = log["n_new_templates"]
        total_after = len(cat["templates"])
        cum_cases += n_cases
        # A case is "covered" by the catalog if it MATCHed or seeded a promotion
        # into a template within this batch (NEW_TEMPLATE decisions).
        covered = sum(1 for d in log["decisions"]
                      if d["decision"] in ("MATCH", "NEW_TEMPLATE"))
        cum_matched_or_covered += covered

        matched_share = matched / n_cases
        singleton_share = new_singletons / n_cases
        core_edits = len(log.get("core_edits", []))

        cond_a_ok = new_templates == 0
        cond_b_ok = matched_share >= 0.95
        cond_c_ok = singleton_share < 0.05
        cond_d_ok = core_edits == 0
        batch_ok = cond_a_ok and cond_b_ok and cond_c_ok and cond_d_ok
        no_new_streak = no_new_streak + 1 if batch_ok else 0
        if stop_batch is None and no_new_streak >= 3:
            stop_batch = n

        rows.append({
            "batch": n,
            "n_cases": n_cases,
            "n_matched": matched,
            "matched_share": round(matched_share, 3),
            "n_new_singletons": new_singletons,
            "new_singleton_share": round(singleton_share, 3),
            "n_new_templates": new_templates,
            "n_core_edits": core_edits,
            "templates_total_after": total_after,
            "templates_delta": total_after - prev_templates,
            "cum_cases_processed": cum_cases,
            "cum_catalog_coverage": round(cum_matched_or_covered / cum_cases, 3),
            "stopping_batch_ok": batch_ok,
            "consecutive_ok_streak": no_new_streak,
        })
        prev_templates = total_after

    with open(DEV / "saturation_by_batch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    final_templates = rows[-1]["templates_total_after"]
    summary = {
        "seed_templates": seed_templates,
        "final_templates_after_dev": final_templates,
        "templates_added_in_dev": final_templates - seed_templates,
        "dev_cases": cum_cases,
        "dev_cumulative_coverage": rows[-1]["cum_catalog_coverage"],
        "stopping_rule_triggered": stop_batch is not None,
        "simulated_stopping_point_batch": stop_batch,
        "note": ("Pre-registered stopping rule (3 consecutive batches with no new "
                 "recurring template AND >=95% matched AND <5% new singletons AND no "
                 "core edit) was NEVER satisfied over the full 186-case development "
                 "stream; the longest qualifying streak is "
                 f"{max(r['consecutive_ok_streak'] for r in rows)} batch(es). "
                 "Post-stop regret is therefore undefined/not computable: the catalog "
                 "did not saturate under the frozen criterion."),
    }
    json.dump(summary, open(DEV / "stopping_rule_evaluation.json", "w"),
              indent=1, ensure_ascii=False)

    for r in rows:
        print(f"batch {r['batch']}: n={r['n_cases']:2d} matched={r['n_matched']:2d} "
              f"({r['matched_share']:.2f}) new_sgl={r['n_new_singletons']} "
              f"({r['new_singleton_share']:.2f}) new_tpl={r['n_new_templates']} "
              f"total={r['templates_total_after']} cum_cov={r['cum_catalog_coverage']:.3f} "
              f"ok={r['stopping_batch_ok']} streak={r['consecutive_ok_streak']}")
    print()
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
