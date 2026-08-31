"""Aggregate three-way results into a single summary CSV + JSON for paper §6 table."""
import csv
import json
from pathlib import Path

NEW_DIR = Path(__file__).resolve().parent
RESULTS_DIR = NEW_DIR / "results"

D1_PRIME_ORDER = [
    "B1","B2","B3","B8","B11","B12",
    "M-012","M-020","M-024",
    "O-005","O-NEW-1","O-NEW-9",
    "OC-NEW-2","OC-NEW-3",
    "CF1","CM1","OF1",
]


def load(tool):
    f = RESULTS_DIR / f"{tool}_d1prime.json"
    if not f.exists(): return {}
    data = json.loads(f.read_text())
    return {r["bug_id"]: r for r in data.get("results", [])}


def main():
    ta = load("trainaudit")
    tc = load("traincheck")
    nv = load("naive")

    csv_path = NEW_DIR / "d1prime_summary.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["bug_id","class","framework","tier",
                    "ta_buggy","ta_fp","tc_buggy","tc_fp","naive_buggy","naive_fp"])
        for bid in D1_PRIME_ORDER:
            ta_r = ta.get(bid, {})
            tc_r = tc.get(bid, {})
            nv_r = nv.get(bid, {})
            cls = ta_r.get("category") or tc_r.get("category") or nv_r.get("category", "?")
            fw  = ta_r.get("framework") or tc_r.get("framework") or nv_r.get("framework", "?")
            tier = ta_r.get("tier") or tc_r.get("tier") or nv_r.get("tier", "?")
            def yn(r, k): return "✓" if r.get(k) else ("✗" if k in r else "-")
            w.writerow([
                bid, cls, fw, tier,
                yn(ta_r, "buggy_detected"), yn(ta_r, "fixed_fp"),
                yn(tc_r, "buggy_detected"), yn(tc_r, "fixed_fp"),
                yn(nv_r, "buggy_detected"), yn(nv_r, "fixed_fp"),
            ])

    # Aggregate metrics
    def detected(d): return sum(1 for r in d.values() if r.get("buggy_detected"))
    def fp(d): return sum(1 for r in d.values() if r.get("fixed_fp"))
    agg = {
        "n_bugs": 17,
        "trainaudit": {"buggy_detection": f"{detected(ta)}/17", "fixed_fp": f"{fp(ta)}/17"},
        "traincheck":  {"buggy_detection": f"{detected(tc)}/17", "fixed_fp": f"{fp(tc)}/17"},
        "naive":       {"buggy_detection": f"{detected(nv)}/17", "fixed_fp": f"{fp(nv)}/17"},
        "class_coverage": "12/13 (loss_computation reserved as §3.3 16.8% boundary)",
    }
    (NEW_DIR / "d1prime_aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"Wrote {csv_path}\nWrote {NEW_DIR}/d1prime_aggregate.json")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
