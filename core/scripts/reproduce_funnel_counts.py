#!/usr/bin/env python3
"""Reproduce paper funnel stage sizes (420→5334→3436→357→45).

Modes:
  1) Sum frozen per-framework breakdown (structural recompute — default)
  2) Validate funnel_counts.csv against paper totals
  3) Recount skip-L3 stress (114/400) from checked-in CSV

Full LLM re-mining of 5334 candidates is intentionally out of band;
this script is the open, deterministic reproduce path claimed by the paper artifact.

CSV lookup order (no overleaf required for open package):
  core_algo/data/ → sdccheck/benchmark/eval/ → overleaf/.../benchmark/eval/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Supports: (1) core_algo/scripts  (2) packed root/scripts
_SCRIPT = Path(__file__).resolve()
_PKG = _SCRIPT.parents[1]  # core_algo/ or packed root/
ROOT = _PKG.parent
CORE_DATA = _PKG / "data"
BREAKDOWN = _PKG / "config" / "funnel_stage_breakdown.json"
CSV_CANDIDATES = [
    CORE_DATA / "funnel_counts.csv",
    _PKG / "core_algo" / "data" / "funnel_counts.csv",
    ROOT / "sdccheck" / "core_algo" / "data" / "funnel_counts.csv",
    ROOT / "overleaf/sdc_llm_icml_2025-v2-back-cc-new-branch/benchmark/eval/funnel_counts.csv",
]
SKIP_L3_CANDIDATES = [
    CORE_DATA / "funnel_skip_l3_results.csv",
    _PKG / "core_algo" / "data" / "funnel_skip_l3_results.csv",
    ROOT / "sdccheck" / "core_algo" / "data" / "funnel_skip_l3_results.csv",
    ROOT / "overleaf/sdc_llm_icml_2025-v2-back-cc-new-branch/benchmark/eval/funnel_skip_l3_results.csv",
]
PAPER = {"L1": 420, "L2": 5334, "L3": 3436, "L4": 357, "Deploy": 45}


def _sum_framework(layer: dict) -> int:
    return int(sum(int(v) for v in layer.values()))


def recompute_from_breakdown(path: Path) -> tuple:
    data = json.loads(path.read_text(encoding="utf-8"))
    pf = data["per_framework"]
    out = {k: _sum_framework(pf[k]) for k in ("L1", "L2", "L3", "L4")}
    out["Deploy"] = int(data["paper_totals"]["Deploy"])
    return out, data


def validate_csv() -> tuple:
    path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError("funnel_counts.csv not found (tried data/ first)")
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["layer"]] = int(r["n_candidates"])
    return rows, path


def recount_skip_l3(path: Path) -> tuple:
    fired = 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            if str(r.get("fired", "")).strip() in ("1", "true", "True"):
                fired += 1
    return fired, n


def resolve_skip_l3(meta: dict) -> Path:
    for p in SKIP_L3_CANDIDATES:
        if p.is_file():
            return p
    rel = meta.get("skip_stress", {}).get("skip_l3_csv", "")
    for base in (_PKG, ROOT):
        cand = base / rel
        if cand.is_file():
            return cand
    raise FileNotFoundError("funnel_skip_l3_results.csv not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-csv-check", action="store_true")
    args = ap.parse_args()

    ok = True
    print("=== 1) Structural recompute from funnel_stage_breakdown.json ===")
    recomputed, meta = recompute_from_breakdown(BREAKDOWN)
    for k, expect in PAPER.items():
        got = recomputed.get(k)
        status = "OK" if got == expect else "MISMATCH"
        if got != expect:
            ok = False
        print(f"  {k}: paper={expect} sum={got} [{status}]")

    if not args.skip_csv_check:
        print("=== 2) CSV artifact check ===")
        try:
            rows, csv_path = validate_csv()
            print(f"  artifact: {csv_path}")
            for k, expect in PAPER.items():
                got = rows.get(k)
                status = "OK" if got == expect else "MISMATCH"
                if got != expect:
                    ok = False
                print(f"  {k}: paper={expect} csv={got} [{status}]")
        except FileNotFoundError as e:
            ok = False
            print(f"  FAIL: {e}")

    print("=== 3) Skip-L3 stress recount ===")
    try:
        skip_path = resolve_skip_l3(meta)
        fired, n = recount_skip_l3(skip_path)
        expect_f = int(meta["skip_stress"]["skip_l3_clean_fp"])
        expect_n = int(meta["skip_stress"]["skip_l3_sample"])
        status = "OK" if (fired, n) == (expect_f, expect_n) else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"  artifact: {skip_path}")
        print(f"  skip_l3 fired/n = {fired}/{n} paper={expect_f}/{expect_n} [{status}]")
    except Exception as e:
        ok = False
        print(f"  FAIL skip_l3: {e}")

    exp = meta.get("enumeration_model", {}).get("mean_expansion_L2_over_L1")
    if exp and recomputed["L1"]:
        ratio = recomputed["L2"] / recomputed["L1"]
        print(f"=== 4) L2/L1 expansion = {ratio:.4f} (paper note ≈ {exp}) ===")

    if ok:
        print("PASS: funnel reproduce (breakdown + CSV + skip-L3)")
        return 0
    print("FAIL: funnel reproduce diverged", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
