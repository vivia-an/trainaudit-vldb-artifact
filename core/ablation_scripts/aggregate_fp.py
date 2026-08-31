"""
Aggregate per-(lib, db) report JSONs (written by `python -m sdccheck --reports-out ...`)
into a single ablation_results.csv for the leave-one-out FP-decomposition table.

Convention: a `failed` constraint on a "fixed-commit" or "clean replay" database
is a False Positive. Categories from the JSON top-level keys (data_parallel,
tensor_parallel, ...) are mapped to paper Patterns P1-P8 via PATTERN_MAP.

Usage:
    python scripts/ablation/aggregate_fp.py \
        --reports-dir logs/ablation \
        --library-json config/lib_full.json \
        --csv-out logs/ablation/ablation_results.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Map JSON constraint categories -> paper pattern label.
# This will need adjustment once the paper's P1-P8 names are finalised.
PATTERN_MAP = {
    "data_parallel":          "P1_DP_consistency",
    "tensor_parallel":        "P3_cross_rank_replication",  # P3-style cross-rank checks
    "pipeline_parallel":      "P2_PP_stage_handoff",
    "zero_optimization":      "P5_optimizer_state_sharding",
    "model_integrity":        "P4_model_integrity",
    "training_progress":      "P7_training_progress",
    "mixed_precision":        "P6_dtype_consistency",
    "sequence_parallel":      "P3_cross_rank_replication",
    "distributed_optimizer":  "P8_distributed_optimizer",
}


def parse_filename(fname):
    """logs/ablation/<lib_name>__<db_base>.json -> (lib_name, db_base)"""
    base = os.path.basename(fname).rsplit(".json", 1)[0]
    if "__" not in base:
        return None, None
    lib, db = base.split("__", 1)
    return lib, db


def constraint_to_category(constraint_name, library_index):
    """Look up which top-level JSON category a constraint name belongs to."""
    return library_index.get(constraint_name, "uncategorized")


def build_library_index(library_json):
    """name -> category dict from the source library JSON."""
    idx = {}
    if not library_json or not os.path.isfile(library_json):
        return idx
    d = json.load(open(library_json, encoding="utf-8"))
    for cat, cdict in d.get("constraints", {}).items():
        if not isinstance(cdict, dict):
            continue
        for cname, cinfo in cdict.items():
            display_name = cinfo.get("name", cname)
            idx[display_name] = cat
            idx[cname] = cat
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default="logs/ablation")
    ap.add_argument("--library-json", default="config/lib_full.json",
                    help="Source-of-truth JSON for constraint -> category mapping")
    ap.add_argument("--csv-out", default="logs/ablation/ablation_results.csv")
    args = ap.parse_args()

    lib_idx = build_library_index(args.library_json)
    print(f"[index] loaded {len(lib_idx)} constraint->category mappings")

    files = sorted(glob.glob(os.path.join(args.reports_dir, "*.json")))
    files = [f for f in files
             if not os.path.basename(f).startswith("ablation_results")
             and not os.path.basename(f).startswith("smoke_")]
    print(f"[scan ] found {len(files)} report JSONs")

    rows = []
    by_cell = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0, "error": 0,
                                   "by_category": defaultdict(int),
                                   "by_pattern": defaultdict(int)})

    for f in files:
        lib, db = parse_filename(f)
        if lib is None:
            continue
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"[warn ] cannot parse {f}: {e}")
            continue
        cell = by_cell[(lib, db)]
        for r in data:
            cell["total"] += 1
            raw = (r.get("status") or "").lower()
            # Normalize "ResultStatus.PASS" / "pass" / "PASS" -> "pass"
            status = raw.split(".")[-1] if raw else ""
            if status not in ("pass", "fail", "error"):
                status = "error"
            cell[status] += 1
            cat = constraint_to_category(r.get("constraint_name", ""), lib_idx)
            if status == "fail":
                cell["by_category"][cat] += 1
                pat = PATTERN_MAP.get(cat, f"_unmapped:{cat}")
                cell["by_pattern"][pat] += 1

    pattern_keys = sorted({k for c in by_cell.values() for k in c["by_pattern"]})

    fieldnames = ["lib", "db", "total", "pass", "fail_FP", "error"] + \
                 [f"FP::{k}" for k in pattern_keys]

    with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for (lib, db), c in sorted(by_cell.items()):
            row = {
                "lib": lib, "db": db,
                "total": c["total"], "pass": c["pass"],
                "fail_FP": c["fail"], "error": c["error"],
            }
            for k in pattern_keys:
                row[f"FP::{k}"] = c["by_pattern"].get(k, 0)
            w.writerow(row)

    print(f"[csv  ] wrote {args.csv_out}")
    print()

    # ASCII summary table
    print("=== Leave-one-out summary (rows = lib, columns = db, values = total FP) ===")
    libs = sorted({k[0] for k in by_cell.keys()})
    dbs = sorted({k[1] for k in by_cell.keys()})
    head = f"{'lib':<20}" + "".join(f"{d[:18]:>20}" for d in dbs)
    print(head)
    for lib in libs:
        row = f"{lib:<20}"
        for db in dbs:
            c = by_cell.get((lib, db))
            if c is None:
                row += f"{'-':>20}"
            else:
                row += f"{c['fail']:>20}"
        print(row)


if __name__ == "__main__":
    main()
