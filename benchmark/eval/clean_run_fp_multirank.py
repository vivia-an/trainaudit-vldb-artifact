#!/usr/bin/env python3
"""Aggregate the authors' recorded MULTI-RANK clean-run rule results.

Background. Section 6.2's clean-run false-positive claim is backed by
`e_clean_baseline_summary.csv`, which lists four runs -- megatron_clean,
megatron_moe, olmo_core_baseline, olmo_core_moe_hybrid -- totalling 111,308
events and 0 fires. All four are SINGLE-RANK, and two of them only reach step 0.

The authors' own `e_status.md` says the multi-rank configurations that section
actually calls for (DP=8, TP=2/DP=4, TP=2/PP=2/DP=2, EP=2/DP=4, FSDP zero3) still
需要 GPU 补跑 -- need a GPU re-run -- and that every one of them is expected to
produce **0 FP**.

Those re-runs were subsequently done, at least in part: `_runs/` holds three
200-step multi-rank clean runs with 2.76M events and the authors' own
`rule_results.json` for each. They are absent from the summary CSV. This script
reads what those files record.

Every firing on a clean run is a false positive by construction, since these runs
carry no injected fault and the authors' stated expectation for them is 0 FP.

CAVEAT on the denominator. `n_violations` saturates at exactly 50 in every file
(verified: no value exceeds 50), while the messages report the true count -- e.g.
`n_violations=50` alongside "201 loss calls with non-'mean' reduction". Any rate
computed from `n_violations` therefore UNDERSTATES the real one, so this script
reports rules-fired counts as the primary figure and marks derived rates as lower
bounds.

Usage:
  python3 benchmark/eval/clean_run_fp_multirank.py [--runs DIR] [--check]
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "clean_run_fp" / "_runs"
CAP = 50


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DEFAULT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    runs = []
    for d in sorted(p for p in a.runs.iterdir() if p.is_dir()):
        f = d / "rule_results.json"
        if f.exists():
            runs.append((d.name, json.loads(f.read_text())))
    if not runs:
        print(f"no rule_results.json under {a.runs}", file=sys.stderr)
        return 2

    print(f"{'run':<20}{'world':>6}{'dp':>6}{'steps':>7}{'rules':>7}{'fired':>7}")
    fires = Counter()
    capped = 0
    for name, j in runs:
        res = j["results"]
        fired = [r for r in res if r["violated"]]
        capped += sum(1 for r in res if r["n_violations"] == CAP)
        for r in fired:
            fires[r["rule_id"]] += 1
        print(f"{name:<20}{j['world_size']:>6}{str(j['dp_type']):>6}"
              f"{str(j['steps']):>7}{len(res):>7}{len(fired):>7}")

    multi = [(n, j) for n, j in runs if j["world_size"] > 1 and j["steps"] >= 200]
    print(f"\n{len(runs)} recorded run(s); {len(multi)} are multi-rank at >=200 steps")

    print("\nrules firing on clean runs (run count / total):")
    for rid, n in fires.most_common():
        flag = "  <-- every run" if n == len(runs) else ""
        print(f"  {n}/{len(runs)}  {rid}{flag}")

    universal = [r for r, n in fires.items() if n == len(runs)]
    print(f"\n{len(universal)} rule(s) fire on EVERY clean run -- a systematic "
          f"signature, not sampling noise.")
    print(f"{capped} rule result(s) sit exactly at the n_violations cap of {CAP}, "
          f"so recorded counts are lower bounds.")

    print("\nAgainst e_clean_baseline_summary.csv, which records 0 fires over "
          "111,308 events on 4 single-rank runs:")
    counts = [sum(1 for r in j["results"] if r["violated"]) for _, j in runs]
    print(f"  the runs here fire {min(counts)}-{max(counts)} rules each, "
          "and none of them appear in that CSV.")

    if a.check:
        # Regression guard: these are recorded facts and must not drift.
        if len(runs) != 4 or not universal:
            print("\nFAIL: unexpected shape", file=sys.stderr)
            return 1
        print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
