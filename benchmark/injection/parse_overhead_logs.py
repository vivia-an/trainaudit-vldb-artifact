#!/usr/bin/env python3
"""Recompute Table `tab:overhead` (§5.5) from the raw Megatron-LM training logs.

Each log line carries Megatron's own per-iteration timing:

    iteration 5/8 | ... | elapsed time per iteration (ms): 690.5 | ...

Iteration 1 is discarded in every configuration: it absorbs CUDA context creation,
cuDNN autotuning, and the collector's first DuckDB write, and runs 3-9x the steady
state (e.g. 6383 ms vs 732 ms at baseline). The reported figure is the mean over the
remaining iterations, which is what the paper quotes.

Usage:
    python3 parse_overhead_logs.py                 # print the table, write overhead_h20.csv
    python3 parse_overhead_logs.py --check         # also assert the paper's values
"""
import argparse
import csv
import pathlib
import re
import statistics
import sys

STEP_RE = re.compile(r"elapsed time per iteration \(ms\): ([0-9.]+)")
HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "overhead_raw"

# The paper reports the 2026-06-30 session. 2026-07-12 is an independent replication
# on the same H20 harness after the fast path became the default.
SESSIONS = {
    "2026-06-30": [
        ("baseline (collector off)", "overhead_baseline.log", "baseline"),
        ("naive: SHA-256 + host stats, full dump", "overhead_collect.log", "reported"),
        ("+ GPU fingerprint (checksum)", "overhead_fast_collect.log", "reported"),
        ("+ fused GPU statistics", "overhead_fast2_collect.log", "reported"),
        ("rejected variant: async hook worker", "overhead_async_collect.log", "rejected"),
        ("rejected variant: per-row batched insert", "overhead_batch_collect.log", "rejected"),
    ],
    "2026-07-12": [
        ("baseline (collector off)", "overhead_baseline.log", "baseline"),
        ("naive: SHA-256 + host stats, full dump", "overhead_full_collect.log", "replication"),
        ("optimised (fingerprint + fused stats)", "overhead_fast_collect.log", "replication"),
    ],
}
# Table tab:overhead, main.tex around line 1009.
PAPER = {"baseline_ms": 732.0, "naive_s": 192.0, "fingerprint_s": 27.5, "fused_s": 25.0,
         "naive_x": 262, "fingerprint_x": 38, "fused_x": 34, "speedup": 7.7}


def steps(path):
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    return [float(m) for m in STEP_RE.findall(text)]


def summarise(path):
    vals = steps(path)
    if len(vals) < 2:
        return None
    warm, steady = vals[0], vals[1:]
    return {
        "n_iters": len(vals),
        "warmup_ms": warm,
        "mean_ms": statistics.mean(steady),
        "median_ms": statistics.median(steady),
        "min_ms": min(steady),
        "max_ms": max(steady),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="assert the values printed in the paper")
    args = ap.parse_args()

    rows = []
    for session, configs in SESSIONS.items():
        sdir = RAW / f"session_{session.replace('-', '')}"
        base = None
        for label, fname, kind in configs:
            s = summarise(sdir / fname)
            if s is None:
                print(f"  !! missing or unparseable: {sdir.name}/{fname}", file=sys.stderr)
                continue
            if kind == "baseline":
                base = s["mean_ms"]
            rows.append({
                "session": session, "configuration": label, "role": kind,
                "log": f"{sdir.name}/{fname}",
                "n_iters": s["n_iters"],
                "warmup_step_ms": round(s["warmup_ms"], 1),
                "steady_mean_ms": round(s["mean_ms"], 1),
                "steady_median_ms": round(s["median_ms"], 1),
                "steady_min_ms": round(s["min_ms"], 1),
                "steady_max_ms": round(s["max_ms"], 1),
                "vs_baseline_x": round(s["mean_ms"] / base, 1) if base else "",
            })

    out = HERE / "overhead_h20.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    width = max(len(r["configuration"]) for r in rows)
    for session in SESSIONS:
        print(f"\n1.2B / H20, DP=2, bf16 — session {session}")
        print(f"  {'configuration':<{width}}  {'dump/step':>10}  {'vs base':>8}")
        for r in (x for x in rows if x["session"] == session):
            ms = r["steady_mean_ms"]
            shown = f"{ms:.0f} ms" if ms < 2000 else f"{ms/1000:.1f} s"
            print(f"  {r['configuration']:<{width}}  {shown:>10}  {str(r['vs_baseline_x']) + 'x':>8}")
    print(f"\nwrote {out.relative_to(HERE.parent.parent)}")

    if args.check:
        by = {r["configuration"]: r for r in rows if r["session"] == "2026-06-30"}
        got = {
            "baseline_ms": by["baseline (collector off)"]["steady_mean_ms"],
            "naive_s": by["naive: SHA-256 + host stats, full dump"]["steady_mean_ms"] / 1000,
            "fingerprint_s": by["+ GPU fingerprint (checksum)"]["steady_mean_ms"] / 1000,
            "fused_s": by["+ fused GPU statistics"]["steady_mean_ms"] / 1000,
        }
        got["naive_x"] = got["naive_s"] * 1000 / got["baseline_ms"]
        got["fingerprint_x"] = got["fingerprint_s"] * 1000 / got["baseline_ms"]
        got["fused_x"] = got["fused_s"] * 1000 / got["baseline_ms"]
        got["speedup"] = got["naive_s"] / got["fused_s"]
        print("\ncheck against the paper (tolerance 2%):")
        bad = 0
        for k, want in PAPER.items():
            have = got[k]
            ok = abs(have - want) <= 0.02 * want
            bad += not ok
            print(f"  {k:<15} paper {want:>8}   measured {have:>9.2f}   {'ok' if ok else 'MISMATCH'}")
        if bad:
            sys.exit(f"{bad} value(s) do not match the paper")
        print("  all values reproduce")


if __name__ == "__main__":
    main()
