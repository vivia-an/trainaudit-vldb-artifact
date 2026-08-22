#!/usr/bin/env python3
"""Test the invariant the lost S2 ablation asserted: removing the pi_topo guard
must have ZERO effect on a single-rank trace.

`run_ablation_s2.py` survives only as bytecode (see docs/O21_S2_ABLATION_RECOVERY.md),
and one of its self-checks was:

    single-rank traces had delta_no_topo != 0 (must be 0)

That is a property of the guard, not of their run, so it is testable here with the
shipped rule sets. A topology guard predicates on cross-rank structure; with one rank
there is no replica group to compare against, so dropping the guard cannot change any
verdict. If it does, the guard is entangled with something other than topology.

The 2-rank runs act as a positive control: there the guard CAN matter, so a nonzero
delta is expected and its absence would mean the guard is inert.

This does NOT reproduce the paper's 25.8/83.3 FP/1M -- that needs the authors'
ablation_s2_results.csv. It tests the mechanism those numbers rest on.

Usage:
  PYTHONPATH=core/trainaudit_pkg python3 benchmark/eval/guard_invariant_singlerank.py \
      --events DIR [--check]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ARMS = {"full": None, "no_topo": "rules_no_topo", "no_precond": "rules_no_precond"}


def violated_ids(db: Path, rules_dir, tier):
    from trainaudit.store import TraceStore
    from trainaudit.verifier import run_rules
    store = TraceStore(str(db))
    res = run_rules(store, tier, rules_dir=rules_dir)
    errs = [r.rule_id for r in res
            if r.message and str(r.message).startswith("rule error")]
    return {r.rule_id for r in res if r.violated}, len(res), errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, type=Path,
                    help="dir of run subdirs, each holding trace_rank*.duckdb")
    ap.add_argument("--pkg", type=Path,
                    default=Path("core/trainaudit_pkg/trainaudit"))
    ap.add_argument("--check", action="store_true",
                    help="exit nonzero if the invariant fails")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    from trainaudit.tiers import Tier
    tier = Tier.T4_INSTANCE  # every rule enabled, so the guard has max opportunity

    runs = sorted(d for d in a.events.iterdir() if d.is_dir())
    if not runs:
        print(f"no run dirs under {a.events}", file=sys.stderr)
        return 2

    rows, rule_errors = [], 0
    for d in runs:
        dbs = sorted(d.glob("*.duckdb"))
        if not dbs:
            continue
        arm_ids, n_eval = {}, 0
        for arm, sub in ARMS.items():
            rules_dir = (a.pkg / sub) if sub else None
            ids = set()
            for db in dbs:
                v, ne, errs = violated_ids(db, rules_dir, tier)
                ids |= v
                n_eval += ne
                rule_errors += len(errs)
            arm_ids[arm] = ids
        rows.append({
            "run": d.name,
            "n_ranks": len(dbs),
            "n_evaluated": n_eval,
            "full": sorted(arm_ids["full"]),
            "delta_no_topo": sorted(arm_ids["no_topo"] - arm_ids["full"]),
            "delta_no_precond": sorted(arm_ids["no_precond"] - arm_ids["full"]),
        })

    single = [r for r in rows if r["n_ranks"] == 1]
    multi = [r for r in rows if r["n_ranks"] > 1]
    bad = [r for r in single if r["delta_no_topo"]]

    w = max(len(r["run"]) for r in rows) + 2
    print(f"tier={tier.name}  runs={len(rows)}  rule errors={rule_errors}\n")
    print(f"{'run':{w}}{'ranks':>6}{'firing':>8}{'d_topo':>8}{'d_precond':>11}")
    for r in rows:
        print(f"{r['run']:{w}}{r['n_ranks']:>6}{len(r['full']):>8}"
              f"{len(r['delta_no_topo']):>8}{len(r['delta_no_precond']):>11}")

    print(f"\nINVARIANT  single-rank delta_no_topo == 0")
    print(f"  {len(single)} single-rank run(s): "
          f"{'all zero -- holds' if not bad else 'FAILED: ' + str([r['run'] for r in bad])}")
    n_ctl = sum(1 for r in multi if r["delta_no_topo"])
    print(f"CONTROL    {len(multi)} multi-rank run(s), {n_ctl} with a nonzero "
          f"delta_no_topo (guard is {'live' if n_ctl else 'INERT -- suspicious'})")
    dp = sum(1 for r in rows if r["delta_no_precond"])
    print(f"           delta_no_precond nonzero on {dp}/{len(rows)} run(s) "
          f"-- pi_precond is not topology-gated, so it may fire at any rank count")

    if a.json:
        a.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {a.json}")

    if a.check:
        if bad:
            print("\nFAIL: invariant violated", file=sys.stderr)
            return 1
        if rule_errors:
            print(f"\nFAIL: {rule_errors} rule error(s)", file=sys.stderr)
            return 1
        print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
