#!/usr/bin/env python3
"""Run the three guard arms over MERGED multi-rank clean traces.

Why merged. The replica-checksum rules that the pi_topo guard protects compare ranks
within a replica group, so they can only do anything if every rank of a run is in ONE
store. The traces are written one DuckDB file per rank, so running them file-by-file
makes those rules structurally unable to fire and any pi_topo ablation trivially empty.
This script merges a run's ranks into a single in-memory store first, offsetting
event_id per rank to keep the primary key unique.

Three arms, from the shipped rule sets:
  full        core/trainaudit_pkg/trainaudit/rules
  no_topo     ...rules_no_topo      -- 4 guard sites: `if group_size <= 1: continue`
                                       in the 3 replica-cksum rules, plus dropping
                                       `and declared > 1` in T1-process-group-size
  no_precond  ...rules_no_precond

The traces themselves are 2.6 GB and are not shipped; `guard_arms_results.json` records
what they produced, and `--check` regression-guards that record.

Usage:
  PYTHONPATH=core/trainaudit_pkg python3 benchmark/eval/guard_arms_multirank.py \
      --runs DIR --out benchmark/eval/clean_run_fp/guard_arms_results.json
  python3 benchmark/eval/guard_arms_multirank.py --check
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path

PKG = Path("core/trainaudit_pkg/trainaudit")
ARMS = [("full", None), ("no_topo", PKG / "rules_no_topo"),
        ("no_precond", PKG / "rules_no_precond")]
DEFAULT_OUT = (Path(__file__).resolve().parent / "clean_run_fp"
               / "guard_arms_results.json")


def merged_store(run_dir: Path):
    from trainaudit.store import TraceStore
    store = TraceStore(":memory:")
    dbs = sorted(glob.glob(f"{run_dir}/*.duckdb"))
    for i, p in enumerate(dbs):
        store.conn.execute(f"ATTACH '{p}' AS r{i} (READ_ONLY)")
        store.conn.execute(
            f"INSERT INTO events SELECT event_id+{i * 10**9}, step, rank, hookpoint, "
            f"ts_ns, schema_version, payload FROM r{i}.events")
    n, nr = store.conn.execute(
        "select count(*), count(distinct rank) from events").fetchone()
    return store, len(dbs), n, nr


def measure(run_dir: Path) -> dict:
    from trainaudit.verifier import run_rules
    from trainaudit.tiers import Tier
    store, n_dbs, n_events, n_ranks = merged_store(run_dir)
    arms = {}
    for arm, rd in ARMS:
        res = run_rules(store, Tier.T4_INSTANCE, rules_dir=rd)
        arms[arm] = {r.rule_id: str(r.message) for r in res if r.violated}
        errs = [r.rule_id for r in res
                if r.message and str(r.message).startswith("rule error")]
        if errs:
            print(f"  !! rule errors in {arm}: {errs}", file=sys.stderr)
    return {"run": run_dir.name, "n_dbs": n_dbs, "n_events": n_events,
            "n_ranks": n_ranks, "arms": arms}


def report(rows) -> int:
    bad = 0
    for r in rows:
        if r["n_events"] == 0:
            print(f"\n{r['run']}  wrote no events -- skipped")
            continue
        full = set(r["arms"]["full"])
        d_topo = sorted(set(r["arms"]["no_topo"]) - full)
        d_pre = sorted(set(r["arms"]["no_precond"]) - full)
        print(f"\n{r['run']}  {r['n_ranks']} ranks, {r['n_events']} events")
        print(f"  full fires {len(full)}: {sorted(full)}")
        print(f"  delta_no_topo    ({len(d_topo)}): {d_topo}")
        print(f"  delta_no_precond ({len(d_pre)}): {d_pre}")

        # The rank-blind monotonicity rule: its SQL orders by event_id with no rank
        # partition, so a merged run drops back to step 1 at each rank boundary.
        msg = r["arms"]["full"].get("T0-optim-step-counter-monotonic")
        if msg:
            n = int(msg.split("in ")[1].split("/")[0])
            ok = n == r["n_ranks"] - 1
            print(f"  rank-blind monotonicity: {n} bad transitions, "
                  f"n_ranks-1 = {r['n_ranks']-1} -> {'matches' if ok else 'MISMATCH'}")
            if not ok:
                bad += 1
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, help="dir of multi-rank run subdirs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="validate the recorded results instead of measuring")
    a = ap.parse_args(argv)

    if a.check and not a.runs:
        if not a.out.exists():
            print(f"missing {a.out}", file=sys.stderr)
            return 2
        rows = json.loads(a.out.read_text())
        bad = report(rows)
        # Recorded invariants: pi_topo inert on every clean run measured, and the
        # rank-blind rule firing exactly n_ranks-1 times.
        for r in rows:
            if r["n_events"] == 0:
                continue  # ep2_dp4_moe_r1 wrote no events; nothing to assert
            if set(r["arms"]["no_topo"]) - set(r["arms"]["full"]):
                print(f"\nFAIL: {r['run']} has a nonzero delta_no_topo", file=sys.stderr)
                return 1
            if not set(r["arms"]["no_precond"]) - set(r["arms"]["full"]):
                print(f"\nFAIL: {r['run']} shows no precond effect", file=sys.stderr)
                return 1
        if bad:
            print("\nFAIL: rank-boundary count mismatch", file=sys.stderr)
            return 1
        print("\nOK")
        return 0

    if not a.runs:
        ap.error("--runs is required unless --check")
    dirs = [d for d in sorted(a.runs.iterdir())
            if d.is_dir() and glob.glob(f"{d}/*.duckdb")]
    rows = []
    for d in dirs:
        print(f"measuring {d.name} ...", flush=True)
        rows.append(measure(d))
    report(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
