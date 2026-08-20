#!/usr/bin/env python3
"""Show the topology guard's skip-vs-check decision on real data.

§4.1's running example turns on one quantity: how many parameter records sit in a replica
group of size 1 — which $\\pi_{\\text{topo}}$ skips, because there is no replica relation to
check — versus size > 1, which it compares. The paper reports 57 skipped of 492 for its
SwitchMLP run. This reports the same split for every events-schema trace given.

    python3 benchmark/injection/audit_guard_groups.py --root <events-traces-dir>
"""
import argparse
import collections
import json
import pathlib
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    try:
        import duckdb
    except ImportError:
        sys.exit("needs duckdb: pip install duckdb")

    root = pathlib.Path(args.root)
    traces = sorted(root.glob("**/*.duckdb"))
    if not traces:
        sys.exit(f"no .duckdb traces under {root}")

    print(f"{'run / rank':<52}{'records':>8}{'skip':>7}{'check':>7}{'not equal':>11}")
    total = collections.Counter()
    exercised = 0
    for p in traces:
        try:
            c = duckdb.connect(str(p), read_only=True)
            rows = c.execute("SELECT payload FROM events WHERE hookpoint='build.snapshot'").fetchall()
            c.close()
        except Exception:
            continue
        g = collections.Counter()
        uneq = 0
        for (pl,) in rows:
            try:
                d = json.loads(pl)
            except (TypeError, json.JSONDecodeError):
                continue
            for key in ("cross_rank_cksums", "cross_rank_buffer_cksums"):
                for x in d.get(key) or []:
                    if not isinstance(x, dict):
                        continue
                    size = x.get("group_size")
                    g[size] += 1
                    if isinstance(size, int) and size > 1 and x.get("all_equal") is False:
                        uneq += 1
        if not g:
            continue
        skip = g.get(1, 0)
        check = sum(v for k, v in g.items() if isinstance(k, int) and k > 1)
        total["skip"] += skip
        total["check"] += check
        total["uneq"] += uneq
        exercised += check > 0
        # Truncate the run name, never the rank: eight ranks of one run must stay
        # distinguishable in the output.
        run, rank = p.parent.name, p.stem
        label = f"{run[:50 - len(rank) - 1]}/{rank}" if len(run) + len(rank) + 1 > 50 else f"{run}/{rank}"
        mark = "" if check else "   (single-rank: guard never checks)"
        print(f"{label:<52}{skip + check:>8}{skip:>7}{check:>7}{uneq:>11}{mark}")

    print(f"\n{exercised} of {len(traces)} traces contain a replica group larger than 1")
    print(f"totals: {total['skip']} skipped, {total['check']} checked, "
          f"{total['uneq']} checked-and-unequal")
    if total["check"]:
        print(f"\nWith the guard in place a clean run yields {total['uneq']} alarms from the "
              f"{total['check']} checked records.")
        print(f"Without it the {total['skip']} skipped records would each be compared across "
              f"ranks and differ by construction — that is the false-positive cost §4.1 quantifies.")
    if not exercised:
        print("\nNone of these traces exercises the guard's check branch: every record sits in a")
        print("group of size 1, so the skip path is taken throughout. A trace from a run with")
        print("TP>1 is needed to see the decision the paper describes.")


if __name__ == "__main__":
    main()
