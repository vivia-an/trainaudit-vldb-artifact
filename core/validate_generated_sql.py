#!/usr/bin/env python3
"""Execute the recovered SQL against a trace and report what happens.

This is the check that matters for Sec 4.5's claim: an accepted rule is a SQL query whose
empty result means the constraint holds and whose non-empty result *is* the violation set.
Running the recovered queries against a clean trace should therefore return mostly empty
results, and against a fault-injected trace should return rows for the relevant rules.

    python3 core/validate_generated_sql.py --db <trace.db> [--limit N]
"""
import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
GEN = HERE / "config" / "generated_sql.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rules")
    ap.add_argument("--show-errors", type=int, default=3)
    args = ap.parse_args()

    try:
        import duckdb
    except ImportError:
        sys.exit("needs duckdb: pip install duckdb")

    rules = json.loads(GEN.read_text())["rules"]
    items = list(rules.items())[: args.limit or None]
    conn = duckdb.connect(args.db, read_only=True)

    stat = collections.Counter()
    errors, violating = [], []
    for name, r in items:
        if not r.get("variants"):
            stat["no_sql"] += 1
            continue
        sql = r["variants"][0]["sql"]
        try:
            rows = conn.execute(sql).fetchall()
        except Exception as exc:                                  # noqa: BLE001
            stat["error"] += 1
            errors.append((name, type(exc).__name__, str(exc).splitlines()[0][:150]))
            continue
        if rows:
            stat["non_empty"] += 1
            violating.append((name, len(rows)))
        else:
            stat["empty"] += 1

    total = sum(stat.values())
    print(f"{pathlib.Path(args.db).parent.parent.name}: {total} rules executed")
    print(f"  empty result (constraint holds)   {stat['empty']}")
    print(f"  non-empty (violation set)         {stat['non_empty']}")
    print(f"  SQL error                         {stat['error']}")
    if stat["error"]:
        print(f"  executes cleanly                  "
              f"{100 * (stat['empty'] + stat['non_empty']) / max(total, 1):.0f}%")
    for name, kind, msg in errors[: args.show_errors]:
        print(f"    ! {name[:56]}  {kind}: {msg}")
    for name, n in sorted(violating, key=lambda x: -x[1])[:5]:
        print(f"    * {name[:56]}  {n} violating rows")


if __name__ == "__main__":
    main()
