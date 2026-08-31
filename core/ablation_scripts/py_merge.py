"""Python replacement for tools/merge.sh — merges per-rank coredump_dp*_tp*_pp*_cp*.db
into a single merged_coredump.db using the duckdb python module.

Usage:
    python py_merge.py <collector_dir> [--out merged_coredump.db]
"""

import argparse
import glob
import os
import re
import sys

import duckdb


PER_RANK_RE = re.compile(r"^coredump_dp\d+_tp\d+_pp\d+_cp\d+\.db$")


def merge_collector(collector_dir, out_name="merged_coredump.db"):
    files = sorted(
        f for f in os.listdir(collector_dir)
        if PER_RANK_RE.match(f)
    )
    if not files:
        print(f"  [skip] no per-rank dbs in {collector_dir}", file=sys.stderr)
        return False

    out_path = os.path.join(collector_dir, out_name)
    if os.path.exists(out_path):
        os.remove(out_path)
    wal = out_path + ".wal"
    if os.path.exists(wal):
        os.remove(wal)

    con = duckdb.connect(out_path)
    try:
        attach_cmds = []
        unions = []
        for i, f in enumerate(files):
            full = os.path.join(collector_dir, f)
            alias = f"db{i}"
            attach_cmds.append(f"ATTACH '{full}' AS {alias} (READ_ONLY);")
            unions.append(f"SELECT * FROM {alias}.coredump")
        for cmd in attach_cmds:
            con.execute(cmd)
        union_sql = " UNION ALL ".join(unions)
        con.execute(f"CREATE TABLE coredump AS {union_sql}")
        n = con.execute("SELECT COUNT(*) FROM coredump").fetchone()[0]
        print(f"  [ok] {collector_dir}/{out_name}: {n} rows from {len(files)} ranks")
        return True
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("collector_dirs", nargs="+",
                    help="One or more Collector/ directories containing per-rank dbs")
    ap.add_argument("--out", default="merged_coredump.db")
    args = ap.parse_args()

    ok = 0
    fail = 0
    for d in args.collector_dirs:
        try:
            if merge_collector(d, args.out):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [err] {d}: {e}", file=sys.stderr)
            fail += 1

    print(f"\n=== merged ok={ok} fail={fail} ===")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
