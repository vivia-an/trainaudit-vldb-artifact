#!/usr/bin/env python3
"""Check whether each run's per-rank captures are genuinely distinct.

Hashing the `.db` file is not sound: DuckDB keeps uncommitted data in a `.db.wal` sidecar,
so every un-checkpointed database is an identical 12 KB header and looks like a duplicate
of every other. This compares content instead, order-independently, over every row.

    python3 benchmark/injection/audit_rank_captures.py --root /path/for/traces
"""
import argparse
import collections
import pathlib
import sys

# Hash the WHOLE payload. An earlier version projected onto
# (step, stage, name, cksum) and reported false matches: seven TP injection runs collided
# on that projection while differing in grad_cksum / requires_grad / shape, and so did two
# clean runs. Fault injections need not change a parameter checksum.
AGG = """SELECT md5(string_agg(s,'' ORDER BY s)) FROM (
  SELECT step||'|'||stage||'|'||CAST(data AS VARCHAR) s FROM coredump)"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    try:
        import duckdb
    except ImportError:
        sys.exit("needs duckdb: pip install duckdb")

    root = pathlib.Path(args.root)
    colls = sorted(p for p in root.glob("**/Collector") if p.is_dir())
    dup, wal_only, by_hash = [], [], collections.defaultdict(list)
    for coll in colls:
        run = coll.parent.relative_to(root).as_posix()
        hashes = {}
        for f in sorted(coll.glob("coredump_dp*.db")):
            if f.stat().st_size <= 16384 and not f.with_suffix(".db.wal").exists():
                wal_only.append(f"{run}/{f.name}")
            try:
                h = duckdb.connect(str(f), read_only=True).execute(AGG).fetchone()[0]
            except Exception:
                h = None
            if h:
                hashes[f.name] = h
                by_hash[h].append(f"{run}/{f.name}")
        if len(hashes) > 1 and len(set(hashes.values())) < len(hashes):
            dup.append((run, hashes))

    print(f"{len(colls)} runs inspected")
    print(f"  runs whose per-rank captures are identical : {len(dup)}")
    for run, h in dup:
        print(f"    {run}")
        for name, v in h.items():
            print(f"      {name:<32}{v[:12]}")
    cross = {h: v for h, v in by_hash.items()
             if len({p.rsplit('/', 2)[0] for p in v}) > 1}
    print(f"  contents shared across different runs     : {len(cross)}")
    for h, v in cross.items():
        runs = sorted({p.rsplit('/', 2)[0] for p in v})
        print(f"    {h[:12]}  {', '.join(runs)}")
    if wal_only:
        print(f"  per-rank files that are empty headers    : {len(wal_only)}")
        print("    (data lives in a .db.wal sidecar that was not fetched)")
        for p in wal_only[:8]:
            print(f"    {p}")


if __name__ == "__main__":
    main()
