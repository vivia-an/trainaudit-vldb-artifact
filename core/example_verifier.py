#!/usr/bin/env python3
"""Example SQL verifier for T01-style guarded replica equality.

A violation is a non-empty query result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

T01_SQL = """
WITH t AS (
  SELECT step,
    json_extract_string(data, '$.name') AS name,
    json_extract_string(data, '$.cksum') AS cksum
  FROM coredump
  WHERE stage = 'model-after-optimizer-step'
    AND (
      lower(json_extract_string(data, '$.name')) LIKE '%layernorm%'
      OR lower(json_extract_string(data, '$.name')) LIKE '%layer_norm%'
      OR lower(json_extract_string(data, '$.name')) LIKE '%router%'
    )
)
SELECT name, step, COUNT(DISTINCT cksum) AS n_cksum
FROM t
GROUP BY name, step
HAVING COUNT(DISTINCT cksum) > 1
""".strip()

DEFAULT_DB = Path(__file__).resolve().parent / "data" / "toy_merged.db"


def verify(db_path: Path, sql: str = T01_SQL) -> list:
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--expect-clean", action="store_true")
    ap.add_argument("--expect-fire", action="store_true")
    args = ap.parse_args()
    if not args.db.is_file():
        print("missing db %s; run: python3 make_toy_trace.py" % args.db, file=sys.stderr)
        return 2
    rows = verify(args.db)
    print("db=%s violations=%d" % (args.db, len(rows)))
    for r in rows[:10]:
        print(" ", r)
    if args.expect_clean and rows:
        return 1
    if args.expect_fire and not rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
