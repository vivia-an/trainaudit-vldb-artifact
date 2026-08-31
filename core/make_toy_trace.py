#!/usr/bin/env python3
"""Build a tiny multi-rank coredump DuckDB for offline tests.

Schema: coredump(step INT, stage VARCHAR, data JSON).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DEFAULT = Path(__file__).resolve().parent / "data" / "toy_merged.db"


def _row(step: int, stage: str, **fields) -> tuple:
    return (step, stage, json.dumps(fields))


def build(path: Path, *, buggy: bool = False) -> Path:
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE coredump (step INTEGER, stage VARCHAR, data JSON)"
    )
    rows = [
        _row(
            1,
            "model-after-optimizer-step",
            name="module.layers.0.layernorm.weight",
            cksum="aaa111",
            tp=0,
            dp=0,
            pp=0,
            type="torch.cuda.BFloat16Tensor",
        ),
        _row(
            1,
            "model-after-optimizer-step",
            name="module.layers.0.layernorm.weight",
            cksum="aaa111",
            tp=1,
            dp=0,
            pp=0,
            type="torch.cuda.BFloat16Tensor",
        ),
        _row(
            1,
            "model-after-optimizer-step",
            name="module.layers.0.router.weight",
            cksum="bbb222",
            tp=0,
            dp=0,
            pp=0,
            type="torch.cuda.BFloat16Tensor",
        ),
        _row(
            1,
            "model-after-optimizer-step",
            name="module.layers.0.router.weight",
            cksum=("bbb222" if not buggy else "EVIL999"),
            tp=1,
            dp=0,
            pp=0,
            type="torch.cuda.BFloat16Tensor",
        ),
        _row(
            1,
            "model-after-optimizer-step",
            name="module.layers.0.self_attention.linear_qkv.weight",
            cksum="qkv_tp0",
            tp=0,
            dp=0,
            pp=0,
            type="torch.cuda.BFloat16Tensor",
        ),
        _row(
            1,
            "model-after-optimizer-step",
            name="module.layers.0.self_attention.linear_qkv.weight",
            cksum="qkv_tp1",
            tp=1,
            dp=0,
            pp=0,
            type="torch.cuda.BFloat16Tensor",
        ),
    ]
    con.executemany("INSERT INTO coredump VALUES (?, ?, ?)", rows)
    con.close()
    print("wrote %s buggy=%s rows=%d" % (path, buggy, len(rows)))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--buggy", action="store_true")
    args = ap.parse_args()
    build(args.out, buggy=args.buggy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
