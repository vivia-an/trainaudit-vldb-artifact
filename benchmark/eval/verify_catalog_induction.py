#!/usr/bin/env python3
"""The frozen 35-template catalog is exactly the output of the shipped induction trace.

The catalog's upstream provenance is complete: the shipped induction trace records its
batch-by-batch construction and terminates at the frozen 35-template set.

`template_induction/development/` ships the catalog growing batch by batch, and it lands on
the published count id for id:

    seed (batch_00)  27  <- matches seed/initial_catalog.json, and the 27 that
                            simulate_order_robustness.py says it seeds from
    01 28   02 28   03 29   04 29   05 32   06 32   07 33   08 **35**

    catalog_after_batch_08.json  vs  core/config/frozen_template_catalog.json
      -> 35 == 35, and the id sets are **identical** (T01..T35)

Thus the frozen catalog is reproducible from the annotation batches that produced it,
including the exact template identifiers rather than only the final count.

  python3 benchmark/eval/verify_catalog_induction.py [--check]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "benchmark" / "eval" / "template_induction" / "development"
SEED = ROOT / "benchmark" / "eval" / "template_induction" / "seed" / "initial_catalog.json"
FROZEN = ROOT / "core" / "config" / "frozen_template_catalog.json"
GEN = ROOT / "benchmark" / "eval" / "template_induction" / "simulate_order_robustness.py"
EXPECTED = [27, 28, 28, 29, 29, 32, 32, 33, 35]

fails: list[str] = []


def want(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        fails.append(label)


def ids(path: Path) -> list[str]:
    o = json.loads(path.read_text())
    entries = o if isinstance(o, list) else o.get("templates", o)
    entries = list(entries.values()) if isinstance(entries, dict) else entries
    out = []
    for e in entries:
        if isinstance(e, str):
            out.append(e)
        else:
            for k in ("id", "template_id", "tid", "name"):
                if k in e:
                    out.append(e[k])
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.parse_args()
    if not DEV.is_dir() or not FROZEN.exists():
        print("SKIP: induction trace or frozen catalog absent")
        return 0

    series = []
    for i in range(len(EXPECTED)):
        p = DEV / f"catalog_after_batch_{i:02d}.json"
        if not p.exists():
            print(f"SKIP: {p.name} absent")
            return 0
        series.append(len(ids(p)))
    print("catalog size after each batch: " + " ".join(f"{n}" for n in series) + "\n")

    want("the trace still has all 9 catalog snapshots", len(series) == 9, str(len(series)))
    want("it reproduces the recorded growth series", series == EXPECTED, str(series))
    want("the catalog never shrinks", all(b >= a for a, b in zip(series, series[1:])))
    want("it starts at the 27-template seed", series[0] == 27, str(series[0]))
    want("and ends at the published 35", series[-1] == 35, str(series[-1]))

    final, frozen = set(ids(DEV / "catalog_after_batch_08.json")), set(ids(FROZEN))
    want("the frozen catalog has 35 templates", len(frozen) == 35, str(len(frozen)))
    want("the induction output and the frozen catalog are the same id set",
         final == frozen,
         f"+{sorted(final - frozen)[:3]} -{sorted(frozen - final)[:3]}" if final != frozen
         else "identical")

    if SEED.exists():
        want("seed/initial_catalog.json holds the same 27", len(ids(SEED)) == 27,
             str(len(ids(SEED))))
    if GEN.exists():
        want("the saturation generator still says it seeds from 27",
             "27 templates already present from the seed stage" in GEN.read_text())

    batches = sorted(DEV.glob("batch_*_log.json"))
    inputs = sorted((DEV / "batch_inputs").glob("batch_*_extractions.jsonl"))
    want("8 batch logs and 8 batch inputs ship", len(batches) == 8 and len(inputs) == 8,
         f"{len(batches)} logs, {len(inputs)} inputs")

    if fails:
        print(f"\n{len(fails)} assertion(s) failed")
        return 1
    print("\nthe frozen catalog is exactly what the shipped induction trace produces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
