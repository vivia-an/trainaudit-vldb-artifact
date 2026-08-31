#!/usr/bin/env python3
"""The release bundles are checksummed from the repo, which is what makes them archival.

PVLDB asks for supplemental material in a publicly accessible archival repository (R5). The
two trace bundles are GitHub release assets rather than repository content, so their
integrity has to be verifiable from something committed -- otherwise a reviewer downloading
them has no way to tell whether they are the files the paper was written against.

They are. `benchmark/injection/trace_db_manifest.csv` and `events_trace_manifest.csv` carry
a per-file SHA-256, and `fetch_trace_dbs.sh` checks every downloaded file against them.
Measured against the fetched bundles: 139 and 37 files, every one listed, **zero unlisted**.

This is the offline half -- that the manifests are complete and every digest well-formed. The download-time half was mutation-tested: flipping one bit in a single file of either bundle makes
`fetch_trace_dbs.sh --verify-only` exit 1, so corruption is caught in practice and not just in
principle.
The download-time comparison is the `--verify-only` path, which runs as its own check group
when the bundles are present.

  python3 scripts/check_release_manifests.py
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = {
    "benchmark/injection/trace_db_manifest.csv": 139,
    "benchmark/injection/events_trace_manifest.csv": 37,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def main() -> int:
    fail = []
    for rel, expect in MANIFESTS.items():
        p = ROOT / rel
        if not p.exists():
            fail.append(f"{rel} is missing")
            print(f"  FAIL  {rel} missing")
            continue
        rows = list(csv.DictReader(p.open()))
        if "sha256" not in (rows[0] if rows else {}):
            fail.append(f"{rel} has no sha256 column")
            print(f"  FAIL  {rel} has no sha256 column")
            continue
        bad = [r["path"] for r in rows if not HEX64.match((r.get("sha256") or "").strip())]
        dupes = {r["path"] for r in rows if [x["path"] for x in rows].count(r["path"]) > 1}
        ok = rows and not bad and not dupes and len(rows) == expect
        print(f"  {'ok  ' if ok else 'FAIL'}  {rel.split('/')[-1]:28} {len(rows):>4} entries, "
              f"{len(rows) - len(bad):>4} well-formed digests"
              + (f", MALFORMED {bad[:3]}" if bad else "")
              + (f", DUPLICATE paths {sorted(dupes)[:3]}" if dupes else "")
              + ("" if len(rows) == expect else f", expected {expect}"))
        if bad:
            fail.append(f"{rel}: {len(bad)} malformed digest(s)")
        if dupes:
            fail.append(f"{rel}: duplicate paths")
        if len(rows) != expect:
            fail.append(f"{rel}: {len(rows)} entries, expected {expect}")

    # Shared digests, classified. Everything currently shared is expected; a NEW cross-run
    # duplicate is not, and would mean two runs shipped the same data under different names.
    import collections
    O31_PAIR = {"tp_normal_db", "tp_router_test_db"}
    EMPTY_SHELLS = {"olmo_core_moe_ep2_actckpt", "olmo_core_moe_reordered_norm",
                    "olmo_core_olmo2_271M", "olmo_core_tp2"}
    EMPTY_HEADER = 12288
    for _name, _expected_cross in (
            ("benchmark/injection/trace_db_manifest.csv", O31_PAIR),
            ("benchmark/injection/events_trace_manifest.csv", EMPTY_SHELLS)):
        core = ROOT / _name
        if not core.exists():
            continue
        rows = list(csv.DictReader(core.open()))
        by = collections.defaultdict(list)
        for r in rows:
            by[r["sha256"]].append(r)
        unexpected = []
        classes = collections.Counter()
        for dig, rs in by.items():
            if len(rs) < 2:
                continue
            sizes = {int(r["size_bytes"]) for r in rs}
            runs = {r["run"] for r in rs}
            if sizes == {EMPTY_HEADER}:
                classes["empty header"] += 1
            elif len(runs) == 1:
                classes["within-run rank pair"] += 1
            elif runs <= _expected_cross:
                classes["recorded duplication"] += 1
            else:
                unexpected.append(f"{dig[:10]} across {sorted(runs)}")
        ok = not unexpected
        print(f"  {'ok  ' if ok else 'FAIL'}  {core.name:28} shared digests all expected  "
              f"[{dict(classes)}]")
        if unexpected:
            for u in unexpected[:4]:
                print(f"        unexpected cross-run duplicate: {u}")
            fail.append(f"{core.name}: unexpected cross-run duplicates {unexpected[:3]}")

    print("\nso a reviewer can detect a tampered or truncated release asset from the repo alone;")
    print("fetch_trace_dbs.sh --verify-only performs the comparison once the bundles are local.")
    if fail:
        print(f"\nFAIL: {fail}", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
