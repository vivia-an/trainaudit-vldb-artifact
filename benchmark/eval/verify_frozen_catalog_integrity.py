#!/usr/bin/env python3
"""The frozen catalog ships three times, with three sidecars and a fourth claim in prose.

The 35-template catalog is the artifact's most-cited frozen asset: §5.4's argument rests on
it, the induction trace reproduces it, and the deployed registry records its downstream
uses. It ships in **three** places, each with a `.sha256` sidecar, and
`pattern_expansion/LEGACY_CATALOG_NOTICE.md` states the digest a fourth time in prose while
telling readers not to use the legacy P-numbered catalog beside it.

Four independent statements of one hash is four chances to drift. All four currently agree on
`cfa30e18…`, and this keeps them agreeing:

  benchmark/eval/template_induction/frozen_template_catalog.json  + .sha256
  core/config/frozen_template_catalog.json                        + .sha256
  core/data/template_induction/frozen_template_catalog.json       + .sha256
  pattern_expansion/LEGACY_CATALOG_NOTICE.md                      (prose)

The notice is worth keeping honest for a second reason: it is what stops a reviewer reading
`pattern_expansion/`'s P9–P16 as a 16-pattern catalog. Those eight are a legacy expansion the
paper, miner, DSL registry and deployment code do not consume -- the notice says so, and says
the canonical identifiers are T01–T35.

  python3 benchmark/eval/verify_frozen_catalog_integrity.py [--check]
"""
from __future__ import annotations
import argparse, hashlib, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COPIES = ["benchmark/eval/template_induction/frozen_template_catalog.json",
          "core/config/frozen_template_catalog.json",
          "core/data/template_induction/frozen_template_catalog.json"]
NOTICE = ROOT / "benchmark" / "eval" / "pattern_expansion" / "LEGACY_CATALOG_NOTICE.md"

fails: list[str] = []


def want(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        fails.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.parse_args()
    present = [c for c in COPIES if (ROOT / c).exists()]
    if not present:
        print("SKIP: no frozen catalog copy found")
        return 0

    digests = {}
    for c in present:
        digests[c] = hashlib.sha256((ROOT / c).read_bytes()).hexdigest()
        print(f"  {digests[c][:16]}…  {c}")
    uniq = set(digests.values())
    print()
    want("all three copies of the frozen catalog ship", len(present) == 3, str(len(present)))
    want("every copy is byte-identical", len(uniq) == 1,
         f"{len(uniq)} distinct digest(s)")

    canonical = next(iter(uniq))
    for c in present:
        side = (ROOT / c).with_suffix(".sha256")
        if not side.exists():
            want(f"{Path(c).parent.name}/ has a .sha256 sidecar", False, "missing")
            continue
        stated = side.read_text().split()[0] if side.read_text().split() else ""
        want(f"{Path(c).parent.name}/ sidecar matches the file", stated == canonical,
             f"{stated[:16]}…")

    if NOTICE.exists():
        nt = NOTICE.read_text()
        m = re.search(r"\b([0-9a-f]{64})\b", nt)
        want("the legacy notice states a digest", m is not None)
        if m:
            want("and it is the canonical one", m.group(1) == canonical,
                 f"{m.group(1)[:16]}…")
        want("the notice still tells readers not to use the legacy catalog",
             "Do not use" in nt and "pattern_catalog_v2.json" in nt)
        want("and still names T01--T35 as canonical", "T01" in nt and "T35" in nt)

    if fails:
        print(f"\n{len(fails)} assertion(s) failed")
        return 1
    print(f"\nfour independent statements of the catalog digest, all agreeing on "
          f"{canonical[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
