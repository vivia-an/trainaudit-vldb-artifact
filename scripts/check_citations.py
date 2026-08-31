#!/usr/bin/env python3
"""Validate citations shared by the manuscript and appendix.

Checks that every cited key resolves, cited entries carry their required fields,
the appendix declares a bibliography, and no duplicate key or title is present.

Run with: python3 scripts/check_citations.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"


def cite_keys(text: str) -> set[str]:
    return {k.strip() for m in re.findall(r"\\cite[tp]?\{([^}]*)\}", text)
            for k in m.split(",") if k.strip()}


def main() -> int:
    fail = []
    bib = PAPER / "main.bib"
    if not bib.exists():
        print("FAIL: paper/main.bib is missing", file=sys.stderr)
        return 1
    entries = set(re.findall(r"@\w+\{([^,]+),", bib.read_text()))

    main_keys = cite_keys((PAPER / "main.tex").read_text())
    apx_text = (PAPER / "appendix.tex").read_text()
    apx_keys = cite_keys(apx_text)

    print(f"main.tex cites {len(main_keys)}, appendix.tex {len(apx_keys)}; "
          f"main.bib holds {len(entries)} entries")
    for label, keys in (("main.tex", main_keys), ("appendix.tex", apx_keys)):
        missing = sorted(keys - entries)
        print(f"  {'ok  ' if not missing else 'FAIL'}  every {label} key resolves"
              f"{'' if not missing else '  MISSING ' + str(missing)}")
        if missing:
            fail.append(f"{label}: undefined {missing}")

    # The supplement builds independently, so it needs its own \bibliography.
    has_bib = re.search(r"\\bibliography\{(\w+)\}", apx_text)
    ok = bool(has_bib) and (PAPER / f"{has_bib.group(1)}.bib").exists()
    print(f"  {'ok  ' if ok else 'FAIL'}  the supplement declares a bibliography  "
          f"{has_bib.group(0) if has_bib else 'NONE'}")
    if not ok:
        fail.append("appendix.tex has no resolvable \\bibliography")

    # Cited entries must also be well-formed: resolution alone does not stop a reference
    # rendering as "[?]" or as an author-and-title with no venue.
    body = bib.read_text()
    parsed = re.findall(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", body, re.S)
    REQUIRED = {"article": ("author", "title", "journal", "year"),
                "inproceedings": ("author", "title", "booktitle", "year"),
                "incollection": ("author", "title", "booktitle", "year"),
                "techreport": ("author", "title", "year"),
                "phdthesis": ("author", "title", "year"),
                "book": ("author", "title", "year")}
    cited = main_keys | apx_keys
    malformed = []
    for typ, key, fields_src in parsed:
        key = key.strip()
        if key not in cited:
            continue
        have = {f.lower() for f in re.findall(r"^\s*(\w+)\s*=", fields_src, re.M)}
        miss = [f for f in REQUIRED.get(typ.lower(), ("title",)) if f not in have]
        if miss:
            malformed.append(f"{key} (@{typ.lower()}) missing {miss}")
    print(f"  {'ok  ' if not malformed else 'FAIL'}  every cited entry carries its "
          f"required fields  [{len(cited)} cited]")
    if malformed:
        for m in malformed[:5]:
            print(f"        {m}")
        fail.append(f"malformed cited entries: {malformed[:3]}")

    keys_all = [k.strip() for _, k, _ in parsed]
    dupes = sorted({k for k in keys_all if keys_all.count(k) > 1})
    print(f"  {'ok  ' if not dupes else 'FAIL'}  no bib key is defined twice"
          f"{'' if not dupes else '  ' + str(dupes)}")
    if dupes:
        fail.append(f"duplicate bib keys: {dupes}")

    def _norm(t):
        return re.sub(r"[^a-z0-9]", "", t.lower())[:70]
    titles = {}
    dup_titles = []
    for _, key, fields_src in parsed:
        m = re.search(r"title\s*=\s*[{\"](.+?)[}\"]\s*,?\s*$", fields_src, re.M | re.S)
        if not m:
            continue
        t = _norm(m.group(1))
        if t in titles:
            dup_titles.append((titles[t], key.strip()))
        titles[t] = key.strip()
    print(f"  {'ok  ' if not dup_titles else 'FAIL'}  no two entries share a title"
          f"{'' if not dup_titles else '  ' + str(dup_titles[:3])}")
    if dup_titles:
        fail.append(f"same work under two keys: {dup_titles[:3]}")

    only = sorted(apx_keys - main_keys)
    print(f"  note  {len(only)} appendix-only key(s): {only}")
    print(f"  note  {len(entries - (main_keys | apx_keys))} bib entries never cited "
          f"(harmless -- BibTeX emits only cited entries)")

    if fail:
        print(f"\nFAIL: {fail}", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
