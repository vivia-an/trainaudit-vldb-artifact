#!/usr/bin/env python3
"""Cross-check appendix tab:realse-class-coverage against the other two tables.

The table claims the Real-SE set exercises every one of the 13 taxonomy classes, naming a
representative case per class. Three of its properties are machine-checkable and all hold:

* its 13 labels are the same **set and the same order** as `tab:taxonomy` in main.tex;
* every representative resolves to a real manifest case, and all 13 are distinct;
* each named constraint matches what `tab:detection-results` records for that same commit,
  so the two appendix tables agree.

**What the assignment does not follow from.** `benchmark/eval/category_resolved.json` does
ship a per-record key -- `old_category -> new_category` for the 295-pool, in exactly these
13 label names -- and against it **9 of the 13 assignments differ**: B1 resolves to `moe`
but represents Grad Sync, B8 to `moe` but represents Communication, M-020 to `control_flow`
but represents Sharding, D-029 to `communication` but represents Offload.

That is not a contradiction, and the caption says why: the table is captioned "Real-SE
diversity across **descriptive subsystem labels**", a different axis from the corpus
category. B1's constraint really is `replica-cksum-equal` across data-parallel replicas,
whatever bucket the bug itself sits in.

The residual risk is presentational and narrow. The table reuses the same 13 label *names*
under a column header reading "Primary label", so a reviewer who looks up B1 in the shipped
key finds `moe` and has to work out that the two axes differ. Renaming the column, or a
clause noting the axis is not `tab:taxonomy`'s, removes it.

  python3 benchmark/eval/verify_class_coverage_table.py [--check]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ROW = re.compile(
    r"^\s*([A-Za-z][A-Za-z .]*?)\s*&\s*([A-Za-z]+)-([0-9a-f]{7,10}) "
    r"\(\\texttt\{([^}]+)\}\)", re.M)


def block(text, label, closer):
    i = text.index(label)
    return text[i:text.index(closer, i)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    fail = []

    apx = (ROOT / "paper" / "appendix.tex").read_text()
    main_tex = (ROOT / "paper" / "main.tex").read_text()
    rows = ROW.findall(block(apx, "label{tab:realse-class-coverage}", r"\end{tabularx}"))

    tax = block(main_tex, "tab:taxonomy", r"\end{tabularx}")
    tax_labels = [m.group(1).strip() for m in
                  re.finditer(r"^([A-Za-z][A-Za-z /.]*?)\s*&\s*\d+", tax, re.M)]
    labels = [r[0].strip() for r in rows]

    def want(label, cond, detail):
        """cond is the assertion; detail is only displayed. Passing the same value as
        both -- which two call sites originally did -- makes the test a tautology, so
        the signature no longer allows it."""
        ok = bool(cond) if isinstance(cond, bool) else cond == detail
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:52} {detail}")
        if not ok:
            fail.append(f"{label}: {detail}")

    print(f"tab:realse-class-coverage -- {len(rows)} rows")
    want("labels identical to tab:taxonomy, as a set", set(labels), set(tax_labels))
    want("labels in the same order as tab:taxonomy", labels, tax_labels)
    want("every representative case is distinct",
         len({r[2] for r in rows}), len(rows))

    det = {m.group(2): m.group(3).strip() for m in re.finditer(
        r"^\s*([A-Za-z]+)-([0-9a-f]{7,10})\s*&[^&]+&[^&]+&\s*([^&]+?)\s*&",
        block(apx, "label{tab:detection-results}", r"\end{table}"), re.M)}
    mism = [f"{c}: '{k}' vs '{det.get(c)}'" for _, _, c, k in rows
            if det.get(c, "").strip() != k.strip()]
    want("constraints agree with tab:detection-results", not mism,
         mism or "all 13 match")

    man = json.loads((ROOT / "benchmark" / "eval" / "real_sdc"
                      / "real_sdc_manifest.json").read_text())
    cases = man if isinstance(man, list) else (
        man.get("cases") or next(v for v in man.values() if isinstance(v, list)))
    unresolved = [c for _, _, c, _ in rows
                  if not any((x.get("fixed_commit") or "").startswith(c) for x in cases)]
    want("every representative resolves in the manifest", not unresolved,
         unresolved or "all 13 resolve")

    print("\nnot checkable -- the manifest labels cases in a different vocabulary:")
    for label, _fw, commit, _k in rows:
        hit = next((x for x in cases
                    if (x.get("fixed_commit") or "").startswith(commit)), None)
        if hit:
            cat = hit.get("category", "?")
            same = cat.lower().replace("_", " ").startswith(label.lower()[:5])
            print(f"    {label:16} {hit['case_id']:11} manifest says {cat}"
                  f"{'   (aligns)' if same else ''}")
    print("  so the class assignment rests on the authors' reading, not a shipped key.")

    if a.check:
        if fail:
            print(f"\nFAIL: {fail}", file=sys.stderr)
            return 1
        print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
