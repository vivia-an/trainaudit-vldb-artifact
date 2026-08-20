#!/usr/bin/env python3
"""Check that appendix-refs.tex still matches what appendix.tex builds to.

main.tex resolves its 18 `Appendix~\\ref{app:...}` calls through appendix-refs.tex, which
is generated from a build of the separately compiled supplement. If the supplement is
edited and that file is not regenerated, the main paper silently prints stale section
letters — a failure that neither document's own build would report.

    python3 paper/check_appendix_refs.py                 # compare against appendix.aux
    python3 paper/check_appendix_refs.py --aux path.aux
"""
import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent


def body(path):
    """File text with LaTeX comments removed — main.tex documents the mechanism in a
    comment containing a literal Appendix reference, which is not a real one."""
    return re.sub(r"(?<!\\)%.*", "", path.read_text(errors="replace"))


def labels(path):
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}",
                         path.read_text(errors="replace")):
        name = m.group(1)
        if not name.endswith("@cref"):
            out[name] = m.group(2)
    return out


def graphics_and_inputs(text):
    """Every path main.tex/appendix.tex pulls in, as written."""
    out = set()
    for pat in (r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                r"\\input\{([^}]+)\}",
                r"\\bibliography\{([^}]+)\}"):
        for m in re.finditer(pat, text):
            out.add(m.group(1))
    return out


def check_inputs_resolve():
    """The documented build is `cd paper && pdflatex main.tex`, so every referenced path
    must resolve relative to paper/. figures/ and benchmark/ are symlinks for this reason."""
    bad = []
    n = 0
    for src in ("main.tex", "appendix.tex"):
        f = HERE / src
        if not f.exists():
            continue
        for rel in sorted(graphics_and_inputs(body(f))):
            n += 1
            cands = [HERE / rel, HERE / (rel + ".tex"), HERE / (rel + ".bib")]
            if not any(c.exists() for c in cands):
                bad.append(f"{src} -> {rel}")
    print(f"{n} graphics/input paths referenced by the paper sources")
    if bad:
        print(f"  {len(bad)} do not resolve from paper/ — the documented build would fail:")
        for b in bad[:10]:
            print(f"    {b}")
        return 1
    print("  all resolve from paper/ (figures/ and benchmark/ are symlinks to the repo root)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default=str(HERE / "appendix-refs.tex"))
    ap.add_argument("--aux", help="appendix.aux from a fresh build of appendix.tex")
    args = ap.parse_args()

    rc = check_inputs_resolve()
    print()
    refs = pathlib.Path(args.refs)
    if not refs.exists():
        sys.exit(f"missing {refs}")
    ship = labels(refs)
    print(f"{refs.name}: {len(ship)} labels")

    if not args.aux:
        found = re.findall(r"Appendix~\\ref\{(app:[^}]+)\}", body(HERE / "main.tex"))
        n, used = len(found), set(found)
        missing = sorted(used - set(ship))
        print(f"main.tex makes {n} Appendix references over {len(used)} distinct labels")
        if missing:
            print(f"  UNRESOLVED: {', '.join(missing)}")
            return 1
        print("  every referenced label is present")
        print("  (pass --aux <appendix.aux> to also compare the numbers)")
        return rc

    fresh = labels(pathlib.Path(args.aux))
    only_ship = sorted(set(ship) - set(fresh))
    only_fresh = sorted(set(fresh) - set(ship))
    changed = [(k, ship[k], fresh[k]) for k in sorted(set(ship) & set(fresh))
               if ship[k] != fresh[k]]
    print(f"fresh build: {len(fresh)} labels")
    bad = 0
    for label, items in (("only in appendix-refs.tex", only_ship),
                         ("only in the fresh build", only_fresh)):
        if items:
            bad += len(items)
            print(f"  {label}: {', '.join(items[:8])}")
    if changed:
        bad += len(changed)
        print("  numbers changed:")
        for k, a, b in changed[:12]:
            print(f"    {k:<44} refs={a!r} build={b!r}")
    if bad:
        print(f"  {bad} discrepancy(ies) — regenerate appendix-refs.tex from appendix.aux")
        return 1
    print("  identical: same labels, same numbers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
