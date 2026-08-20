#!/usr/bin/env python3
"""Measure clean-trace false positives with and without the topology guard.

§5.3 reports 25.8 FP per million clean rule evaluations, rising to 83.3 without
$\\pi_{\\text{topo}}$, over a six-case subset and 504K evaluations. That run is not in the
artifact (see docs/GAP_AUDIT.md O21). This is an independent measurement of the same effect
using what is here: the recovered SQL, the two guard libraries, and a clean multi-rank trace.

It is **not** a reproduction of 25.8/83.3 — the denominators differ. What it does give is the
guard's effect on a trace anyone can download, and a per-rule breakdown of what fires.

    python3 benchmark/injection/measure_clean_fp.py --db <clean trace> --tp 2
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LIBS = ROOT / "core" / "config" / "ablation_libraries"
GEN = ROOT / "core" / "config" / "generated_sql.json"


def index(path):
    lib = json.loads(path.read_text())
    out = {}
    for cat, rules in lib["constraints"].items():
        for name, r in rules.items():
            out[r.get("name") or name] = r.get("applicable_conditions") or {}
    return out


def enabled(guard, topo):
    """Only the topology keys gate here; stage/precondition live inside the SQL."""
    for k, v in guard.items():
        if k not in topo:
            continue
        v = str(v).strip()
        try:
            if v.startswith(">="):
                ok = topo[k] >= int(v[2:])
            elif v.startswith(">"):
                ok = topo[k] > int(v[1:])
            elif v.startswith("<"):
                ok = topo[k] < int(v[1:])
            elif v.startswith("="):
                ok = topo[k] == int(v[1:])
            else:
                ok = True
        except ValueError:
            ok = True
        if not ok:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="a clean trace, ideally multi-rank")
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--top", type=int, default=10, help="how many firing rules to list")
    ap.add_argument("--gate", type=float, default=1.0,
                    help="clean-run gate: flag rules returning more than this %% of the "
                         "trace's rows (default 1.0)")
    args = ap.parse_args()
    try:
        import duckdb
    except ImportError:
        sys.exit("needs duckdb: pip install duckdb")

    topo = {"dp": args.dp, "tp": args.tp, "pp": args.pp, "ep": 1, "sp": 1}
    gen = json.loads(GEN.read_text())["rules"]
    have = [n for n in gen if gen[n].get("variants")]
    arms = {
        "guarded": index(LIBS / "lib_full.json"),
        "topology-stripped": index(LIBS / "lib_no_topo.json"),
    }
    con = duckdb.connect(args.db, read_only=True)
    rows = con.execute("SELECT count(*) FROM coredump").fetchone()[0]
    print(f"trace: {pathlib.Path(args.db).parent.parent.name}, {rows} rows, "
          f"DP={args.dp} TP={args.tp} PP={args.pp}\n")

    detail = {}
    for arm, guards in arms.items():
        sel = [n for n in have if n in guards and enabled(guards[n], topo)]
        fired = viol = err = ok = 0
        per_rule = []
        for n in sel:
            try:
                r = con.execute(gen[n]["variants"][0]["sql"]).fetchall()
            except Exception:
                err += 1
                continue
            ok += 1
            if r:
                fired += 1
                viol += len(r)
                per_rule.append((len(r), n))
        detail[arm] = sorted(per_rule, reverse=True)
        print(f"{arm:<20} enabled={len(sel):<4} executed={ok:<4} failed_to_compile={err:<4} "
              f"fired={fired:<4} violating_rows={viol}")

    g, s = detail["guarded"], detail["topology-stripped"]
    if g:
        print(f"\nremoving the topology guard takes firing rules from {len(g)} to {len(s)} "
              f"({len(s) / len(g):.1f}x)")
    print(f"\nrules firing on this clean trace under the guarded arm (top {args.top}):")
    for k, n in g[:args.top]:
        print(f"  {k:>8} rows   {n[:72]}")
    if g:
        share = 100 * g[0][0] / sum(k for k, _ in g)
        print(f"\nthe top rule alone accounts for {share:.0f}% of the flagged rows — on a clean "
              f"trace that points at the rule, not the run (see GAP_AUDIT O37)")

    # A clean-run acceptance gate, applied to every rule with recovered SQL rather than to
    # one arm: a rule returning a large fraction of a clean trace is defective by
    # construction, whatever its guard.
    executed = 0
    over = []
    for n in have:
        try:
            r = con.execute(gen[n]["variants"][0]["sql"]).fetchall()
        except Exception:
            continue
        executed += 1
        if len(r) > args.gate / 100 * rows:
            over.append((len(r), n))
    over.sort(reverse=True)
    print(f"\nclean-run gate at {args.gate}% of {rows} rows: {len(over)} of {executed} "
          f"executable rules would be rejected")
    for k, n in over:
        print(f"  {100 * k / rows:>5.1f}% ({k:>7} rows)  {n[:64]}")


if __name__ == "__main__":
    main()
