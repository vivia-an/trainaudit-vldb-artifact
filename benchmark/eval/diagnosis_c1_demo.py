#!/usr/bin/env python3
"""Run the shipped C1 expander on a real trace and show what it produces.

app:diagnosis quantifies the diagnosis chain as candidate-set cardinalities: 9/17 at
|S_rule| = 1, a median C1 expansion to 6 with a peak of 12, and the 1 -> 6 -> 1 trajectory in
fig:diagnosis-mechanism. GAP_AUDIT O43 records that no shipped *data* file holds such a
cardinality. This checks the shipped *code*.

    python3 benchmark/eval/diagnosis_c1_demo.py --db <events-schema trace>
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "core" / "trainaudit_pkg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="an events-schema trace (trace-events-v2)")
    ap.add_argument("--rule", default="T1-replica-cksum-equal")
    args = ap.parse_args()
    try:
        from trainaudit.store import TraceStore
        from trainaudit.diagnosis import expander
    except ImportError as exc:
        sys.exit(f"cannot import trainaudit: {exc}")

    st = TraceStore(args.db)
    rows = st.query("SELECT event_id FROM events WHERE hookpoint='build.snapshot' LIMIT 1")
    if not rows:
        sys.exit("no build.snapshot event in this trace")
    rep = expander.expand_violation(st, args.rule, rows[0][0])
    d = rep.to_dict()

    print(f"C1 expander on {pathlib.Path(args.db).parent.name}, rule {args.rule}\n")
    for k, v in d.items():
        print(f"  {k:<22}{str(v)[:96]}")
    print(f"  {'context_events':<22}{len(rep.context_events)}")

    cardinality = [k for k in d if any(w in k for w in ("cand", "size", "set"))]
    print(f"\ncardinality-like fields in the report: {cardinality or 'NONE'}")
    print("""
So the chain produces a *located suspect* with a deterministic hypothesis — rule, event,
rank, step — not a candidate set that shrinks. That is a real diagnosis capability, and it
matches the qualitative description in the paper. What it does not produce is the quantity
app:diagnosis reports: there is no |S_rule|, |S_C1| or |S_C2| here, just as there is none in
the data files. The 53% and the median of 6 come from neither.""")


if __name__ == "__main__":
    main()
