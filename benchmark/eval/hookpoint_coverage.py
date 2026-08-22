#!/usr/bin/env python3
"""Compare three hookpoint vocabularies: the paper's, the corpus's, and the collector's.

§4.4 describes the instrumentation as five canonical lifecycle points plus three auxiliary
taps, "the 8 hookpoints used by the verifier's indexer". Two other vocabularies exist and
neither matches it name-for-name:

  * what the corpus says each bug needs   (annotations_392_v2.json, `check_stage`, 11 values)
  * what the collector actually emits     (events-schema traces, `hookpoint`, 13 values)

This lays them side by side. Re-aggregation of recorded data; the mapping between vocabularies
is stated, not inferred silently.

    python3 benchmark/eval/hookpoint_coverage.py [--traces DIR]
"""
import argparse
import collections
import glob
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

# §4.4, verbatim: five lifecycle points + three auxiliary taps.
PAPER = ["before-forward", "after-forward", "main-grad-in-backward", "after-backward",
         "before-optimizer", "checkpoint save/load", "distributed.all_reduce",
         "build snapshot"]

# The correspondence a reader would draw. Stated so it can be argued with.
CORPUS_TO_PAPER = {
    "before_forward": "before-forward",
    "after_forward": "after-forward",
    "main_grad_in_backward": "main-grad-in-backward",
    "after_backward": "after-backward",
    "before_optimizer": "before-optimizer",
    "checkpoint_save": "checkpoint save/load",
    "checkpoint_load": "checkpoint save/load",
    "all_reduce": "distributed.all_reduce",
    "build": "build snapshot",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", help="a directory of events-schema traces")
    args = ap.parse_args()

    f = HERE / "v2_full" / "annotations_392_v2.json"
    if not f.exists():
        sys.exit(f"missing {f}")
    recs = json.loads(f.read_text())["annotations"]
    stage = collections.Counter(str(r.get("check_stage")) for r in recs)

    print(f"the paper's §4.4 list: {len(PAPER)} hookpoints\n")
    print(f"corpus `check_stage` ({len(stage)} distinct, {len(recs)} records):")
    mapped = unmapped = 0
    for k, n in stage.most_common():
        tgt = CORPUS_TO_PAPER.get(k)
        if tgt:
            mapped += n
        else:
            unmapped += n
        print(f"  {n:>4}  {k:<24}{'-> ' + tgt if tgt else '** no counterpart in §4.4'}")
    print(f"\nrecords whose stage maps onto the paper's list: {mapped}/{len(recs)} "
          f"({100 * mapped / len(recs):.0f}%)")
    print(f"records whose stage does not:                  {unmapped}/{len(recs)} "
          f"({100 * unmapped / len(recs):.0f}%)")
    absent = [k for k in stage if k not in CORPUS_TO_PAPER]
    print(f"  the unmapped stages: {', '.join(f'{k} ({stage[k]})' for k in absent)}")

    if not args.traces:
        print("\n(pass --traces DIR, from TAG=trace-events-v2 fetch_trace_dbs.sh, to add the\n"
              " vocabulary the collector actually emits)")
        return 0
    try:
        import duckdb
    except ImportError:
        sys.exit("needs duckdb")
    hp = collections.Counter()
    for p in sorted(glob.glob(str(pathlib.Path(args.traces) / "*" / "trace_rank0.duckdb"))):
        try:
            c = duckdb.connect(p, read_only=True)
            for h, n in c.execute("SELECT hookpoint, count(*) FROM events GROUP BY 1").fetchall():
                hp[h] += n
            c.close()
        except Exception:
            continue
    print(f"\ncollector `hookpoint` ({len(hp)} distinct, across {len(list(glob.glob(str(pathlib.Path(args.traces)/'*'/'trace_rank0.duckdb'))))} traces):")
    for h, n in hp.most_common():
        print(f"  {n:>9}  {h}")
    print("""
Three vocabularies, none matching name-for-name. Worth reconciling in §4.4:

  * The collector emits hookpoints the paper does not mention — `functional.softmax`,
    `loss.call`, `dataloader.batch`, `jitter.probe`.
  * It has one backward tap (`module.bwd`) where §4.4 names two points
    (`main-grad-in-backward` and `after-backward`).
  * The corpus's `init` stage has no dedicated hookpoint in either list; `build.snapshot` is
    the nearest, and it fires once per run.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
