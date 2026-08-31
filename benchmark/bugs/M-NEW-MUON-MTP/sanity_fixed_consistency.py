"""Sanity check on the fixed long-run telemetry.

Theory: fixed commit tags every word_embeddings replica, all replicas land in
AdamW. Gradients are all-reduced via _allreduce_word_embedding_grads, init is
replicated, AdamW state is therefore identical across PP ranks. The two tied
copies must be bit-identical (modulo BF16 reduction order) at every step.

If |ck(pp_rank=0, step=t) - ck(pp_rank=1, step=t)| > tol on the fixed run, the
divergence is inside the telemetry plumbing (e.g. reading a stale shard,
different proj device/dtype path) -- NOT inside the model. Abort and surface.

Usage:
    python sanity_fixed_consistency.py /path/to/fixed/steps.jsonl

Reads `<path>.rank0` and `<path>.rank1` automatically.
"""
import json
import sys
from collections import defaultdict


def load_steps(path):
    by_step = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "_meta" in rec or "step" not in rec:
                continue
            by_step[rec["step"]] = rec
    return by_step


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <fixed/steps.jsonl>", file=sys.stderr)
        sys.exit(2)

    base = sys.argv[1]
    rank0 = load_steps(base + ".rank0")
    rank1 = load_steps(base + ".rank1")

    common = sorted(set(rank0) & set(rank1))
    if not common:
        print(f"FAIL: no overlapping steps between {base}.rank0 and .rank1",
              file=sys.stderr)
        sys.exit(1)

    tol = 1e-3
    bad = []
    deltas = []
    for s in common:
        c0 = rank0[s].get("embed_proj_checksum")
        c1 = rank1[s].get("embed_proj_checksum")
        if c0 is None or c1 is None:
            continue
        d = abs(c0 - c1)
        deltas.append(d)
        if d > tol:
            bad.append((s, c0, c1, d))

    n = len(deltas)
    print(f"[sanity] fixed run: {n} steps with both ranks reporting checksum")
    if n:
        print(f"[sanity] delta stats: min={min(deltas):.3e} "
              f"max={max(deltas):.3e} mean={sum(deltas)/n:.3e}")

    if bad:
        print(f"FAIL: {len(bad)} steps exceed tol={tol}", file=sys.stderr)
        for s, c0, c1, d in bad[:5]:
            print(f"  step={s} pp0={c0:.6e} pp1={c1:.6e} |delta|={d:.3e}",
                  file=sys.stderr)
        sys.exit(1)
    print(f"PASS: all deltas <= tol={tol}")


if __name__ == "__main__":
    main()
