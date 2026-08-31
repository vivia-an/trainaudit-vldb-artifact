"""Tolerance-aware prototype rules: instead of `field > 0` (trivial), use
`field > <auto-learned p1 from healthy>` to catch outliers below healthy
distribution. Tests whether tolerance-learned predicates have actual
detection power on the dead-block-0 trace.
"""
import json
import statistics
import sys
from pathlib import Path

_REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
sys.path.insert(0, str(_REPO / "trainaudit"))

from trainaudit.dsl import (Bound, BoundKind, Predicate, Scope, Template,
                              violation_event_ids)
from trainaudit.store import TraceStore


HEALTHY_TRACES = [
    "benchmark/eval/hunt_log/novel_hunt/olmo_core_moe_hybrid/trace_rank0.duckdb",
    "benchmark/eval/hunt_log/novel_hunt/olmo_core_smallmoe/trace_rank0.duckdb",
    "benchmark/eval/hunt_log/novel_hunt/olmo_core_olmoe/trace_rank0.duckdb",
]

BUGGY_TRACES = [
    ("small_hybrid_moe (dead block-0 MoE)",
     "benchmark/eval/hunt_log/novel_hunt/olmo_core_moe/trace_rank0.duckdb"),
    ("dense reordered_norm long (transient block-0 attn)",
     "benchmark/eval/hunt_log/novel_hunt/olmo_core_dense_reordered_long/trace_rank0.duckdb"),
]

# (hookpoint, json_field, percentile_for_lower_bound)
TARGETS = [
    ("module.fwd.post", "output.l2_norm", 1),       # learn p1 lower bound
    ("module.fwd.post", "output.abs_max", 1),
    ("optim.step.pre", "total_grad_l2", 1),
    ("comm.pre", "tensor_pre.l2_norm", 1),
    ("comm.post", "tensor_post.l2_norm", 1),
]


def learn_threshold(hp, field, pct):
    """Sample healthy traces, return pct-th percentile of `field` at `hp`."""
    samples = []
    for path in HEALTHY_TRACES:
        full = _REPO / path
        if not full.exists():
            continue
        store = TraceStore(str(full))
        store.flush()
        rows = store.conn.execute(
            f"SELECT CAST(json_extract(payload, '$.{field}') AS DOUBLE) "
            f"FROM events WHERE hookpoint = ?", [hp]
        ).fetchall()
        for (v,) in rows:
            if v is not None:
                samples.append(float(v))
    if not samples:
        return None, 0
    samples.sort()
    n = len(samples)
    idx = int(n * pct / 100.0)
    return samples[idx], n


def build_proto(hp, field, lower_bound):
    return Predicate(
        id=f"proto-tol/{hp}/{field}_above_p1_{lower_bound:.3e}",
        template=Template.PAYLOAD_FIELD_COMPARE,
        family="F-PROTO-TOL",
        scope=Scope(hookpoint=hp),
        bound=Bound(kind=BoundKind.BOUND, field=field,
                    op=">", value=lower_bound),
        description=f"{field} at {hp} must be > p1-learned ({lower_bound:.3e})",
    )


def main():
    print(f"\n{'='*80}\nLearn p1 thresholds from healthy traces\n{'='*80}")
    protos = []
    for hp, field, pct in TARGETS:
        thresh, n = learn_threshold(hp, field, pct)
        if thresh is None:
            print(f"  {hp}.{field}: no data")
            continue
        print(f"  {hp}.{field}: p{pct} = {thresh:.4e}  (n={n} samples)")
        protos.append(build_proto(hp, field, thresh))

    print(f"\n{'='*80}\nEvaluate {len(protos)} tolerance prototypes\n{'='*80}\n")
    rows = []
    for label, path in BUGGY_TRACES:
        full = _REPO / path
        if not full.exists():
            continue
        store = TraceStore(str(full))
        for p in protos:
            try:
                ids = violation_event_ids(store, p)
                fire = len(ids)
            except Exception as e:
                fire = f"ERR:{type(e).__name__}"
            print(f"  [BUG] {label[:40]:<40} | {p.id[20:60]:<40} | fire = {fire}")
            rows.append({"trace": label, "kind": "BUG",
                         "proto_id": p.id, "fire": fire})

    # FP audit on healthy
    print()
    fp_seen = {p.id: 0 for p in protos}
    for path in HEALTHY_TRACES:
        full = _REPO / path
        if not full.exists():
            continue
        store = TraceStore(str(full))
        label = Path(path).parent.name
        for p in protos:
            ids = violation_event_ids(store, p)
            if ids:
                fp_seen[p.id] += len(ids)
            print(f"  [HEALTHY] {label[:30]:<30} | {p.id[20:60]:<40} | "
                  f"fire = {len(ids)}")

    print(f"\n{'='*80}\nSummary\n{'='*80}\n")
    for p in protos:
        bug_fires = [r["fire"] for r in rows if r["proto_id"] == p.id]
        bug_count = sum(1 for x in bug_fires if isinstance(x, int) and x > 0)
        print(f"  {p.id[20:]}")
        print(f"    bug-fire: {bug_count}/{len(BUGGY_TRACES)}  "
              f"buggy-counts: {bug_fires}  "
              f"healthy-FP-total: {fp_seen[p.id]}")

    out_path = _REPO / "benchmark" / "eval" / "mining_prototype_tolerance.json"
    out_path.write_text(json.dumps(
        {"prototypes": [{"id": p.id, "desc": p.description} for p in protos],
         "rows": rows, "healthy_FP_total": fp_seen},
        indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
