"""Step 1 of plan A follow-through: take 5 prototype rules from the parametric
L2 output and evaluate them on:
  - 2 buggy traces (small_hybrid_moe + dense reordered_norm — both fire
    existing T0-norm-output-unit-rms; we want to know if our new prototypes
    independently detect the same anomaly OR something orthogonal)
  - 5 healthy traces (must fire 0 — confirms FP rate)

Output: a TP/FP matrix per prototype.
"""
import json
import sys
from pathlib import Path

_REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
sys.path.insert(0, str(_REPO / "trainaudit"))

from trainaudit.dsl import (Bound, BoundKind, Predicate, Scope, Template,
                              violation_event_ids)
from trainaudit.store import TraceStore


# ---- 5 prototype rules from L2-accepted novel set ----------------------------

PROTOTYPES = [
    Predicate(
        id="proto/comm.pre/tensor_pre.l2_norm_nonzero",
        template=Template.PAYLOAD_FIELD_COMPARE,
        family="F-PROTO",
        scope=Scope(hookpoint="comm.pre"),
        bound=Bound(kind=BoundKind.BOUND, field="tensor_pre.l2_norm",
                    op=">", value=0),
        description="pre-allreduce gradient l2_norm must be > 0 "
                    "(catches all-zero grad sent into comm)",
    ),
    Predicate(
        id="proto/comm.post/tensor_post.l2_norm_positive",
        template=Template.PAYLOAD_FIELD_COMPARE,
        family="F-PROTO",
        scope=Scope(hookpoint="comm.post"),
        bound=Bound(kind=BoundKind.BOUND, field="tensor_post.l2_norm",
                    op=">", value=0),
        description="post-allreduce gradient l2_norm must be > 0 "
                    "(catches comm collapse/scatter loss)",
    ),
    Predicate(
        id="proto/optim.step.pre/total_grad_l2_nonzero",
        template=Template.PAYLOAD_FIELD_COMPARE,
        family="F-PROTO",
        scope=Scope(hookpoint="optim.step.pre"),
        bound=Bound(kind=BoundKind.BOUND, field="total_grad_l2",
                    op=">", value=0),
        description="total_grad_l2 must be > 0 at every optim.step.pre "
                    "(catches dead-grad before optimizer updates params)",
    ),
    Predicate(
        id="proto/module.fwd.post/output.l2_norm_nonzero",
        template=Template.PAYLOAD_FIELD_COMPARE,
        family="F-PROTO",
        scope=Scope(hookpoint="module.fwd.post"),
        bound=Bound(kind=BoundKind.BOUND, field="output.l2_norm",
                    op=">", value=0),
        description="every module forward output must have l2_norm > 0 "
                    "(catches dead-activation paths e.g. dead block-0 MoE)",
    ),
    Predicate(
        id="proto/module.fwd.post/output.abs_max_nonzero",
        template=Template.PAYLOAD_FIELD_COMPARE,
        family="F-PROTO",
        scope=Scope(hookpoint="module.fwd.post"),
        bound=Bound(kind=BoundKind.BOUND, field="output.abs_max",
                    op=">", value=0),
        description="every module forward output must have abs_max > 0 "
                    "(catches all-zero activations differently from l2 sum)",
    ),
]


# ---- traces: 2 buggy, 5 healthy -------------------------------------------

BUGGY_TRACES = [
    ("small_hybrid_moe (dead block-0 MoE)",
     "benchmark/eval/hunt_log/novel_hunt/olmo_core_moe/trace_rank0.duckdb"),
    ("dense reordered_norm long (transient block-0 attn)",
     "benchmark/eval/hunt_log/novel_hunt/olmo_core_dense_reordered_long/"
     "trace_rank0.duckdb"),
]

HEALTHY_TRACES = [
    ("olmo_core moe_hybrid (pre-norm)",
     "benchmark/eval/hunt_log/novel_hunt/olmo_core_moe_hybrid/trace_rank0.duckdb"),
    ("olmo_core smallmoe (with shared_mlp)",
     "benchmark/eval/hunt_log/novel_hunt/olmo_core_smallmoe/trace_rank0.duckdb"),
    ("olmo_core olmoe_1B_7B (production-scale)",
     "benchmark/eval/hunt_log/novel_hunt/olmo_core_olmoe/trace_rank0.duckdb"),
    ("olmo_core olmo2_370M (large dense reordered_norm)",
     "benchmark/eval/hunt_log/novel_hunt/olmo_core_olmo2_370M/trace_rank0.duckdb"),
    ("megatron clean (dense)",
     "benchmark/eval/hunt_log/novel_hunt/megatron_clean/trace_rank0.duckdb"),
]


def main():
    rows = []
    print(f"\n{'='*80}\nProtoype rule evaluation matrix\n{'='*80}\n")

    # Headers
    proto_ids = [p.id.split("/")[-1] for p in PROTOTYPES]

    print(f"\n{'TRACE':<55} | " + " | ".join(f"{p[:28]:<28}" for p in proto_ids))
    print("-" * 200)

    for label, path in BUGGY_TRACES + HEALTHY_TRACES:
        full = _REPO / path
        if not full.exists():
            print(f"  MISSING: {path}")
            continue
        store = TraceStore(str(full))
        verdicts = []
        for p in PROTOTYPES:
            try:
                ids = violation_event_ids(store, p)
                verdicts.append(len(ids))
            except Exception as e:  # noqa: BLE001
                verdicts.append(f"ERR:{type(e).__name__}")
        prefix = "[BUG]" if (label, path) in BUGGY_TRACES else "[ok]"
        print(f"{prefix} {label:<48} | " +
              " | ".join(f"{str(v):<28}" for v in verdicts))
        rows.append({"trace": label, "kind": prefix.strip("[]"),
                     "verdicts": dict(zip(proto_ids, verdicts))})

    out = {
        "n_prototypes": len(PROTOTYPES),
        "n_buggy_traces": len(BUGGY_TRACES),
        "n_healthy_traces": len(HEALTHY_TRACES),
        "prototypes": [{"id": p.id, "description": p.description}
                       for p in PROTOTYPES],
        "rows": rows,
    }
    out_path = _REPO / "benchmark" / "eval" / "mining_prototype_eval.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")

    # Compute summary metrics: TP rate (bug detection), FP rate (clean fires)
    print(f"\n{'='*80}\nSummary\n{'='*80}\n")
    for i, p in enumerate(PROTOTYPES):
        proto_id = proto_ids[i]
        bug_fires = [r["verdicts"][proto_id] for r in rows
                     if r["kind"] == "BUG"]
        clean_fires = [r["verdicts"][proto_id] for r in rows
                       if r["kind"] == "ok"]
        bug_count = sum(1 for x in bug_fires if isinstance(x, int) and x > 0)
        clean_count = sum(1 for x in clean_fires if isinstance(x, int) and x > 0)
        print(f"  {proto_id}")
        print(f"    bug-fire: {bug_count}/{len(bug_fires)}    "
              f"clean-fire: {clean_count}/{len(clean_fires)}    "
              f"(bug counts: {bug_fires}; clean counts: {clean_fires})")


if __name__ == "__main__":
    main()
