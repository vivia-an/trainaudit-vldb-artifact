"""Mining pipeline L1-L4 baseline: feed L1's hypotheses (already obtained from
Agent-as-LLM) through L2 enumerate → L3 validate → L4 filter (heuristic stub)
on real healthy traces from our novel-hunt iteration. Reports survival counts
and the surviving predicate texts.

Hypotheses input is hard-coded from the Agent run for reproducibility.
"""
import json
import sys
from pathlib import Path

_REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
sys.path.insert(0, str(_REPO / "trainaudit"))

from trainaudit.mining.hypothesis_schema import Hypothesis, RelationType
from trainaudit.mining.layer2_enumerate import enumerate_predicates, schema_introspect
from trainaudit.mining.layer3_validate import validate_against_healthy
from trainaudit.store import TraceStore


# ---------- L1 output (from the Agent invocation, see git log) ----------
import os as _os
_HYP_SET = _os.environ.get("HYP_SET", "ds_grad")

if _HYP_SET == "ds_grad":
    L1_HYPOTHESES_RAW = [
        {"id": "H1", "relation_type": "payload_field_compare",
         "entities": ["comm.pre", "comm.post"],
         "dimensions": ["tensor.dtype", "communication_data_type"],
         "rationale": "comm dtype roundtrip — if comm_dtype != tensor.dtype the "
                      "post-copy must restore original; mismatch indicates silent "
                      "precision corruption."},
        {"id": "H2", "relation_type": "tensor_stat_bound",
         "entities": ["comm.post:all_reduce"],
         "dimensions": ["output.l2_norm", "dp_world_size",
                        "sequence_parallel_size", "gradient_predivide_factor"],
         "rationale": "post-allreduce gradient norm must match expected scale "
                      "by sequence_parallel_size/dp_world_size."},
        {"id": "H3", "relation_type": "cross_rank_equal",
         "entities": ["comm.post:all_reduce"],
         "dimensions": ["output.cksum", "communication_data_type", "bucket_id"],
         "rationale": "after all_reduce, every DP rank must hold identical "
                      "reduced bytes."},
        {"id": "H4", "relation_type": "structural_presence",
         "entities": ["average_tensor:rank_and_offsets"],
         "dimensions": ["partition_id", "numel", "tensor.numel"],
         "rationale": "sum of per-slice numel must equal tensor.numel and every "
                      "partition_id must be < world_size."},
        {"id": "H5", "relation_type": "conditional_check",
         "entities": ["gradient_reduction_w_predivide"],
         "dimensions": ["postscale_gradients", "gradient_predivide_factor",
                        "dp_world_size", "sequence_parallel_size"],
         "rationale": "div by 0/NaN scale path silently corrupts gradients."},
        {"id": "H6", "relation_type": "cross_step_monotonic",
         "entities": ["optim.step.pre"],
         "dimensions": ["total_grad_l2", "step"],
         "rationale": "across consecutive steps, total_grad_l2 should not jump "
                      "by orders of magnitude (process_group misrouting signal)."},
    ]
elif _HYP_SET == "olmo_block":
    L1_HYPOTHESES_RAW = [
        {"id": "H1", "relation_type": "tensor_stat_bound",
         "entities": ["module.fwd.post:post_attention_norm",
                      "module.fwd.post:post_feed_forward_norm"],
         "dimensions": ["output.l2_norm"],
         "rationale": "Peri-LN tail norm output must have bounded magnitude "
                      "~sqrt(d_model)."},
        {"id": "H2", "relation_type": "conditional_check",
         "entities": ["module.fwd.post:attention_norm",
                      "module.fwd.post:feed_forward_norm"],
         "dimensions": ["is_normalizer", "output.l2_norm", "block_idx"],
         "rationale": "ReorderedNorm at block_idx==0: post-norm output l2_norm "
                      "must be non-trivially > epsilon (block-0 underflow)."},
        {"id": "H3", "relation_type": "cross_step_monotonic",
         "entities": ["module.fwd.post"],
         "dimensions": ["output.abs_max", "step"],
         "rationale": "Residual stream abs_max should grow only sub-linearly "
                      "across steps; sudden jump signals residual blowup."},
        {"id": "H4", "relation_type": "payload_field_compare",
         "entities": ["module.fwd.pre:post_attention_norm",
                      "module.fwd.post:post_attention_norm"],
         "dimensions": ["input.l2_norm", "output.l2_norm"],
         "rationale": "Peri-LN tail norm must compress scale: output l2 <= "
                      "input l2 * small_factor."},
        {"id": "H5", "relation_type": "cross_rank_equal",
         "entities": ["module.fwd.post:post_feed_forward_norm"],
         "dimensions": ["output.l2_norm", "output.has_nan"],
         "rationale": "After TP allreduce on post_feed_forward_norm, output "
                      "stats must be equal across TP ranks within DP group."},
        {"id": "H6", "relation_type": "structural_presence",
         "entities": ["module.fwd.post:post_attention_norm",
                      "module.fwd.post:post_feed_forward_norm"],
         "dimensions": ["module_name", "count_per_step"],
         "rationale": "Peri-LN tail norms must fire once per block per step; "
                      "missing events imply silent fallback to pre-norm."},
    ]
elif _HYP_SET == "ds_scheduler":
    L1_HYPOTHESES_RAW = [
        {"id": "H1", "relation_type": "cross_step_monotonic",
         "entities": ["scheduler.step.post"],
         "dimensions": ["last_batch_iteration"],
         "rationale": "last_batch_iteration must increase by 1 per step; "
                      "regressions = step counter desync."},
        {"id": "H2", "relation_type": "conditional_check",
         "entities": ["scheduler.step.post"],
         "dimensions": ["last_batch_iteration", "param_groups[*].lr",
                        "warmup_num_steps"],
         "rationale": "During warmup lr must be non-decreasing."},
        {"id": "H3", "relation_type": "tensor_stat_bound",
         "entities": ["scheduler.step.post"],
         "dimensions": ["param_groups[*].lr", "initial_lr",
                        "cos_min_ratio", "warmup_min_ratio"],
         "rationale": "lr in [min*initial_lr, initial_lr]."},
        {"id": "H4", "relation_type": "structural_presence",
         "entities": ["scheduler.init", "optim.step.pre"],
         "dimensions": ["param_groups[*].initial_lr"],
         "rationale": "every param_group must have initial_lr after resume."},
        {"id": "H5", "relation_type": "cross_rank_equal",
         "entities": ["scheduler.step.post"],
         "dimensions": ["last_batch_iteration", "param_groups[*].lr"],
         "rationale": "scheduler state must be identical across ranks."},
        {"id": "H6", "relation_type": "payload_field_compare",
         "entities": ["scheduler.step.post"],
         "dimensions": ["param_groups[*].lr", "param_groups[*].initial_lr"],
         "rationale": "lr/initial_lr ratio equal across all param_groups."},
    ]
elif _HYP_SET == "megatron_router":
    L1_HYPOTHESES_RAW = [
        {"id": "H1", "relation_type": "tensor_stat_bound",
         "entities": ["module.fwd.post"],
         "dimensions": ["routing_load_imbalance"],
         "rationale": "routing entropy floor — no >95%-on-one-expert."},
        {"id": "H2", "relation_type": "structural_presence",
         "entities": ["module.fwd.post"],
         "dimensions": ["top_k", "moe_router_topk"],
         "rationale": "runtime indices.shape[-1] == configured top_k."},
        {"id": "H3", "relation_type": "conditional_check",
         "entities": ["module.fwd.post", "build.snapshot"],
         "dimensions": ["routing_type", "moe_aux_loss_coeff"],
         "rationale": "non-sinkhorn + coeff==0 silently disables LB."},
        {"id": "H4", "relation_type": "cross_rank_equal",
         "entities": ["build.snapshot"],
         "dimensions": ["moe_router_topk", "num_moe_experts", "routing_type"],
         "rationale": "router config equal across EP/TP ranks."},
        {"id": "H5", "relation_type": "tensor_stat_bound",
         "entities": ["module.fwd.post"],
         "dimensions": ["routing_load_imbalance"],
         "rationale": "load imbalance < threshold (~4 * num_experts/topk)."},
        {"id": "H6", "relation_type": "cross_step_monotonic",
         "entities": ["build.snapshot"],
         "dimensions": ["expert_bias", "step"],
         "rationale": "expert_bias must update across optim.step events."},
    ]
elif _HYP_SET == "ds_clipgrad":
    L1_HYPOTHESES_RAW = [
        {"id": "H1", "relation_type": "tensor_stat_bound",
         "entities": ["utils.clip_grad.post"],
         "dimensions": ["clip_coef"],
         "rationale": "clip_coef in (0, 1] — values > 1 indicate inverted min/max."},
        {"id": "H2", "relation_type": "payload_field_compare",
         "entities": ["utils.clip_grad.post"],
         "dimensions": ["post_norm", "max_norm"],
         "rationale": "post-clip grad norm must not exceed max_norm."},
        {"id": "H3", "relation_type": "cross_rank_equal",
         "entities": ["utils.clip_grad.post"],
         "dimensions": ["total_grad_l2"],
         "rationale": "post-allreduce total_norm identical across ranks."},
        {"id": "H4", "relation_type": "conditional_check",
         "entities": ["utils.clip_grad.pre", "utils.clip_grad.post"],
         "dimensions": ["pre_norm", "clip_coef"],
         "rationale": "if pre_norm<=max_norm then clip_coef==1; else "
                      "clip_coef==max_norm/(pre_norm+eps)."},
        {"id": "H5", "relation_type": "tensor_stat_bound",
         "entities": ["utils.clip_grad.post"],
         "dimensions": ["total_grad_l2", "clip_coef"],
         "rationale": "total_grad_l2 finite and non-neg, clip_coef finite."},
        {"id": "H6", "relation_type": "structural_presence",
         "entities": ["comm.pre", "comm.post"],
         "dimensions": ["all_reduce_on_dp_group"],
         "rationale": "exactly one SUM all_reduce on DP group between "
                      "clip_grad.pre and clip_grad.post."},
    ]
else:
    raise ValueError(f"Unknown HYP_SET={_HYP_SET}")


HEALTHY_TRACES = [
    "benchmark/eval/hunt_log/novel_hunt/olmo_core_moe_hybrid/trace_rank0.duckdb",  # 0/25 fire
    "benchmark/eval/hunt_log/novel_hunt/olmo_core_smallmoe/trace_rank0.duckdb",     # 0/25 fire
    "benchmark/eval/hunt_log/novel_hunt/olmo_core_olmoe/trace_rank0.duckdb",        # 0/25 fire
]


def _hyp_from_raw(raw):
    return Hypothesis(
        relation_type=RelationType(raw["relation_type"]),
        entities=raw["entities"],
        dimensions=raw["dimensions"],
        rationale=raw["rationale"],
    )


def main():
    print("=== L1: 6 hypotheses input ===")
    for h in L1_HYPOTHESES_RAW:
        print(f"  {h['id']}: {h['relation_type']:25s} | {h['rationale'][:80]}")

    # Open healthy traces
    stores = []
    for p in HEALTHY_TRACES:
        full = _REPO / p
        if not full.exists():
            print(f"  WARN: {full} missing")
            continue
        s = TraceStore(str(full))
        stores.append((str(full), s))
    print(f"\nOpened {len(stores)} healthy trace(s)")

    # Use the first store for schema introspection
    if stores:
        schema = schema_introspect(stores[0][1])
        print(f"\nSchema introspection: {len(schema)} hookpoints, "
              f"{sum(len(v) for v in schema.values())} total fields")
        for hp, fields in sorted(schema.items())[:8]:
            print(f"  {hp}: {len(fields)} fields")
    else:
        schema = {}

    print("\n=== L2: enumerate predicates per hypothesis ===")
    all_predicates = []
    for raw in L1_HYPOTHESES_RAW:
        try:
            hyp = _hyp_from_raw(raw)
            preds = enumerate_predicates(hyp, schema)
            print(f"  {raw['id']} ({raw['relation_type']:25s}): "
                  f"{len(preds)} candidate predicates")
            for p in preds[:3]:
                print(f"      - {p.id}: {p.description[:80]}")
            all_predicates.extend((raw["id"], p) for p in preds)
        except Exception as e:
            print(f"  {raw['id']}: ERROR {type(e).__name__}: {e}")

    print(f"\nTotal candidate predicates from L2: {len(all_predicates)}")

    print("\n=== L3: validate predicates against healthy traces ===")
    accepted = []
    rejected = []
    healthy_only = [s for _, s in stores]
    for hyp_id, p in all_predicates:
        result = validate_against_healthy(p, healthy_only,
                                           max_allowed_violations=0)
        if result.accepted:
            accepted.append((hyp_id, p))
        else:
            rejected.append((hyp_id, p, result.rejection_reason))
    print(f"  accepted: {len(accepted)}/{len(all_predicates)}")
    print(f"  rejected: {len(rejected)}/{len(all_predicates)}")
    if rejected:
        print("\n  reject reasons sample:")
        for hyp_id, p, reason in rejected[:5]:
            print(f"    [{hyp_id}] {p.id}: {reason[:100]}")

    print("\n=== L3 accepted predicates (would feed L4 LLM filter) ===")
    for hyp_id, p in accepted:
        print(f"  [{hyp_id}] {p.id}")
        print(f"      desc: {p.description[:120]}")
        print(f"      template: {p.template.value}, family: {p.family}")
        if p.bound:
            print(f"      bound: {p.bound}")

    # Output summary
    out = {
        "n_hypotheses": len(L1_HYPOTHESES_RAW),
        "n_predicates_l2": len(all_predicates),
        "n_predicates_accepted_l3": len(accepted),
        "n_predicates_rejected_l3": len(rejected),
        "schema_hookpoints": list(schema.keys()),
        "schema_field_count": sum(len(v) for v in schema.values()),
        "accepted": [
            {"hyp_id": hyp_id, "pred_id": p.id, "desc": p.description,
             "template": p.template.value, "family": p.family}
            for hyp_id, p in accepted
        ],
        "rejected_summary": [
            {"hyp_id": hyp_id, "pred_id": p.id, "reason": reason}
            for hyp_id, p, reason in rejected
        ],
    }
    out_path = (_REPO / "benchmark" / "eval"
                / f"mining_baseline_result_{_HYP_SET}.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
