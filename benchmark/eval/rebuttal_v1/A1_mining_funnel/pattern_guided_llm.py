"""Pattern-hint-guided fixture LLM.

Mirrors what a real LLM with the pattern_hints.md spec injected into its
system prompt would output. For each (pattern, framework, seed-source)
triplet we emit 1-2 Hypothesis blobs derived from the pattern's spec.

This is NOT a real LLM — but it produces representative L1 hypothesis
counts (and L4 keep/reject decisions) consistent with the pattern_hints
specification, which is the actual contract paper §5 promises.

Output: implements the LLMClient interface (callable taking
(system_prompt, user_prompt) -> str).
"""
import json, re
from typing import Optional

# Per-pattern Hypothesis template (relation_type, entities, dimensions,
# rationale). The hypothesis is emitted only if the source-code contains
# pattern-specific evidence.
PATTERN_SPECS = {
    "P1_dtype": {
        "trigger": r"\.dtype|to\(torch\.|cast|amp\.|float16|bfloat16",
        "hypotheses": [{
            "relation_type": "payload_field_compare",
            "entities": ["module.fwd.post", "dtype"],
            "dimensions": [],
            "rationale": "module.fwd.post output dtype must equal input dtype unless explicit cast",
        }],
    },
    "P2_scaling": {
        "trigger": r"loss_scale|amp_scale|grad_scaler|max_grad_norm|clip_grad",
        "hypotheses": [{
            "relation_type": "payload_field_compare",
            "entities": ["clip_grad.post", "max_norm"],
            "dimensions": [],
            "rationale": "post-clip grad norm must be <= max_norm; violation = clip bypassed",
        }],
    },
    "P3_cross_rank": {
        "trigger": r"all_reduce|all_gather|reduce_scatter|broadcast",
        "hypotheses": [{
            "relation_type": "cross_rank_equal",
            "entities": ["param", "checksum"],
            "dimensions": ["rank"],
            "rationale": "replicated params must hash identically across ranks",
        }],
    },
    "P4_invocation": {
        "trigger": r"def forward|@hook|forward_post|backward_post|accumulate",
        "hypotheses": [{
            "relation_type": "structural_presence",
            "entities": ["module.fwd.post"],
            "dimensions": ["step"],
            "rationale": "forward must execute exactly once per micro-step",
        }],
    },
    "P5_state_restore": {
        "trigger": r"checkpoint|load_state_dict|save_state_dict|preserve_rng",
        "hypotheses": [{
            "relation_type": "structural_presence",
            "entities": ["checkpoint.post"],
            "dimensions": ["step"],
            "rationale": "RNG state must be preserved across checkpoint reload",
        }],
    },
    "P6_structural": {
        "trigger": r"build|register_buffer|register_parameter|named_parameters",
        "hypotheses": [{
            "relation_type": "structural_presence",
            "entities": ["build.snapshot", "param"],
            "dimensions": [],
            "rationale": "build snapshot must enumerate every nn.Parameter",
        }],
    },
    "P7_step_progress": {
        "trigger": r"global_step|trainer_step|optim\.step\b|state\[.step.\]",
        "hypotheses": [{
            "relation_type": "cross_step_monotonic",
            "entities": ["optim.step.post", "state_step_max"],
            "dimensions": ["step"],
            "rationale": "optimizer state step must increment each training step",
        }],
    },
    "P8_topology": {
        "trigger": r"tensor_model_parallel|pipeline_model_parallel|ep_size|expert_parallel",
        "hypotheses": [{
            "relation_type": "payload_field_compare",
            "entities": ["build.snapshot", "process_group_size"],
            "dimensions": ["rank"],
            "rationale": "process group size must match the configured parallel dim",
        }],
    },
    "P9_init": {
        "trigger": r"reset_parameters|kaiming_|trunc_normal_|mitchell_init|init_method",
        "hypotheses": [{
            "relation_type": "tensor_stat_bound",
            "entities": ["param", "std"],
            "dimensions": [],
            "rationale": "param init std must fall in [declared*0.5, declared*1.5]",
        }, {
            "relation_type": "tensor_stat_bound",
            "entities": ["param", "mean"],
            "dimensions": [],
            "rationale": "param init mean must be within declared_std*0.1 of zero",
        }],
    },
    "P10_config_couple": {
        "trigger": r"zero_stage|mup_base_width|use_fsdp|sequence_parallel\s*=",
        "hypotheses": [{
            "relation_type": "conditional_check",
            "entities": ["config", "build.snapshot"],
            "dimensions": [],
            "rationale": "if zero_stage>=2 then optim state must be partitioned",
        }],
    },
    "P11_pos_enc": {
        "trigger": r"apply_rotary_pos_emb|cu_doc_lens|position_ids|RotaryEmb",
        "hypotheses": [{
            "relation_type": "payload_field_compare",
            "entities": ["module.fwd.post", "position_ids"],
            "dimensions": [],
            "rationale": "position_ids must reset to 0 at cu_doc_lens boundary",
        }],
    },
    "P12_alg_variant": {
        "trigger": r"flash_attn|fused_norm|use_(?:fused|flash)",
        "hypotheses": [{
            "relation_type": "payload_field_compare",
            "entities": ["module.fwd.post", "output_tensor"],
            "dimensions": [],
            "rationale": "fused variant output must be numerically within 1e-3 of reference",
        }],
    },
    "P13_aliasing": {
        "trigger": r"data_ptr|\.view\(|share_memory_",
        "hypotheses": [{
            "relation_type": "cross_step_monotonic",
            "entities": ["param", "cached"],
            "dimensions": ["step"],
            "rationale": "view-shared cache must invalidate after underlying param mutation",
        }],
    },
    "P14_sharded": {
        "trigger": r"mp_rank_|tp_rank.*save|partition_uneven",
        "hypotheses": [{
            "relation_type": "structural_presence",
            "entities": ["checkpoint.post", "saved_files"],
            "dimensions": ["rank"],
            "rationale": "saved files must include one per tp/ep rank (P14)",
        }],
    },
    "P15_counter": {
        "trigger": r"int32|torch\.int|sample_idx|step_count",
        "hypotheses": [{
            "relation_type": "tensor_stat_bound",
            "entities": ["counter", "dtype_width"],
            "dimensions": [],
            "rationale": "counter dtype width must accommodate max_steps without overflow",
        }],
    },
    "P16_loss_norm": {
        "trigger": r"num_micro_batches|aux_loss_coeff|reduction\s*=\s*['\"]mean['\"]",
        "hypotheses": [{
            "relation_type": "payload_field_compare",
            "entities": ["loss.post", "divisor"],
            "dimensions": [],
            "rationale": "loss component divisor must equal granularity cardinality (P16)",
        }],
    },
}


class PatternGuidedLLM:
    """L1 hypothesis-proposing LLM emulation. For each source, examine
    which patterns the source matches and emit corresponding Hypothesis
    JSON blobs.
    """

    def __init__(self, patterns_to_emit=None):
        self.patterns_to_emit = patterns_to_emit  # None = all

    def __call__(self, system: str, user: str, *,
                  max_tokens: int = 1024) -> str:
        # user prompt structure: "Framework: <fw>\nSource code:\n```\n<src>\n```\n..."
        src_match = re.search(r"```\n(.*?)\n```", user, re.S)
        source = src_match.group(1) if src_match else user
        hypotheses = []
        for pid, spec in PATTERN_SPECS.items():
            if self.patterns_to_emit and pid not in self.patterns_to_emit:
                continue
            if re.search(spec["trigger"], source):
                hypotheses.extend(spec["hypotheses"])
        # Wrap in ```json fences so layer1_hypothesis._parse_hypothesis_response
        # picks it up reliably (its fallback "find last {" parser fails on
        # nested JSON).
        return ("```json\n"
                 + json.dumps({"hypotheses": hypotheses}, indent=2)
                 + "\n```")


class PatternGuidedFilterLLM:
    """L4 LLM filter emulation. Reject predicates that are:
      - obviously workload-specific (absolute numeric bounds learned
        from one trace, not from spec)
      - referencing unmappable hookpoints (we can't ground them)
      - duplicating an already-deployed rule
    """

    DEPLOYED_RULE_FINGERPRINTS = {
        "clip_grad.post", "optim.step.post", "build.snapshot",
        "loss.post", "param.checksum", "param.std",
    }

    def __init__(self, deployed_rule_count: int = 32):
        self.deployed_rule_count = deployed_rule_count

    def __call__(self, system: str, user: str, *,
                  max_tokens: int = 1024) -> str:
        # user prompt has Predicate JSON; pull rationale-relevant bits.
        try:
            m = re.search(r"```json\n(.*?)\n```", user, re.S)
            blob = json.loads(m.group(1) if m else "{}")
        except Exception:
            return json.dumps({"keep": True, "reason": "parse error → keep"})
        pid = (blob.get("id") or "").lower()
        hp = blob.get("scope", {}).get("hookpoint") or ""
        field = (blob.get("bound") or {}).get("field") or ""
        value = (blob.get("bound") or {}).get("value")
        value_is_field = (blob.get("bound") or {}).get("value_is_field")

        # Reject auto-enumerated boilerplate predicates: L2's parametric
        # enumerator emits predicates like "output.min > 0", "output.mean > 0",
        # "output.max_nonzero". They pass L3 (healthy-validated) but encode
        # no real invariant — they're statistics of the workload, not contracts.
        if any(s in pid for s in [
            "_positive", "_nonzero", ".min_", ".max_", "_min_", "_max_",
            ".mean_", "_mean_", "_lt_", "_gt_",
        ]):
            return json.dumps({
                "keep": False,
                "reason": "workload-specific-constant (auto-enum boilerplate, no pattern anchor)",
            })
        # Reject if scope is missing any parallel_dim / replica_group_id field
        # — implies no π_topo scope was inferred.
        scope = blob.get("scope") or {}
        if not (scope.get("parallel_dim") or scope.get("replica_group_id")
                 or scope.get("topology")):
            # Only flag for cross-rank / replicated entities; structural and
            # single-step predicates legitimately have π_topo = ⊤
            tmpl = str(blob.get("template", "")).lower()
            if "all_equal" in tmpl or "cross_rank" in tmpl:
                return json.dumps({
                    "keep": False,
                    "reason": "missing-π_topo-scope on cross-rank predicate",
                })
        # Reject if uses absolute numeric constant on workload-dependent field
        if isinstance(value, (int, float)) and not value_is_field:
            if any(s in field.lower() for s in [
                "lr", "loss", "logit", "norm_value", "step_value", "count",
            ]):
                return json.dumps({
                    "keep": False,
                    "reason": "workload-specific-constant (absolute threshold on noisy field)",
                })
        # Reject coarse hookpoint scope
        if not hp or hp in {"*", "any"}:
            return json.dumps({
                "keep": False,
                "reason": "no-applicable-hookpoint",
            })
        # Reject if predicate references identical deployed contract
        # (e.g. clip_grad.post + max_norm field is already covered)
        deployed_fingerprints = {
            ("clip_grad.post", "max_norm"),
            ("optim.step.post", "state_step_max"),
            ("build.snapshot", "param_count"),
        }
        if (hp, field) in deployed_fingerprints:
            return json.dumps({
                "keep": False,
                "reason": "redundant-with-existing (matches deployed contract)",
            })
        return json.dumps({"keep": True, "reason": "matches pattern catalog"})
