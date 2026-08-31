"""Fault injection harness: 33 synthetic faults across all 18 rules.

Closes doc 22 §5 blocking item for paper §4.1: a controlled benchmark
where each fault is a small, hand-crafted event sequence that emulates
exactly one silent-error pattern. Faults span:

  - severity (severe / moderate / subthreshold)
  - tier (T0 / T1)
  - rule family (numerical / dtype / gradient / replica / structural / mining)

Three intentional boundary cases (`bnd_*`) sit just inside tolerance and
SHOULD NOT fire — they characterise the system's sensitivity floor and
are the honest false-negative source for paper §4.1 narrative
("possible-observability boundary").

Output (when run as CLI):
  benchmark/eval/fault_injection.csv         per-fault verdict
  benchmark/eval/paper_table_fault_injection.md   paper §4.1 table
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_TA_ROOT = (_HERE / ".." / ".." / "trainaudit").resolve()
if str(_TA_ROOT) not in sys.path:
    sys.path.insert(0, str(_TA_ROOT))

import trainaudit  # noqa: E402


# ---- Recipe registry ----------------------------------------------------


@dataclass
class FaultRecipe:
    fault_id: str
    expected_rule: str          # rule that should fire ("" if subthreshold)
    severity: str               # severe | moderate | subthreshold
    category: str               # numerical | dtype | replica | structural ...
    tier: str                   # T0 / T1
    description: str
    events: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)


_REGISTRY: List[FaultRecipe] = []


def _recipe(fault_id, expected_rule, severity, category, tier, description):
    def deco(fn):
        events = fn()
        _REGISTRY.append(FaultRecipe(
            fault_id=fault_id, expected_rule=expected_rule,
            severity=severity, category=category, tier=tier,
            description=description, events=events,
        ))
        return fn
    return deco


# ---- T0-no-nan-inf (3 variants) -----------------------------------------


@_recipe("nan_in_fwd_post", "T0-no-nan-inf", "severe",
         "numerical", "T0",
         "NaN in module forward output tensor")
def _f1():
    return [("module.fwd.post", {
        "module_class": "Linear", "module_id": 1, "module_name": "head",
        "is_normalizer": False, "training": True,
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 1.0, "has_nan": True, "has_inf": False,
                   "abs_max": 1.0, "max": 1.0, "min": -1.0, "mean": 0.0},
    })]


@_recipe("inf_in_optim_pre", "T0-no-nan-inf", "severe",
         "numerical", "T0",
         "Inf in optimizer step pre-event param")
def _f2():
    return [("optim.step.pre", {
        "kind": "optim_step", "optimizer_class": "AdamW",
        "param_groups": [], "n_params": 1,
        "total_grad_l2": 1.0, "total_param_l2": 1.0,
        "params": [{"dtype": "torch.bfloat16", "shape": [16],
                     "l2_norm": float("inf"), "has_nan": False,
                     "has_inf": True, "abs_max": 0.0, "max": 0.0,
                     "min": 0.0, "mean": 0.0}],
    })]


@_recipe("nan_in_comm_post", "T0-no-nan-inf", "severe",
         "numerical", "T0",
         "NaN in distributed all-reduce output tensor")
def _f3():
    return [("comm.post", {
        "kind": "comm", "op": "all_reduce", "group_size": 4,
        "tensor_post": {"dtype": "torch.float32", "shape": [16],
                         "l2_norm": 1.0, "has_nan": True, "has_inf": False,
                         "abs_max": 1.0, "max": 1.0, "min": -1.0, "mean": 0.0},
    })]


# ---- T0-clip-grad-bounded (3 severities) --------------------------------


@_recipe("clip_2x_violation", "T0-clip-grad-bounded", "moderate",
         "gradient", "T0",
         "post_norm 2x max_norm — moderate clip bypass")
def _f4():
    return [("utils.clip_grad.post", {
        "kind": "clip_grad", "fn": "torch.nn.utils.clip_grad_norm_",
        "max_norm": 1.0, "pre_norm": 2.0, "post_norm": 2.0, "ratio": 1.0,
    })]


@_recipe("clip_10x_violation", "T0-clip-grad-bounded", "severe",
         "gradient", "T0",
         "post_norm 10x max_norm — severe clip bypass")
def _f5():
    return [("utils.clip_grad.post", {
        "kind": "clip_grad", "fn": "torch.nn.utils.clip_grad_norm_",
        "max_norm": 0.1, "pre_norm": 1.0, "post_norm": 1.0, "ratio": 1.0,
    })]


@_recipe("clip_100x_violation", "T0-clip-grad-bounded", "severe",
         "gradient", "T0",
         "post_norm 100x max_norm — extreme clip bypass (B11 fingerprint)")
def _f6():
    return [("utils.clip_grad.post", {
        "kind": "clip_grad", "fn": "torch.nn.utils.clip_grad_norm_",
        "max_norm": 0.01, "pre_norm": 1.0, "post_norm": 1.0, "ratio": 1.0,
    })]


# ---- T0-optim-lr-positive (2) --------------------------------------------


@_recipe("lr_zero", "T0-optim-lr-positive", "severe",
         "optimizer", "T0",
         "AdamW param_group with lr=0 — optimizer disabled")
def _f7():
    return [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "optimizer": {"param_groups": [
            {"index": 0, "lr": 0.0, "betas": [0.9, 0.95]}]},
    })]


@_recipe("lr_negative", "T0-optim-lr-positive", "severe",
         "optimizer", "T0",
         "AdamW param_group with lr<0 — sign-flipped updates")
def _f8():
    return [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "optimizer": {"param_groups": [
            {"index": 0, "lr": -1e-4, "betas": [0.9, 0.95]}]},
    })]


# ---- T0-build-has-modules (2) -------------------------------------------


@_recipe("zero_params_in_build", "T0-build-has-modules", "severe",
         "structural", "T0",
         "Build snapshot reports zero parameters")
def _f9():
    return [("build.snapshot", {
        "model": {"n_parameters": 0, "n_modules": 5},
    })]


@_recipe("zero_modules_in_build", "T0-build-has-modules", "severe",
         "structural", "T0",
         "Build snapshot reports zero modules")
def _f10():
    return [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 0},
    })]


# ---- T0-initial-lr-present (2) -------------------------------------------


@_recipe("scheduler_resume_no_initial_lr", "T0-initial-lr-present", "severe",
         "lr_schedule", "T0",
         "scheduler.init last_epoch=10 + has_initial_lr=False (B12)")
def _f11():
    return [("scheduler.init", {
        "kind": "scheduler_init",
        "scheduler_class": "CosineAnnealingLR", "optimizer_class": "AdamW",
        "last_epoch": 10,
        "param_groups": [{"index": 0, "has_initial_lr": False, "lr": 1e-4,
                           "keys": ["lr"]}],
    })]


@_recipe("scheduler_partial_initial_lr", "T0-initial-lr-present", "moderate",
         "lr_schedule", "T0",
         "scheduler.init resume + 1 of 2 groups missing initial_lr")
def _f12():
    return [("scheduler.init", {
        "kind": "scheduler_init",
        "scheduler_class": "CosineAnnealingLR", "optimizer_class": "AdamW",
        "last_epoch": 5,
        "param_groups": [
            {"index": 0, "has_initial_lr": True, "lr": 1e-4, "keys": ["lr", "initial_lr"]},
            {"index": 1, "has_initial_lr": False, "lr": 1e-3, "keys": ["lr"]},
        ],
    })]


# ---- T0-token-id-in-vocab (2) -------------------------------------------


@_recipe("token_id_overflow", "T0-token-id-in-vocab", "severe",
         "data_loading", "T0",
         "DataLoader batch with absurdly large token id (5M, O-NEW-9)")
def _f13():
    return [("dataloader.batch", {
        "kind": "data_load",
        "input_ids": {"dtype": "torch.int64", "shape": [2, 16],
                       "min": 0, "max": 5_000_000,
                       "has_nan": False, "has_inf": False,
                       "abs_max": 5_000_000.0, "mean": 100.0,
                       "l2_norm": 1.0},
    })]


@_recipe("token_id_extreme", "T0-token-id-in-vocab", "severe",
         "data_loading", "T0",
         "DataLoader batch with int64 max id (2^31)")
def _f14():
    return [("dataloader.batch", {
        "kind": "data_load",
        "input_ids": {"dtype": "torch.int64", "shape": [2, 16],
                       "min": 0, "max": 2147483647,
                       "has_nan": False, "has_inf": False,
                       "abs_max": 2147483647.0, "mean": 1000.0,
                       "l2_norm": 1.0},
    })]


# ---- T0-norm-output-unit-rms (3 magnitudes) ------------------------------


def _norm_event(rms_target: float):
    """rms = l2 / sqrt(numel); shape [8,4] → numel=32, l2 = rms*sqrt(32)."""
    import math
    l2 = rms_target * math.sqrt(32)
    return ("module.fwd.post", {
        "module_class": "RMSLayerNorm", "module_id": 1,
        "module_name": "blocks.0.attn_norm",
        "is_normalizer": True, "training": True,
        "output": {"dtype": "torch.float32", "shape": [8, 4],
                   "l2_norm": l2, "has_nan": False, "has_inf": False,
                   "abs_max": rms_target * 2, "max": rms_target * 2,
                   "min": -rms_target * 2, "mean": 0.0},
    })


@_recipe("norm_rms_low", "T0-norm-output-unit-rms", "severe",
         "normalization", "T0",
         "RMSNorm output rms ≈ 0.33 (O-NEW-1 fingerprint)")
def _f15():
    return [_norm_event(0.33)]


@_recipe("norm_rms_high", "T0-norm-output-unit-rms", "moderate",
         "normalization", "T0",
         "RMSNorm output rms ≈ 3.0 (over-amplified)")
def _f16():
    return [_norm_event(3.0)]


@_recipe("norm_rms_extreme", "T0-norm-output-unit-rms", "severe",
         "normalization", "T0",
         "RMSNorm output rms ≈ 10.0 (numerical instability ahead)")
def _f17():
    return [_norm_event(10.0)]


# ---- T0-softmax-degenerate (2) -------------------------------------------


@_recipe("router_size1_softmax", "T0-softmax-degenerate", "severe",
         "moe_router", "T0",
         "TopKRouter (4,1) softmax — every row is 1.0 (M-014)")
def _f18():
    return [("module.fwd.post", {
        "module_class": "TopKRouter", "module_id": 1, "module_name": "router",
        "is_normalizer": False, "training": True,
        "output": {"dtype": "torch.float32", "shape": [4, 1],
                   "l2_norm": 2.0, "has_nan": False, "has_inf": False,
                   "abs_max": 1.0, "max": 1.0, "min": 1.0, "mean": 1.0},
    })]


@_recipe("functional_softmax_one_hot", "T0-softmax-degenerate", "severe",
         "moe_router", "T0",
         "F.softmax of post-topk(1) tensor — one-hot signature")
def _f19():
    return [("functional.softmax", {
        "kind": "functional", "fn": "F.softmax", "dim": -1,
        "output": {"dtype": "torch.float32", "shape": [16, 1],
                   "l2_norm": 4.0, "has_nan": False, "has_inf": False,
                   "abs_max": 1.0, "max": 1.0, "min": 1.0, "mean": 1.0},
    })]


# ---- T0-optim-step-counter-monotonic (2) ---------------------------------


@_recipe("step_frozen", "T0-optim-step-counter-monotonic", "severe",
         "optimizer", "T0",
         "state['step'] frozen at 5 across calls (OC-NEW-2 fingerprint)")
def _f20():
    return [
        ("optim.step.post", {"kind": "optim_step",
                              "optimizer_class": "SkipStepAdamW",
                              "state_step_min": 5.0, "state_step_max": 5.0,
                              "state_step_n_params": 4, "total_param_l2": 1.0}),
        ("optim.step.post", {"kind": "optim_step",
                              "optimizer_class": "SkipStepAdamW",
                              "state_step_min": 5.0, "state_step_max": 5.0,
                              "state_step_n_params": 4, "total_param_l2": 1.0}),
        ("optim.step.post", {"kind": "optim_step",
                              "optimizer_class": "SkipStepAdamW",
                              "state_step_min": 5.0, "state_step_max": 5.0,
                              "state_step_n_params": 4, "total_param_l2": 1.0}),
    ]


@_recipe("step_regress", "T0-optim-step-counter-monotonic", "severe",
         "optimizer", "T0",
         "state['step'] regresses from 10 → 8 (counter rollback)")
def _f21():
    return [
        ("optim.step.post", {"kind": "optim_step",
                              "optimizer_class": "AdamW",
                              "state_step_min": 10.0, "state_step_max": 10.0,
                              "state_step_n_params": 4, "total_param_l2": 1.0}),
        ("optim.step.post", {"kind": "optim_step",
                              "optimizer_class": "AdamW",
                              "state_step_min": 8.0, "state_step_max": 8.0,
                              "state_step_n_params": 4, "total_param_l2": 1.0}),
    ]


# ---- T0-checkpoint-preserve-rng (1) -------------------------------------


@_recipe("checkpoint_no_rng_with_dropout", "T0-checkpoint-preserve-rng",
         "severe", "checkpoint", "T0",
         "checkpoint(preserve_rng_state=False) with Dropout module (O-005)")
def _f22():
    return [
        ("build.snapshot", {
            "model": {"n_parameters": 100, "n_modules": 3,
                       "parameters": [{"name": "blk.0.dropout"}]},
        }),
        ("module.fwd.post", {
            "module_class": "Dropout", "module_id": 1,
            "module_name": "blk.0.dropout",
            "is_normalizer": False, "training": True,
            "output": {"dtype": "torch.float32", "shape": [4, 8],
                       "l2_norm": 1.0, "has_nan": False, "has_inf": False,
                       "abs_max": 1.0, "max": 1.0, "min": -1.0, "mean": 0.0},
        }),
        ("checkpoint.call", {
            "kind": "checkpoint", "function": "Block.forward",
            "preserve_rng_state": False, "use_reentrant": True,
            "kwargs_keys": ["preserve_rng_state", "use_reentrant"],
        }),
    ]


# ---- T1-replica-cksum-equal (3 patterns) ---------------------------------


@_recipe("replica_outlier_rank0", "T1-replica-cksum-equal", "severe",
         "replica", "T1",
         "router.weight diverged at rank 0 (4-rank group)")
def _f23():
    return [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {},
        "cross_rank_cksums": [{
            "name": "router.weight", "group_size": 4, "local_cksum": 999,
            "gathered_cksums": [999, 100, 100, 100], "all_equal": False,
        }],
    })]


@_recipe("replica_outlier_rank2", "T1-replica-cksum-equal", "severe",
         "replica", "T1",
         "router.weight diverged at rank 2 (4-rank group, M-005)")
def _f24():
    return [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {},
        "cross_rank_cksums": [{
            "name": "router.weight", "group_size": 4, "local_cksum": 100,
            "gathered_cksums": [100, 100, 999, 100], "all_equal": False,
        }],
    })]


@_recipe("replica_2rank_disagree", "T1-replica-cksum-equal", "moderate",
         "replica", "T1",
         "norm.weight diverged in 2-rank group — outlier ambiguous")
def _f25():
    return [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {},
        "cross_rank_cksums": [{
            "name": "blocks.0.norm.weight", "group_size": 2, "local_cksum": 100,
            "gathered_cksums": [100, 999], "all_equal": False,
        }],
    })]


# ---- T1 active-probe rules (4) ------------------------------------------


@_recipe("residual_clobbered", "T1-residual-stream-preserved",
         "severe", "residual_connection", "T1",
         "OLMo block output closer to normed than original input (B13)")
def _f26():
    return [("residual.probe", {
        "block_class": "OLMoSequentialBlock",
        "d_to_original_input": 5.0, "d_to_normed_input": 0.5,
    })]


@_recipe("router_missing_attr", "T1-router-has-calculate-per-token-loss",
         "moderate", "moe_router", "T1",
         "Megatron router missing calculate_per_token_loss when feature on")
def _f27():
    return [
        ("build.snapshot", {
            "model": {"n_parameters": 100, "n_modules": 5},
            "framework_invariants": {
                "megatron": {"calculate_per_token_loss": True},
            },
        }),
        ("module.fwd.post", {
            "module_class": "TopKRouter", "module_id": 1,
            "module_name": "router",
            "is_normalizer": False, "training": True,
            "semantic": {"is_router": True,
                          "has_calculate_per_token_loss": False},
            "output": {"dtype": "torch.float32", "shape": [4, 8],
                       "l2_norm": 1.0, "has_nan": False, "has_inf": False,
                       "abs_max": 1.0, "max": 1.0, "min": -1.0, "mean": 0.0},
        }),
    ]


@_recipe("expert_bias_bf16", "T1-expert-bias-fp32", "severe",
         "dtype", "T1",
         "TopKRouter.expert_bias silently demoted to bfloat16 (M-012)")
def _f28():
    return [("module.fwd.post", {
        "module_class": "TopKRouter", "module_id": 1, "module_name": "router",
        "is_normalizer": False, "training": True,
        "semantic": {"is_router": True, "expert_bias_dtype": "bfloat16"},
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 1.0, "has_nan": False, "has_inf": False,
                   "abs_max": 1.0, "max": 1.0, "min": -1.0, "mean": 0.0},
    })]


@_recipe("layer_count_mismatch", "T1-layer-count-strict", "moderate",
         "structural", "T1",
         "num_layers=24 % pp_size=5 != 0 (M-020)")
def _f29():
    return [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {
            "megatron": {
                "num_layers": 24,
                "pipeline_model_parallel_size": 5,
                "n_transformer_layers_in_local_module": 4,
            },
        },
    })]


@_recipe("jitter_promotes_dtype", "T1-jitter-preserves-dtype", "severe",
         "dtype", "T1",
         "Megatron apply_input_jitter promotes bf16 → fp32 (M-024)")
def _f30():
    return [("jitter.probe", {
        "module_class": "TopKRouter",
        "input_dtype": "bfloat16", "output_dtype": "float32",
        "dtypes_match": False,
    })]


@_recipe("sqrt_decay_inverted", "T1-sqrt-decay-front-loaded", "severe",
         "lr_schedule", "T1",
         "OLMo-core _sqrt_decay slope inverted (slow-then-fast, OC-NEW-3)")
def _f31():
    return [
        ("decay.probe", {"kind": "sqrt_decay", "progress": p, "result": r,
                          "initial_lr": 1.0, "decay_min_lr": 0.0})
        for p, r in [(0.0, 0.0), (0.1, 0.32), (0.3, 0.55),
                      (0.6, 0.77), (0.9, 0.95), (1.0, 1.0)]
    ]


# ---- Boundary cases (3 subthreshold — SHOULD NOT fire) ------------------


@_recipe("bnd_clip_within_tolerance", "", "subthreshold",
         "gradient", "T0",
         "post_norm = max_norm * 1.005 — within 1% rel tol")
def _bnd1():
    return [("utils.clip_grad.post", {
        "kind": "clip_grad", "fn": "torch.nn.utils.clip_grad_norm_",
        "max_norm": 1.0, "pre_norm": 1.005, "post_norm": 1.005, "ratio": 1.0,
    })]


@_recipe("bnd_norm_rms_at_boundary", "", "subthreshold",
         "normalization", "T0",
         "RMSNorm rms = 0.55 — just inside [0.5, 2.0] band")
def _bnd2():
    return [_norm_event(0.55)]


@_recipe("bnd_token_id_just_under", "", "subthreshold",
         "data_loading", "T0",
         "max_id = 1048576 (= 2^20 ceiling, edge inclusive)")
def _bnd3():
    return [("dataloader.batch", {
        "kind": "data_load",
        "input_ids": {"dtype": "torch.int64", "shape": [2, 16],
                       "min": 0, "max": 1048576,  # exactly 2^20
                       "has_nan": False, "has_inf": False,
                       "abs_max": 1048576.0, "mean": 100.0,
                       "l2_norm": 1.0},
    })]


# ---- Runner -------------------------------------------------------------


def run_one(recipe: FaultRecipe) -> Dict[str, Any]:
    """Emit recipe events into a fresh trace store, run rules, record verdict."""
    tier = (trainaudit.Tier.T1_FW_METADATA if recipe.tier == "T1"
            else trainaudit.Tier.T0_PYTORCH)
    with tempfile.TemporaryDirectory() as tmp:
        store = trainaudit.enable(tier=tier,
                                   db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            for hookpoint, payload in recipe.events:
                store.emit(hookpoint, payload)
            store.flush()
            results = trainaudit.run_rules()
        finally:
            trainaudit.disable()

    fired = sorted(r.rule_id for r in results if r.violated)
    if recipe.severity == "subthreshold":
        verdict = "TRUE_NEGATIVE" if not fired else "FALSE_POSITIVE"
        match = (verdict == "TRUE_NEGATIVE")
    else:
        match = recipe.expected_rule in fired
        verdict = "DETECTED" if match else "MISSED"
    return {
        "fault_id": recipe.fault_id,
        "expected_rule": recipe.expected_rule,
        "severity": recipe.severity,
        "category": recipe.category,
        "tier": recipe.tier,
        "verdict": verdict,
        "fired_rules": ";".join(fired),
        "description": recipe.description,
    }


def run_all() -> List[Dict[str, Any]]:
    return [run_one(r) for r in _REGISTRY]


def write_paper_table(rows: List[Dict[str, Any]], path: Path) -> None:
    n = len(rows)
    severe = [r for r in rows if r["severity"] != "subthreshold"]
    n_severe = len(severe)
    detected = sum(1 for r in severe if r["verdict"] == "DETECTED")
    boundary = [r for r in rows if r["severity"] == "subthreshold"]
    tn = sum(1 for r in boundary if r["verdict"] == "TRUE_NEGATIVE")
    fp = sum(1 for r in boundary if r["verdict"] == "FALSE_POSITIVE")

    out: List[str] = []
    out.append("# Paper §4.1 — Fault injection benchmark\n\n")
    out.append(f"- Total injected faults: **{n}** ({n_severe} bug-class + "
                f"{len(boundary)} subthreshold boundary cases)\n")
    out.append(f"- **Detection rate: {detected}/{n_severe} = "
                f"{detected/max(n_severe,1):.1%}** "
                f"(severe + moderate faults only)\n")
    out.append(f"- Boundary cases: **{tn}/{len(boundary)} true negative** "
                f"(zero FP at sensitivity floor)\n\n")

    # Per-tier
    out.append("## By tier\n\n| tier | faults | detected | det_rate |\n|---|---:|---:|---:|\n")
    for tier in ("T0", "T1"):
        ts = [r for r in severe if r["tier"] == tier]
        td = sum(1 for r in ts if r["verdict"] == "DETECTED")
        out.append(f"| {tier} | {len(ts)} | {td} | "
                    f"{td/max(len(ts),1):.1%} |\n")

    # Per-category
    out.append("\n## By category\n\n| category | faults | detected | det_rate |\n"
                "|---|---:|---:|---:|\n")
    cats: Dict[str, list] = {}
    for r in severe:
        cats.setdefault(r["category"], []).append(r)
    for cat in sorted(cats):
        rs = cats[cat]
        nd = sum(1 for r in rs if r["verdict"] == "DETECTED")
        out.append(f"| {cat} | {len(rs)} | {nd} | "
                    f"{nd/max(len(rs),1):.1%} |\n")

    # Per-fault listing
    out.append("\n## Per-fault verdicts\n\n"
                "| fault_id | expected_rule | severity | tier | category | "
                "verdict | description |\n"
                "|---|---|---|---|---|---|---|\n")
    for r in rows:
        out.append(f"| {r['fault_id']} | `{r['expected_rule']}` | "
                    f"{r['severity']} | {r['tier']} | {r['category']} | "
                    f"**{r['verdict']}** | {r['description'][:60]} |\n")

    path.write_text("".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/eval/fault_injection.csv")
    ap.add_argument("--table-out",
                    default="benchmark/eval/paper_table_fault_injection.md")
    args = ap.parse_args()

    rows = run_all()

    fields = ["fault_id", "expected_rule", "severity", "category", "tier",
              "verdict", "fired_rules", "description"]
    with Path(args.out).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    write_paper_table(rows, Path(args.table_out))

    n = len(rows)
    severe = [r for r in rows if r["severity"] != "subthreshold"]
    detected = sum(1 for r in severe if r["verdict"] == "DETECTED")
    print(f"{detected}/{len(severe)} severe faults detected; "
          f"{n - len(severe)} boundary cases. -> {args.out}")
    print(f"-> {args.table_out}")


if __name__ == "__main__":
    main()
