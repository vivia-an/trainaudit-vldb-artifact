"""Runner skeleton for D1' three-way detection comparison.

Runs each of D1' (17 bugs) through TrainAudit / TrainCheck / Naïve, writes per-tool
JSON results to results/.

D1' = 14 kept (in benchmark/eval/traincheck_surrogates/) + 3 new (in benchmark/eval/d1_prime/).

USAGE:
    python3 run_d1prime_threeway.py [--tool trainaudit|traincheck|naive|all]

REQUIREMENTS:
    - For TrainAudit: paper TrainAudit pipeline (placeholder hooks below)
    - For TrainCheck: GPU environment (CUDA driver) — see _traincheck_*.py
    - For Naïve: pure Python, no GPU needed

NOTE: Phase 2 of brief 30 expects this script to wire into the paper's existing
runner. Specific calls left as TODO until paper pipeline path is confirmed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUR_DIR = ROOT / "benchmark/eval/traincheck_surrogates"
NEW_DIR = ROOT / "benchmark/eval/d1_prime"
RESULTS_DIR = NEW_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


D1_PRIME = [
    # 14 kept from existing traincheck_surrogates/
    ("B1",       "gradient_sync",   "Megatron-LM",    "T1", SUR_DIR),
    ("B2",       "gradient_sync",   "Megatron-LM",    "T1", SUR_DIR),
    ("B3",       "dtype",           "DeepSpeed",      "T1", SUR_DIR),
    ("B8",       "moe",             "DeepSpeed",      "T1", SUR_DIR),
    ("B11",      "numerical",       "DeepSpeed",      "T0", SUR_DIR),
    ("B12",      "lr_schedule",     "OLMo-core",      "T0", SUR_DIR),
    ("M-012",    "moe",             "Megatron-LM",    "T1", SUR_DIR),
    ("M-020",    "sharding",        "Megatron-LM",    "T1", SUR_DIR),
    ("M-024",    "dtype",           "Megatron-LM",    "T1", SUR_DIR),
    ("O-005",    "checkpoint",      "OLMo",           "T0", SUR_DIR),
    ("O-NEW-1",  "numerical",       "OLMo",           "T0", SUR_DIR),
    ("O-NEW-9",  "data_loading",    "OLMo",           "T0", SUR_DIR),
    ("OC-NEW-2", "optimizer_state", "OLMo-core",      "T0", SUR_DIR),
    ("OC-NEW-3", "lr_schedule",     "OLMo-core",      "T1", SUR_DIR),
    # 3 new in d1_prime/
    ("CF1",      "control_flow",    "(surrogate)",    "T0", NEW_DIR),
    ("CM1",      "communication",   "(surrogate)",    "T1", NEW_DIR),
    ("OF1",      "offload",         "(surrogate)",    "T1", NEW_DIR),
]


def run_naive(bug_id, src_dir):
    """Naïve = NaN/Inf detection + loss-spike threshold + grad-norm bound.

    For surrogate: run buggy.py and fixed.py, capture printed metrics, check thresholds.
    """
    result = {"bug_id": bug_id, "tool": "naive"}
    for variant in ["buggy", "fixed"]:
        f = src_dir / f"{bug_id}_{variant}.py"
        if not f.exists():
            result[f"{variant}_status"] = "MISSING"; continue
        try:
            out = subprocess.run(
                [sys.executable, str(f)], capture_output=True, text=True, timeout=120
            )
            text = out.stdout + out.stderr
            triggered = (
                "nan" in text.lower() or "inf" in text.lower() or
                "RuntimeError" in text or "exit code" in text.lower()
            )
            result[f"{variant}_triggered"] = triggered
            result[f"{variant}_returncode"] = out.returncode
        except Exception as e:
            result[f"{variant}_error"] = str(e)
    result["buggy_detected"] = result.get("buggy_triggered", False)
    result["fixed_fp"] = result.get("fixed_triggered", False)
    return result


def run_traincheck(bug_id, src_dir):
    """TrainCheck = learn invariants from healthy trace, query against buggy trace.

    Requires GPU environment (CUDA driver). Returns SKIPPED if not available.
    """
    fixed_adapter = src_dir / f"_traincheck_{bug_id}_fixed.py"
    buggy_adapter = src_dir / f"_traincheck_{bug_id}_buggy.py"
    if not fixed_adapter.exists() or not buggy_adapter.exists():
        return {"bug_id": bug_id, "tool": "traincheck", "status": "ADAPTER_MISSING"}

    # TODO: integrate with paper's TrainCheck invariant learning + checking pipeline
    # The pipeline:
    #   1. Run fixed adapter → get healthy trace at /tmp/tc_<bug>/trace_fixed/
    #   2. Run TrainCheck inference on healthy trace → get learned invariants
    #   3. Run buggy adapter → get buggy trace at /tmp/tc_<bug>/trace_buggy/
    #   4. Query learned invariants against buggy trace → count violations
    return {
        "bug_id": bug_id, "tool": "traincheck",
        "status": "REQUIRES_GPU_AND_PAPER_PIPELINE",
        "fixed_adapter_exists": fixed_adapter.exists(),
        "buggy_adapter_exists": buggy_adapter.exists(),
    }


def run_trainaudit(bug_id, src_dir):
    """TrainAudit = pattern-rule instantiation + hook/SQL/static check.

    For each bug, paper TrainAudit pipeline expects framework-specific driver.
    For surrogate, we can implement direct rule check inline.
    """
    # TODO: integrate with paper TrainAudit pipeline. Each bug has a known
    # rule (see source_bug_mapping.json) — for surrogate we can implement
    # a minimal rule check directly:
    #   - CF1: assert tracker.calls_per_step == 1 at end of each step
    #   - CM1: assert metric_log_per_rank[0] == metric_log_per_rank[r] for all r
    #   - OF1: assert gnorm.dtype == gnorm_after_offload.dtype
    return {
        "bug_id": bug_id, "tool": "trainaudit",
        "status": "REQUIRES_PAPER_PIPELINE",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["trainaudit","traincheck","naive","all"], default="all")
    args = ap.parse_args()

    runners = {"naive": run_naive, "traincheck": run_traincheck, "trainaudit": run_trainaudit}
    tools_to_run = list(runners.keys()) if args.tool == "all" else [args.tool]

    for tool in tools_to_run:
        print(f"\n=== {tool} ===")
        all_results = []
        for bug_id, cls, fw, tier, src_dir in D1_PRIME:
            print(f"  [{tool}] {bug_id} ({cls}, {fw}, {tier})")
            r = runners[tool](bug_id, src_dir)
            r.update({"category": cls, "framework": fw, "tier": tier})
            all_results.append(r)
        out_file = RESULTS_DIR / f"{tool}_d1prime.json"
        out_file.write_text(json.dumps({
            "tool": tool, "n_bugs": len(D1_PRIME), "results": all_results
        }, indent=2))
        print(f"  → wrote {out_file}")


if __name__ == "__main__":
    main()
