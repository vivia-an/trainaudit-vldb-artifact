"""
Build three constraint libraries for the leave-one-out ablation:

  - lib_full.json         : unchanged (current deployed library)
  - lib_no_topo.json      : strip parallel/topology keys from applicable_conditions
                            (dp/tp/pp/ep/sp/zero/sharded/...) so SQL fires regardless
                            of the active parallel config — simulates dropping pi_topo
  - lib_no_precond.json   : strip non-topology preconditions (stage/dtype/requires_grad/
                            checkpoint_restored/moe/sp/...) so SQL fires regardless of
                            phase/sharding — simulates dropping pi_precond
  - lib_no_adversarial.json : strip ALL applicable_conditions — Phase-1 proxy for
                            theta_conf=0 mining (rules never scoped by the S3 funnel);
                            full re-mine via run_mining_no_adversarial.sh when needed

The goal is to produce drop-in replacements that the verifier can load via
the --constraints-file flag added to __main__.py.
"""

import argparse
import copy
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "config" / "predefined_constraints.json"
OUT_DIR = REPO / "config"

# Topology / parallel-config keys — these encode pi_topo (which ranks are
# expected to agree under the active parallel topology).
TOPO_KEYS = {
    "dp", "tp", "pp", "ep", "sp",
    "tp_size", "pp_size", "dp_size", "ep_size", "sp_size",
    "tensor_model_parallel", "pipeline_model_parallel",
    "data_parallel", "expert_model_parallel",
    "zero", "zero_stage", "zero_optimization",
    "tpl", "param_sharded", "param_full_replica", "sharded",
}

# Precondition keys — these encode pi_precond (when/which-modules the check
# is valid: training stage, dtype, requires_grad, etc).
PRECOND_KEYS_INFER = None  # we treat "everything not in TOPO_KEYS and not _skip" as precond


def strip_topo(constraint):
    ac = constraint.get("applicable_conditions", {}) or {}
    constraint["applicable_conditions"] = {
        k: v for k, v in ac.items() if k not in TOPO_KEYS
    }
    return constraint


def strip_precond(constraint):
    ac = constraint.get("applicable_conditions", {}) or {}
    constraint["applicable_conditions"] = {
        k: v for k, v in ac.items() if (k in TOPO_KEYS or k.startswith("_"))
    }
    return constraint


def strip_all_adversarial(constraint):
    """Proxy for no-adversarial mining: drop all runtime scoping guards."""
    constraint["applicable_conditions"] = {}
    return constraint


def transform(src, op):
    out = copy.deepcopy(src)
    for cat, cdict in out.get("constraints", {}).items():
        if not isinstance(cdict, dict):
            continue
        for name, c in cdict.items():
            op(c)
    return out


def stat(lib, label):
    total = 0
    with_ac = 0
    ac_keys = set()
    for cat, cdict in lib.get("constraints", {}).items():
        if not isinstance(cdict, dict):
            continue
        for name, c in cdict.items():
            total += 1
            ac = c.get("applicable_conditions", {}) or {}
            if ac:
                with_ac += 1
                ac_keys.update(ac.keys())
    print(f"  [{label}] total constraints={total}, "
          f"with applicable_conditions={with_ac}, "
          f"ac keys present={sorted(ac_keys)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    src = json.load(open(args.src, encoding="utf-8"))
    print(f"[load] {args.src}")
    stat(src, "FULL (source)")

    full = copy.deepcopy(src)
    no_topo = transform(src, strip_topo)
    no_precond = transform(src, strip_precond)
    no_adversarial = transform(src, strip_all_adversarial)

    print("\n[derived]")
    stat(full, "lib_full")
    stat(no_topo, "lib_no_topo")
    stat(no_precond, "lib_no_precond")
    stat(no_adversarial, "lib_no_adversarial")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lib in [
        ("lib_full.json", full),
        ("lib_no_topo.json", no_topo),
        ("lib_no_precond.json", no_precond),
        ("lib_no_adversarial.json", no_adversarial),
    ]:
        p = out_dir / name
        json.dump(lib, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[write] {p} ({os.path.getsize(p)//1024} KiB)")


if __name__ == "__main__":
    main()
