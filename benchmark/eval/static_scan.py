"""Static rule-pattern scanner — Phase 1 of the trainaudit hunt.

For each of trainaudit's 22 rules, run a grep recipe against the 4 framework
checkouts under exp/frameworks/. Each recipe encodes the source-level
anti-pattern that the corresponding rule catches at runtime. Matches are
*candidates*, not proven bugs — Phase 2 (dynamic_confirm.py) confirms by
running training on a minimal reproducer and checking that the rule
actually fires.

Usage:
    python benchmark/eval/static_scan.py
        --out  benchmark/eval/hunt_log/static_findings.md

Output is a markdown table grouped by rule_id, each row:
    | framework | file:line | excerpt | hypothesis |

Recipe schema (per rule):
    {
        "rule_id":   "T0-checkpoint-preserve-rng",
        "hypothesis": "preserve_rng_state hard-coded False with dropout active",
        "frameworks": {
            "OLMo-core": {
                "paths": ["src/olmo_core/nn/"],
                "include_pattern": r"preserve_rng_state\s*=\s*False",
                "exclude_pattern": r"#.*never|test_",
            },
            ...
        }
    }

Adding a new rule is just adding a new dict — no flow changes.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_FW_ROOT = _REPO / "exp" / "frameworks"

_FRAMEWORKS = {
    "DeepSpeed": _FW_ROOT / "DeepSpeed",
    "Megatron-LM": _FW_ROOT / "Megatron-LM",
    "OLMo": _FW_ROOT / "OLMo",
    "OLMo-core": _FW_ROOT / "OLMo-core",
}


@dataclass
class Match:
    rule_id: str
    framework: str
    file: str
    line: int
    excerpt: str
    hypothesis: str


# -----------------------------------------------------------------------------
# Recipes — one entry per rule_id. Frameworks left out are not scanned for
# that rule (typically because the anti-pattern doesn't apply there).
# -----------------------------------------------------------------------------

RECIPES: List[Dict] = [
    {
        "rule_id": "T0-checkpoint-preserve-rng",
        "hypothesis": "preserve_rng_state hard-coded False — when the model "
                      "uses dropout/RNG ops, activation checkpointing recomputes "
                      "with a different RNG state → silent grad mis-match.",
        "frameworks": {
            "OLMo-core": {
                "paths": ["src/olmo_core/"],
                "include_pattern": r"preserve_rng_state\s*=\s*False",
            },
            "OLMo": {
                "paths": ["olmo/"],
                "include_pattern": r"preserve_rng_state\s*=\s*False",
            },
            "DeepSpeed": {
                "paths": ["deepspeed/runtime/activation_checkpointing/"],
                "include_pattern": r"preserve_rng_state\s*=\s*False",
            },
            "Megatron-LM": {
                "paths": ["megatron/"],
                "include_pattern": r"preserve_rng_state\s*=\s*False",
            },
        },
    },
    {
        "rule_id": "T0-clip-grad-bounded",
        "hypothesis": "clip_coef formula uses max(coef, 1.0) instead of "
                      "min(coef, 1.0); or `if clip_coef > 1: scale grad` "
                      "(direction reversed).",
        "frameworks": {
            "DeepSpeed": {
                "paths": ["deepspeed/runtime/", "deepspeed/moe/"],
                "include_pattern": r"clip_coef\s*=\s*max_norm\s*/|"
                                    r"torch\.max\s*\([^,]+,\s*clip_coef|"
                                    r"if\s+clip_coef\s*>\s*1\s*:",
            },
            "Megatron-LM": {
                "paths": ["megatron/core/optimizer/", "megatron/optimizer/"],
                "include_pattern": r"clip_coef\s*=|grad_norm\s*=.+max_norm",
            },
        },
    },
    {
        "rule_id": "T0-optim-step-counter",
        "hypothesis": "Optimizer state['step'] either not initialized or not "
                      "incremented every step. Look for state[..]['step'] "
                      "without an obvious add_/+= 1 in the same function.",
        "frameworks": {
            "OLMo-core": {
                "paths": ["src/olmo_core/optim/"],
                "include_pattern": r"state\[.step.\]|step_factor",
            },
            "Megatron-LM": {
                "paths": ["megatron/core/optimizer/", "megatron/optimizer/"],
                "include_pattern": r"state\[.step.\]",
            },
            "DeepSpeed": {
                "paths": ["deepspeed/runtime/", "deepspeed/ops/adam/"],
                "include_pattern": r"state\[.step.\]",
            },
        },
    },
    {
        "rule_id": "T0-norm-output-rms",
        "hypothesis": "RMS/LayerNorm forward path is missing weight scaling, "
                      "or weight init is non-standard (zeros, off-init).",
        "frameworks": {
            "OLMo": {
                "paths": ["olmo/model.py"],
                "include_pattern": r"class\s+\w*Norm.*\(.*Module|"
                                    r"def forward.*y\s*=.*\.pow\(2\)\.mean",
            },
            "OLMo-core": {
                "paths": ["src/olmo_core/nn/", "src/olmo_core/nn/transformer/"],
                "include_pattern": r"class\s+\w*Norm.*\(.*Module|self\.weight\s*=",
            },
            "Megatron-LM": {
                "paths": ["megatron/core/transformer/torch_norm.py",
                          "megatron/core/fusions/"],
                "include_pattern": r"class\s+\w*Norm.*\(.*Module",
            },
        },
    },
    {
        "rule_id": "T0-softmax-degenerate",
        "hypothesis": "Router uses top_k=1 hard routing or softmax dim wrong; "
                      "all probability mass on one expert silently kills "
                      "load-balance loss.",
        "frameworks": {
            "Megatron-LM": {
                "paths": ["megatron/core/transformer/moe/"],
                "include_pattern": r"top_k\s*=\s*1\b|topk\s*\([^,]+,\s*1\b",
            },
            "DeepSpeed": {
                "paths": ["deepspeed/moe/"],
                "include_pattern": r"top_k\s*=\s*1\b|num_experts_per_token\s*=\s*1",
            },
            "OLMo-core": {
                "paths": ["src/olmo_core/nn/moe/"],
                "include_pattern": r"top_k\s*=\s*1\b|num_experts_per_tok\s*=\s*1",
            },
        },
    },
    {
        "rule_id": "T1-comm-dtype-matches-training",
        "hypothesis": "Explicit .to(torch.float16) / .half() inside a "
                      "distributed collective path. If model trains in BF16, "
                      "comm uses different dtype → silent precision loss.",
        "frameworks": {
            "DeepSpeed": {
                "paths": ["deepspeed/runtime/comm/", "deepspeed/runtime/"],
                "include_pattern": r"all_reduce\([^)]*\.(half|to\(torch\.float16)|"
                                    r"\.half\(\)\.\s*detach.*all_reduce|"
                                    r"reduce_scatter.*\.(half|to\(torch\.float16)",
            },
            "Megatron-LM": {
                "paths": ["megatron/core/distributed/", "megatron/core/tensor_parallel/"],
                "include_pattern": r"all_reduce\([^)]*\.(half|to\(torch\.float16)|"
                                    r"reduce_scatter.*\.(half|to\(torch\.float16)",
            },
        },
    },
    {
        "rule_id": "T1-process-group-size",
        "hypothesis": "Process-group init uses a wrong sizing variable (e.g. "
                      "`num_experts` instead of `ep_size`). EP comm group then "
                      "routes to the wrong rank set.",
        "frameworks": {
            "DeepSpeed": {
                "paths": ["deepspeed/moe/", "deepspeed/utils/"],
                "include_pattern": r"new_group\([^)]*num_experts|"
                                    r"new_group\([^)]*world_size\s*//\s*\w*expert",
            },
            "Megatron-LM": {
                "paths": ["megatron/core/parallel_state.py"],
                "include_pattern": r"new_group\([^)]*num_experts|"
                                    r"expert_model_parallel_size\s*=\s*\w*world_size",
            },
        },
    },
    {
        "rule_id": "T1-router-attribute",
        "hypothesis": "Router class missing an attribute that downstream "
                      "(e.g. aux-loss reduction) reads conditionally — when "
                      "the attr is missing, the path is silently skipped.",
        "frameworks": {
            "Megatron-LM": {
                "paths": ["megatron/core/transformer/moe/router.py"],
                "include_pattern": r"class\s+\w*Router.*\(.*Module|"
                                    r"calculate_per_token_loss",
            },
            "OLMo-core": {
                "paths": ["src/olmo_core/nn/moe/"],
                "include_pattern": r"class\s+\w*Router.*\(.*Module|"
                                    r"calculate_per_token_loss",
            },
        },
    },
    {
        "rule_id": "T1-sqrt-decay-direction",
        "hypothesis": "sqrt-style decay returns lr that increases over the "
                      "decay window (sign of formula reversed).",
        "frameworks": {
            "OLMo-core": {
                "paths": ["src/olmo_core/optim/scheduler.py"],
                "include_pattern": r"_sqrt_decay|sqrt\(.*progress|"
                                    r"sqrt\([^)]*current\s*/\s*decay",
            },
            "Megatron-LM": {
                "paths": ["megatron/optimizer_param_scheduler.py",
                          "megatron/training/optimizer_param_scheduler.py"],
                "include_pattern": r"sqrt|inverse_square_root",
            },
        },
    },
    {
        "rule_id": "T0-grad-norm-finite",
        "hypothesis": "Gradient norm computation in fp16 without explicit "
                      "fp32 cast — overflows silently → norm = inf, "
                      "downstream clip becomes a no-op.",
        "frameworks": {
            "DeepSpeed": {
                "paths": ["deepspeed/runtime/utils.py",
                          "deepspeed/runtime/zero/"],
                "include_pattern": r"\.norm\([^)]*\).*\.item\(\)|"
                                    r"torch\.linalg\.norm\([^)]*\)",
            },
            "Megatron-LM": {
                "paths": ["megatron/core/optimizer/clip_grads.py",
                          "megatron/optimizer/clip_grads.py"],
                "include_pattern": r"\.norm\(|torch\.linalg\.norm",
            },
        },
    },
    {
        "rule_id": "T0-token-id-range",
        "hypothesis": "Tokenizer or data pipeline produces ids ≥ vocab_size; "
                      "embedding lookup silently returns garbage rows.",
        "frameworks": {
            "Megatron-LM": {
                "paths": ["megatron/core/datasets/", "megatron/training/tokenizer/"],
                "include_pattern": r"vocab_size|tokens\.clamp",
            },
            "OLMo": {
                "paths": ["olmo/data/", "olmo/tokenizer.py"],
                "include_pattern": r"vocab_size|tokens\.clamp|input_ids\s*%",
            },
        },
    },
    {
        "rule_id": "T1-residual-stream",
        "hypothesis": "Residual add path applies norm BEFORE the residual "
                      "save (`x = norm(x); x = x + sub(x)` rather than "
                      "`x = x + sub(norm(x))`); changes effective compute.",
        "frameworks": {
            "Megatron-LM": {
                "paths": ["megatron/core/transformer/transformer_layer.py"],
                "include_pattern": r"residual\s*=\s*\w+\s*$|"
                                    r"hidden_states\s*=\s*self\.input_layernorm",
            },
            "OLMo-core": {
                "paths": ["src/olmo_core/nn/transformer/block.py"],
                "include_pattern": r"residual\s*=|x\s*=\s*x\s*\+\s*self\.",
            },
        },
    },
    {
        "rule_id": "T1-grad-replica-cksum-equal",
        "hypothesis": "DP grad all-reduce skipped on a frozen / requires_grad=False "
                      "module; replicas drift silently across DP ranks.",
        "frameworks": {
            "Megatron-LM": {
                "paths": ["megatron/core/distributed/"],
                "include_pattern": r"requires_grad\s*=\s*False|"
                                    r"if\s+param\.requires_grad\s*:.*all_reduce|"
                                    r"continue\s*$",
            },
            "DeepSpeed": {
                "paths": ["deepspeed/runtime/zero/"],
                "include_pattern": r"requires_grad\s*=\s*False|"
                                    r"if\s+not\s+param\.requires_grad\s*:",
            },
        },
    },
    {
        "rule_id": "T1-jitter-dtype",
        "hypothesis": "MoE jitter noise tensor created in default float32 "
                      "while model runs in BF16 — mixed-dtype add silently "
                      "downcasts.",
        "frameworks": {
            "Megatron-LM": {
                "paths": ["megatron/core/transformer/moe/"],
                "include_pattern": r"jitter|noise.*torch\.(rand|randn)",
            },
            "DeepSpeed": {
                "paths": ["deepspeed/moe/"],
                "include_pattern": r"jitter|noise.*torch\.(rand|randn)",
            },
        },
    },
    {
        "rule_id": "T1-expert-bias-dtype",
        "hypothesis": "Expert bias initialized in fp32 while expert weights "
                      "are bf16; bias add silently upcasts/loses bits.",
        "frameworks": {
            "Megatron-LM": {
                "paths": ["megatron/core/transformer/moe/"],
                "include_pattern": r"expert.*bias.*float|"
                                    r"bias\s*=\s*torch\.(zeros|empty)\(",
            },
            "DeepSpeed": {
                "paths": ["deepspeed/moe/"],
                "include_pattern": r"expert.*bias.*float|"
                                    r"bias\s*=\s*torch\.(zeros|empty)\(",
            },
        },
    },
    {
        "rule_id": "T0-dtype-propagation",
        "hypothesis": "Hard-coded torch.float16 cast somewhere on the "
                      "forward/backward path; in BF16 training causes silent "
                      "precision drop.",
        "frameworks": {
            "DeepSpeed": {
                "paths": ["deepspeed/runtime/"],
                "include_pattern": r"\.to\(torch\.float16\)|\.half\(\)",
            },
            "Megatron-LM": {
                "paths": ["megatron/core/"],
                "include_pattern": r"\.to\(torch\.float16\)|\.half\(\)",
            },
        },
    },
]


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------

def _grep(pattern: str, root: Path, path_subs: List[str],
          exclude: Optional[str] = None) -> List[tuple]:
    """Run grep -RnE pattern over each subpath of root. Return list of (file, lineno, text)."""
    out: List[tuple] = []
    for sub in path_subs:
        target = root / sub if sub else root
        if not target.exists():
            continue
        cmd = ["grep", "-RInE", "--include=*.py",
               "--exclude-dir=__pycache__", "--exclude-dir=tests",
               "--exclude-dir=test", "--exclude-dir=.git",
               pattern, str(target)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            continue
        if res.returncode not in (0, 1):
            # non-zero unexpected, but grep returns 1 when no matches — fine
            continue
        for line in res.stdout.splitlines():
            # Format: <path>:<lineno>:<text>
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            fpath, lineno, text = parts
            if not lineno.isdigit():
                continue
            text = text.rstrip()
            if exclude and re.search(exclude, text):
                continue
            out.append((fpath, int(lineno), text))
    return out


def run_recipes(recipes: List[Dict]) -> List[Match]:
    matches: List[Match] = []
    for r in recipes:
        rule_id = r["rule_id"]
        hypothesis = r["hypothesis"]
        for fw_name, fw_cfg in r["frameworks"].items():
            fw_root = _FRAMEWORKS.get(fw_name)
            if fw_root is None or not fw_root.exists():
                continue
            pat = fw_cfg["include_pattern"]
            excl = fw_cfg.get("exclude_pattern")
            for fpath, lineno, text in _grep(pat, fw_root, fw_cfg["paths"], excl):
                rel = os.path.relpath(fpath, fw_root)
                excerpt = text.lstrip()
                if len(excerpt) > 140:
                    excerpt = excerpt[:140] + " …"
                matches.append(Match(
                    rule_id=rule_id, framework=fw_name,
                    file=rel, line=lineno,
                    excerpt=excerpt, hypothesis=hypothesis))
    return matches


def render_md(matches: List[Match], out: Path):
    by_rule: Dict[str, List[Match]] = {}
    for m in matches:
        by_rule.setdefault(m.rule_id, []).append(m)

    lines: List[str] = []
    lines.append("# Trainaudit hunt — Phase 1 static findings\n\n")
    lines.append("Auto-generated by `benchmark/eval/static_scan.py`. Each "
                  "row is a *candidate* — Phase 2 (`dynamic_confirm.py`) "
                  "must confirm the rule actually fires under a minimal "
                  "reproducer before we count it as a real silent error.\n\n")

    rule_count = len(by_rule)
    match_count = sum(len(v) for v in by_rule.values())
    lines.append(f"**{match_count} matches across {rule_count} rules.**\n\n")

    lines.append("## Quick rule-level summary\n\n")
    lines.append("| rule_id | matches | frameworks |\n|---|---:|---|\n")
    for rid in sorted(by_rule):
        rms = by_rule[rid]
        fws = sorted(set(m.framework for m in rms))
        lines.append(f"| `{rid}` | {len(rms)} | {', '.join(fws)} |\n")
    lines.append("\n")

    for rid in sorted(by_rule):
        rms = by_rule[rid]
        lines.append(f"## {rid}\n\n")
        if rms:
            lines.append(f"_{rms[0].hypothesis}_\n\n")
        lines.append("| framework | file:line | excerpt |\n|---|---|---|\n")
        for m in rms:
            esc = m.excerpt.replace("|", "\\|")
            lines.append(f"| {m.framework} | `{m.file}:{m.line}` | "
                          f"`{esc}` |\n")
        lines.append("\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    print(f"-> {out}  ({match_count} matches, {rule_count} rules)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/eval/hunt_log/static_findings.md")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    matches = run_recipes(RECIPES)
    render_md(matches, out)


if __name__ == "__main__":
    main()
