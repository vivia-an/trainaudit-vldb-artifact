"""Long clean run FP audit.

Runs N healthy training workloads × M steps each, captures every rule
violation. Any rule that fires on a clean run is a FP candidate that
needs precondition tightening or rule reformulation.

3 model archetypes:
  - mlp-2l: 2-layer MLP + LayerNorm — exercises module/optim hooks
  - gpt-tiny: 2-block transformer with attn_norm + mlp_norm
  - moe-style: Sequential containing a top-k softmax router + experts
    (clean — top-k=2, k>1 so no degenerate softmax)

For each model × {200, 500} steps, log all rule firings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

_HERE = Path(__file__).resolve().parent
_TA_ROOT = (_HERE / ".." / ".." / "trainaudit").resolve()
if str(_TA_ROOT) not in sys.path:
    sys.path.insert(0, str(_TA_ROOT))

import trainaudit  # noqa: E402


def build_mlp_2l():
    return nn.Sequential(
        nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, 32))


def build_gpt_tiny():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_norm = nn.LayerNorm(128)
            self.q = nn.Linear(128, 128)
            self.k = nn.Linear(128, 128)
            self.v = nn.Linear(128, 128)
            self.o = nn.Linear(128, 128)
            self.mlp_norm = nn.LayerNorm(128)
            self.mlp = nn.Sequential(nn.Linear(128, 256), nn.GELU(),
                                     nn.Linear(256, 128))
        def forward(self, x):
            h = self.attn_norm(x)
            h = self.o(self.q(h) * self.k(h) * self.v(h))
            x = x + h
            return x + self.mlp(self.mlp_norm(x))
    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(64, 128)
            self.blocks = nn.ModuleList([Block() for _ in range(2)])
            self.head = nn.Linear(128, 32)
        def forward(self, x):
            x = self.embed(x)
            for b in self.blocks:
                x = b(x)
            return self.head(x)
    return TinyGPT()


def build_moe_style():
    """Healthy MoE-like structure with proper top-k=2 router (not degenerate)."""
    class CleanRouter(nn.Module):
        def __init__(self, hidden, n_experts=4):
            super().__init__()
            self.linear = nn.Linear(hidden, n_experts)
        def forward(self, x):
            scores = self.linear(x)
            return torch.softmax(scores, dim=-1)
    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(64, 128)
            self.router = CleanRouter(128)
            self.head = nn.Linear(128, 32)
        def forward(self, x):
            x = self.embed(x)
            _ = self.router(x)
            return self.head(x)
    return MoE()


_BUILDERS = {
    "mlp-2l": (build_mlp_2l, 64, 32),
    "gpt-tiny": (build_gpt_tiny, 64, 32),
    "moe-style": (build_moe_style, 64, 32),
}


def run_clean(model_name: str, steps: int) -> List[Dict]:
    builder, in_dim, _ = _BUILDERS[model_name]
    with tempfile.TemporaryDirectory() as tmp:
        store = trainaudit.enable(
            tier=trainaudit.Tier.T1_FW_METADATA,
            db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = builder()
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            for step in range(steps):
                trainaudit.set_step(step)
                opt.zero_grad()
                x = torch.randn(4, in_dim)
                model(x).pow(2).sum().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                max_norm=10.0)
                opt.step()
            store.flush()
            results = trainaudit.run_rules()
        finally:
            trainaudit.disable()

    violations: List[Dict] = []
    for r in results:
        if r.violated:
            ev = r.evidence or {}
            violations.append({
                "rule_id": r.rule_id,
                "message": r.message,
                "n_violations": len(ev.get("violation_event_ids", []) or []),
                "sample": (ev.get("sample") or [])[:1],
            })
    return violations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="benchmark/eval/fp_audit.csv")
    ap.add_argument("--report",
                    default="benchmark/eval/paper_table_fp_audit.md")
    args = ap.parse_args()

    configs = [(m, s) for m in _BUILDERS for s in (200, 500)]
    rows = []
    fp_summary: Counter = Counter()
    print(f"=== Long clean run FP audit ===")
    print(f"{'config':25s} {'steps':>6s} {'violations':>11s} {'unique rules':>12s}")
    for (model_name, steps) in configs:
        violations = run_clean(model_name, steps)
        n_total_v = sum(v["n_violations"] for v in violations)
        unique_rules = len(violations)
        for v in violations:
            fp_summary[v["rule_id"]] += v["n_violations"]
        print(f"{model_name:25s} {steps:>6} {n_total_v:>11} {unique_rules:>12}")
        rows.append({"model": model_name, "steps": steps,
                     "n_violations": n_total_v,
                     "unique_rules": unique_rules,
                     "violations": violations})

    # CSV
    import csv
    with Path(args.out).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "steps", "rule_id", "n_violations", "message"])
        for r in rows:
            for v in r["violations"]:
                w.writerow([r["model"], r["steps"], v["rule_id"],
                            v["n_violations"], v["message"][:200]])

    # Paper table
    out = ["# Paper §4.3 — Long clean run FP audit\n\n"]
    out.append(f"3 healthy models × 2 step counts = 6 clean runs at "
                f"T1_FW_METADATA tier.\n\n")
    out.append("| model | steps | total violations | unique rules fired |\n"
                "|---|---:|---:|---:|\n")
    for r in rows:
        out.append(f"| {r['model']} | {r['steps']} | {r['n_violations']} | "
                    f"{r['unique_rules']} |\n")
    if fp_summary:
        out.append("\n## Rules that fired on healthy traces (FP candidates)\n\n"
                    "| rule_id | total fires across all clean runs |\n"
                    "|---|---:|\n")
        for rule_id, n in fp_summary.most_common():
            out.append(f"| `{rule_id}` | {n} |\n")
        out.append("\n**Each of these is a FP candidate** — needs "
                    "precondition tightening, more conservative tolerance, "
                    "or scope refinement.\n")
    else:
        out.append("\n**Zero rule fires across all 6 clean runs — 0 FP "
                    "across the audit corpus.**\n")
    Path(args.report).write_text("".join(out))
    print(f"\n-> {args.out}")
    print(f"-> {args.report}")


if __name__ == "__main__":
    main()
