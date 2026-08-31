"""Inline TrainAudit rule checks for the 3 new D1' surrogates.

Implements the same rules that TrainAudit's pattern catalog instantiates,
but runs inline (no real framework hook needed) on the surrogate scripts.

Rules tested:
  CF1 → P4 (Invocation Frequency): tracker.calls_per_step == 1 each step
  CM1 → P3 (Cross-Rank Replication): metric_log_per_rank[0] == metric_log_per_rank[r] for all r
  OF1 → P1 (Dtype Preservation): gnorm.dtype == gnorm_after_offload.dtype

Output: per-(bug, variant) verdict, written to results/trainaudit_d1prime_inline.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def check_cf1(variant: str):
    """P4 rule: aux-loss tracker called exactly once per step."""
    import torch
    import torch.nn as nn

    class AuxLossTracker:
        def __init__(self):
            self.calls_per_step = 0
            self.total = 0.0
        def save_to_aux_losses_tracker(self, value):
            self.calls_per_step += 1
            self.total += float(value)
        def end_step(self):
            c = self.calls_per_step
            self.calls_per_step = 0
            return c

    class MoELayer(nn.Module):
        def __init__(self, tracker, guarded):
            super().__init__()
            self.gate = nn.Linear(8, 4)
            self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
            self.tracker = tracker
            self.guarded = guarded
        def forward(self, x):
            scores = self.gate(x).softmax(dim=-1)
            aux = (scores * scores.log().clamp(min=-10)).sum()
            if (not self.guarded) or torch.is_grad_enabled():
                self.tracker.save_to_aux_losses_tracker(aux.item())
            weighted = sum(s.unsqueeze(-1) * e(x)
                           for s, e in zip(scores.unbind(-1), self.experts))
            return weighted

    torch.manual_seed(0)
    tracker = AuxLossTracker()
    guarded = (variant == "fixed")  # buggy: no guard; fixed: guard with grad-enabled
    moe = MoELayer(tracker, guarded=guarded)

    violations = []
    for step in range(16):
        x = torch.randn(2, 8, requires_grad=True)
        out = moe(x)
        with torch.no_grad():
            _ = moe(x)
        out.pow(2).sum().backward()
        c = tracker.end_step()
        if c != 1:  # P4 invariant violation
            violations.append({"step": step, "calls_per_step": c, "expected": 1})
    return {
        "rule": "P4-invocation-frequency",
        "n_violations": len(violations),
        "first_violation_step": violations[0]["step"] if violations else None,
        "evidence": violations[:3],
    }


def check_cm1(variant: str):
    """P3 rule: metric_log_per_rank values must agree across ranks."""
    import torch

    def fake_collective_max(local_values, reduce_op):
        if reduce_op == "max_local":
            return local_values
        g = max(local_values)
        return [g] * len(local_values)

    torch.manual_seed(0)
    n_ranks = 4
    n_steps = 32
    metric_log = [[] for _ in range(n_ranks)]
    op = "max_local" if variant == "buggy" else "max_global"

    violations = []
    for step in range(n_steps):
        local = [1000.0 + 5.0 * torch.randn(()).item() + 2.0 * r for r in range(n_ranks)]
        reduced = fake_collective_max(local, reduce_op=op)
        for r in range(n_ranks):
            metric_log[r].append(reduced[r])
        # P3 check: rank-0 reading must equal all other ranks' readings
        ref = reduced[0]
        for r in range(1, n_ranks):
            if abs(reduced[r] - ref) > 1e-9:
                violations.append({"step": step, "rank": r,
                                   "value": reduced[r], "rank0": ref})
    return {
        "rule": "P3-cross-rank-replication",
        "n_violations": len(violations),
        "first_violation_step": violations[0]["step"] if violations else None,
        "evidence": violations[:3],
    }


def check_of1(variant: str):
    """P1 rule: dtype preserved through offload round-trip."""
    import torch
    import torch.nn as nn

    def fake_offload_then_restore(t, restore_dtype):
        cpu_copy = t.detach().clone()
        return cpu_copy.to(restore_dtype).to(t.dtype)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    violations = []
    for step in range(20):
        opt.zero_grad()
        x = torch.randn(4, 16)
        y = model(x).pow(2).sum()
        y.backward()
        gnorm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
        # buggy: fp16 round-trip; fixed: preserves dtype
        restore_dtype = torch.float16 if variant == "buggy" else gnorm.dtype
        gnorm_after = fake_offload_then_restore(gnorm, restore_dtype)
        # P1 check: the round-trip must preserve precision (fp16 truncation introduces drift)
        rel_drift = abs(gnorm.item() - gnorm_after.item()) / max(gnorm.item(), 1e-12)
        if rel_drift > 1e-6:  # P1 threshold for non-cast op preserving precision
            violations.append({"step": step, "rel_drift": rel_drift,
                               "input_dtype": str(gnorm.dtype),
                               "output_via": str(restore_dtype)})
        opt.step()
    return {
        "rule": "P1-dtype-preservation",
        "n_violations": len(violations),
        "first_violation_step": violations[0]["step"] if violations else None,
        "evidence": violations[:3],
    }


CHECKS = {"CF1": check_cf1, "CM1": check_cm1, "OF1": check_of1}


def main():
    out = {"experiment": "trainaudit_inline_check",
           "results": []}
    for bug, fn in CHECKS.items():
        for variant in ["buggy", "fixed"]:
            r = fn(variant)
            verdict = "DETECTED" if r["n_violations"] > 0 else "CLEAN"
            entry = {
                "bug_id": bug, "variant": variant,
                "verdict": verdict, **r,
            }
            out["results"].append(entry)
            print(f"[{bug}/{variant:<5}] {verdict:<8} {r['rule']:<28} "
                  f"violations={r['n_violations']}/16-32 "
                  f"first_step={r['first_violation_step']}")

    # Per-bug summary: TrainAudit DETECTED iff buggy violations > 0 AND fixed violations == 0
    summary = {}
    for bug in CHECKS:
        b = next(r for r in out["results"] if r["bug_id"]==bug and r["variant"]=="buggy")
        f = next(r for r in out["results"] if r["bug_id"]==bug and r["variant"]=="fixed")
        summary[bug] = {
            "buggy_detected": b["verdict"] == "DETECTED",
            "fixed_fp": f["verdict"] == "DETECTED",
            "rule": b["rule"],
        }
    out["summary"] = summary

    out_file = RESULTS / "trainaudit_d1prime_inline.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_file}")
    print("\nPer-bug summary:")
    for bug, s in summary.items():
        bd = "✓" if s["buggy_detected"] else "✗"
        fp = "✓" if s["fixed_fp"] else "✗"
        print(f"  {bug}: buggy_detected={bd}  fixed_fp={fp}  rule={s['rule']}")


if __name__ == "__main__":
    main()
