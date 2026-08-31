"""Inline TrainAudit rule checks for the 8 D2-new surrogates (P9-P16).

Each surrogate's rule lifts directly from the new pattern catalog yaml in
benchmark/eval/pattern_expansion/new_patterns/.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def check_id1(variant):
    """P9: param.std() ∈ [declared × 0.5, declared × 1.5]."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    declared_std = 0.02
    layer = nn.Linear(64, 256)
    actual_std_used = 0.5 if variant == "buggy" else declared_std
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=actual_std_used)
    actual = layer.weight.std().item()
    lo, hi = declared_std * 0.5, declared_std * 1.5
    violations = []
    if not (lo <= actual <= hi):
        violations.append({"actual_std": actual, "declared": declared_std, "band": [lo, hi]})
    return {"rule": "P9-init-distribution-consistency", "n_violations": len(violations),
            "first_violation": violations[0] if violations else None, "evidence": violations}


def check_cc1(variant):
    """P10: config.zero_stage > 0 ⟹ optimizer.has_partitioned_state."""
    config = {"zero_stage": 2}
    has_partitioned = (variant == "fixed")  # buggy=False, fixed=True
    violations = []
    if config["zero_stage"] > 0 and not has_partitioned:
        violations.append({"config_zero_stage": config["zero_stage"],
                           "has_partitioned_state": has_partitioned})
    return {"rule": "P10-config-implied-coupling", "n_violations": len(violations),
            "first_violation": violations[0] if violations else None, "evidence": violations}


def check_pe1(variant):
    """P11: positions[cu_doc_lens[k]] == 0 for k > 0 (boundary reset)."""
    import torch
    seq_len = 64
    cu_doc_lens = [16, 40, 64]
    if variant == "buggy":
        pos = torch.arange(seq_len)
    else:
        pos = torch.zeros(seq_len, dtype=torch.long)
        ptr = 0
        for end in cu_doc_lens:
            pos[ptr:end] = torch.arange(end - ptr)
            ptr = end
    violations = []
    for k, end in enumerate(cu_doc_lens[:-1]):
        if pos[end].item() != 0:
            violations.append({"k": k, "boundary_idx": end, "pos_at_boundary": pos[end].item()})
    return {"rule": "P11-position-encoding-integrity", "n_violations": len(violations),
            "first_violation": violations[0] if violations else None, "evidence": violations}


def check_av1(variant):
    """P12: max |fused(x) - ref(x)| / |ref(x)| < 1e-5."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    class Norm(nn.Module):
        def __init__(self, dim, eps=1e-5, broken=False):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
            self.eps = eps
            self.broken = broken
        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            if self.broken:
                std = x.std(-1, keepdim=True)
                return self.weight * (x - mean) / std + self.bias
            var = x.var(-1, keepdim=True, unbiased=False)
            return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias

    fused = Norm(64, broken=(variant == "buggy"))
    ref = Norm(64, broken=False)
    with torch.no_grad():
        ref.weight.copy_(fused.weight); ref.bias.copy_(fused.bias)

    violations = []
    for step in range(20):
        x = torch.randn(4, 64)
        rel = ((fused(x) - ref(x)).abs().max() / ref(x).abs().max()).item()
        if rel > 1e-5:
            violations.append({"step": step, "rel_diff": rel})
    return {"rule": "P12-algorithm-variant-equivalence", "n_violations": len(violations),
            "first_violation": violations[0] if violations else None, "evidence": violations[:3]}


def check_ta1(variant):
    """P13: cached_norm == param.data.norm() after each mutation."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    layer = nn.Linear(32, 64)
    cached = layer.weight.data.norm().item()
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    violations = []
    for step in range(20):
        x = torch.randn(4, 32)
        loss = layer(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if variant == "fixed":
            cached = layer.weight.data.norm().item()  # refresh
        actual = layer.weight.data.norm().item()
        if abs(cached - actual) > 1e-6:
            violations.append({"step": step, "cached": cached, "actual": actual})
    return {"rule": "P13-tensor-aliasing-stale-state", "n_violations": len(violations),
            "first_violation": violations[0] if violations else None, "evidence": violations[:3]}


def check_sc1(variant):
    """P14: saved files cover all TP ranks [0, tp_size)."""
    tp_size = 2
    save_dir = tempfile.mkdtemp(prefix="sc1_check_")
    if variant == "buggy":
        with open(os.path.join(save_dir, "mp_rank_00_model_states.pt"), "w") as f: f.write("x")
    else:
        for r in range(tp_size):
            with open(os.path.join(save_dir, f"mp_rank_{r:02d}_model_states.pt"), "w") as f: f.write("x")
    saved = {f for f in os.listdir(save_dir) if f.startswith("mp_rank_")}
    expected = {f"mp_rank_{r:02d}_model_states.pt" for r in range(tp_size)}
    missing = expected - saved
    shutil.rmtree(save_dir)
    violations = [{"missing_ranks": sorted(missing)}] if missing else []
    return {"rule": "P14-sharded-state-completeness", "n_violations": len(violations),
            "first_violation": violations[0] if violations else None, "evidence": violations}


def check_cw1(variant):
    """P15: counter must be monotonic across observed steps."""
    import torch
    counter_dtype = torch.int8 if variant == "buggy" else torch.int64
    value = torch.tensor(0, dtype=counter_dtype)
    history = []
    violations = []
    for step in range(200):
        value = (value + 1).to(counter_dtype)
        v = value.item()
        if history and v < history[-1]:
            violations.append({"step": step, "prev": history[-1], "now": v})
        history.append(v)
    return {"rule": "P15-counter-width-adequacy", "n_violations": len(violations),
            "first_violation": violations[0] if violations else None, "evidence": violations[:3]}


def check_ln1(variant):
    """P16: aux_loss divisor matches declared reduction granularity (token-level)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    micro_batch, seq_len, vocab = 4, 16, 32
    head = nn.Linear(8, vocab)
    expected_divisor = micro_batch * seq_len
    declared_divisor_used = micro_batch if variant == "buggy" else (micro_batch * seq_len)
    violations = []
    if declared_divisor_used != expected_divisor:
        # check magnitude over 20 steps
        for step in range(20):
            x = torch.randn(micro_batch, seq_len, 8)
            logits = head(x)
            mask = torch.ones(micro_batch, seq_len, dtype=torch.bool)
            raw = (logits * logits).sum().item()
            buggy_loss = raw / declared_divisor_used
            correct_loss = raw / expected_divisor
            if abs(buggy_loss / correct_loss) > 2.0 or abs(buggy_loss / correct_loss) < 0.5:
                violations.append({"step": step, "ratio": buggy_loss / correct_loss})
    return {"rule": "P16-loss-component-normalization", "n_violations": len(violations),
            "first_violation": violations[0] if violations else None, "evidence": violations[:3]}


CHECKS = {
    "ID1": check_id1, "CC1": check_cc1, "PE1": check_pe1, "AV1": check_av1,
    "TA1": check_ta1, "SC1": check_sc1, "CW1": check_cw1, "LN1": check_ln1,
}


def main():
    out = {"experiment": "trainaudit_inline_d2_new", "results": []}
    for bug, fn in CHECKS.items():
        for variant in ["buggy", "fixed"]:
            r = fn(variant)
            verdict = "DETECTED" if r["n_violations"] > 0 else "CLEAN"
            entry = {"bug_id": bug, "variant": variant, "verdict": verdict, **r}
            out["results"].append(entry)
            print(f"[{bug}/{variant:<5}] {verdict:<8} {r['rule']:<40} violations={r['n_violations']}")

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

    p = RESULTS / "trainaudit_d2_new_inline.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {p}")
    print("\nPer-bug summary:")
    for bug, s in summary.items():
        print(f"  {bug}: buggy_detected={'✓' if s['buggy_detected'] else '✗'}  "
              f"fixed_fp={'✓' if s['fixed_fp'] else '✗'}  rule={s['rule']}")


if __name__ == "__main__":
    main()
