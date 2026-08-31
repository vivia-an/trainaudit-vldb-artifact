"""Naïve metric-monitoring baseline for the case-aligned surrogates.

Tracks four scalar signals during surrogate training and flags any anomaly as
a detection.

Signals:
  loss_spike     : loss[t] > k1 * median(loss[t-W:t])         (k1=10, W=20)
  loss_nan       : not isfinite(loss[t])
  grad_nan       : any non-finite element in any gradient
  gradnorm_spike : grad_norm[t] > k2 * max(grad_norm[t-W:t])  (k2=10, W=20)

Output schema:
  bug_id, framework, category, phase, set,
  verdict, trigger_signal, trigger_step, duration_s, note
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_ROOT = REPO_ROOT / "benchmark" / "eval"

K_SPIKE = 10.0
WINDOW = 20

CSV_FIELDS = [
    "bug_id", "framework", "category", "phase", "set",
    "verdict", "trigger_signal", "trigger_step", "duration_s", "note",
]


# ---- Naïve detector ---------------------------------------------------------


def _naive_detector(losses: List[float],
                    grad_norms: List[float],
                    grad_nan_step: Optional[int]) -> Tuple[str, str, int]:
    """Return (verdict, trigger_signal, trigger_step). 'CLEAN' if nothing fires."""
    for t, lv in enumerate(losses):
        if not math.isfinite(lv):
            return "DETECTED", "loss_nan", t
    if grad_nan_step is not None:
        return "DETECTED", "grad_nan", grad_nan_step
    for t in range(WINDOW, len(losses)):
        window = losses[t - WINDOW:t]
        if not window:
            continue
        med = statistics.median(window)
        if med > 0 and losses[t] > K_SPIKE * med:
            return "DETECTED", "loss_spike", t
    for t in range(WINDOW, len(grad_norms)):
        window = grad_norms[t - WINDOW:t]
        if not window:
            continue
        ref = max(window)
        if ref > 0 and grad_norms[t] > K_SPIKE * ref:
            return "DETECTED", "gradnorm_spike", t
    return "CLEAN", "", -1


# ---- Naïve runners (T0 only — T1 bugs have no metric stream) ----------------


def _capture_metrics(train_fn: Callable[[Callable], None]) -> Tuple[List[float], List[float], Optional[int]]:
    """train_fn receives a record callback and is responsible for emitting
    one (loss_value, grad_norm_value) pair per step.
    """
    losses: List[float] = []
    grad_norms: List[float] = []
    grad_nan_step: List[Optional[int]] = [None]

    def record(loss_val: float, gn_val: float, step: int,
               grad_has_nan: bool = False) -> None:
        losses.append(loss_val)
        grad_norms.append(gn_val)
        if grad_has_nan and grad_nan_step[0] is None:
            grad_nan_step[0] = step

    train_fn(record)
    return losses, grad_norms, grad_nan_step[0]


def _run_torch_loop(buggy_setup: Callable, n_steps: int,
                    record: Callable[[float, float, int, bool], None]) -> None:
    """Common torch training-loop driver. `buggy_setup(record)` returns
    (model, opt, step_fn). step_fn runs ONE training step and returns
    (loss_value, grad_norm_value, grad_has_nan).
    """
    import torch
    torch.manual_seed(0)
    model, opt, step_fn = buggy_setup()
    for step in range(n_steps):
        opt.zero_grad()
        loss_v, gn_v, grad_nan = step_fn()
        record(loss_v, gn_v, step, grad_nan)


def _grad_stats(model) -> Tuple[float, bool]:
    import torch
    sq = 0.0
    has_nan = False
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            has_nan = True
        sq += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(sq), has_nan


# ---- Bug-specific Naïve runners --------------------------------------------


def _naive_run_b11(buggy: bool):
    import torch
    import torch.nn as nn
    import torch.nn.utils as nn_utils
    import torch.optim as optim

    real_clip = nn_utils.clip_grad_norm_

    def buggy_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        params = [p for p in parameters if p is not None and p.grad is not None]
        if not params:
            return torch.tensor(0.0)
        return torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad.detach().float(), norm_type)
             for p in params]), norm_type)

    if buggy:
        nn_utils.clip_grad_norm_ = buggy_clip
    try:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
        opt = optim.AdamW(model.parameters(), lr=1e-3)

        losses, grad_norms, grad_nan_step = [], [], None
        for step in range(25):
            opt.zero_grad()
            x = torch.randn(2, 8)
            loss = (model(x).pow(2).sum() * 1e6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            gn, gn_nan = _grad_stats(model)
            losses.append(loss.item())
            grad_norms.append(gn)
            if gn_nan and grad_nan_step is None:
                grad_nan_step = step
            opt.step()
        return losses, grad_norms, grad_nan_step
    finally:
        nn_utils.clip_grad_norm_ = real_clip


def _naive_run_b12(buggy: bool):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    if buggy:
        for g in opt.param_groups:
            g.pop("initial_lr", None)
    else:
        for g in opt.param_groups:
            g.setdefault("initial_lr", g["lr"])
    try:
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, last_epoch=10)
    except KeyError:
        sched = None

    losses, grad_norms, grad_nan_step = [], [], None
    for step in range(25):
        opt.zero_grad()
        x = torch.randn(2, 8)
        loss = model(x).pow(2).sum()
        loss.backward()
        gn, gn_nan = _grad_stats(model)
        losses.append(loss.item())
        grad_norms.append(gn)
        if gn_nan and grad_nan_step is None:
            grad_nan_step = step
        opt.step()
        if sched is not None:
            sched.step()
    return losses, grad_norms, grad_nan_step


def _naive_run_m014(buggy: bool):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class DegenerateRouter(nn.Module):
        def __init__(self, hidden, n=4):
            super().__init__()
            self.linear = nn.Linear(hidden, n)
        def forward(self, x):
            s = self.linear(x)
            t, _ = s.topk(1, dim=-1)
            return torch.softmax(t, dim=-1)

    class CleanRouter(nn.Module):
        def __init__(self, hidden, n=4):
            super().__init__()
            self.linear = nn.Linear(hidden, n)
        def forward(self, x):
            return torch.softmax(self.linear(x), dim=-1)

    torch.manual_seed(0)
    router_cls = DegenerateRouter if buggy else CleanRouter
    model = nn.Sequential(nn.Linear(8, 16), router_cls(16))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    losses, grad_norms, grad_nan_step = [], [], None
    for step in range(25):
        opt.zero_grad()
        x = torch.randn(2, 8)
        loss = model(x).pow(2).sum()
        loss.backward()
        gn, gn_nan = _grad_stats(model)
        losses.append(loss.item())
        grad_norms.append(gn)
        if gn_nan and grad_nan_step is None:
            grad_nan_step = step
        opt.step()
    return losses, grad_norms, grad_nan_step


def _naive_run_o_new_1(buggy: bool):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class BrokenRMSNorm(nn.LayerNorm):
        def forward(self, x):
            return super().forward(x) * 0.33

    torch.manual_seed(0)
    norm_cls = BrokenRMSNorm if buggy else nn.LayerNorm
    model = nn.Sequential(nn.Linear(8, 16), norm_cls(16), nn.GELU(), nn.Linear(16, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    losses, grad_norms, grad_nan_step = [], [], None
    for step in range(25):
        opt.zero_grad()
        x = torch.randn(2, 8)
        loss = model(x).pow(2).sum()
        loss.backward()
        gn, gn_nan = _grad_stats(model)
        losses.append(loss.item())
        grad_norms.append(gn)
        if gn_nan and grad_nan_step is None:
            grad_nan_step = step
        opt.step()
    return losses, grad_norms, grad_nan_step


def _naive_run_oc_new_2(buggy: bool):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class FrozenStepAdamW(optim.AdamW):
        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = torch.tensor(0.0)
                    p.data.add_(p.grad, alpha=-group["lr"])
            return None

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
    opt_cls = FrozenStepAdamW if buggy else optim.AdamW
    opt = opt_cls(model.parameters(), lr=1e-3)
    if buggy:
        for p in model.parameters():
            opt.state[p]["step"] = torch.tensor(0.0)

    losses, grad_norms, grad_nan_step = [], [], None
    for step in range(25):
        opt.zero_grad()
        x = torch.randn(2, 8)
        loss = model(x).pow(2).sum()
        loss.backward()
        gn, gn_nan = _grad_stats(model)
        losses.append(loss.item())
        grad_norms.append(gn)
        if gn_nan and grad_nan_step is None:
            grad_nan_step = step
        opt.step()
    return losses, grad_norms, grad_nan_step


def _naive_run_o_005(buggy: bool):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils import checkpoint as cp

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.dropout = nn.Dropout(p=0.5)
        def forward(self, x):
            return self.dropout(self.linear(x))

    class Top(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(8, 8)
            self.blk = Block()
            self.head = nn.Linear(8, 4)
        def forward(self, x):
            h = self.embed(x)
            h = cp.checkpoint(self.blk, h, use_reentrant=False,
                              preserve_rng_state=(not buggy))
            return self.head(h)

    torch.manual_seed(0)
    model = Top()
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    losses, grad_norms, grad_nan_step = [], [], None
    for step in range(25):
        opt.zero_grad()
        x = torch.randn(2, 8, requires_grad=True)
        loss = model(x).pow(2).sum()
        loss.backward()
        gn, gn_nan = _grad_stats(model)
        losses.append(loss.item())
        grad_norms.append(gn)
        if gn_nan and grad_nan_step is None:
            grad_nan_step = step
        opt.step()
    return losses, grad_norms, grad_nan_step


def _naive_run_o_new_9(buggy: bool):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    class Ds(Dataset):
        def __init__(self, n=24, seq_len=16, vocab=50000, buggy=False):
            self.n, self.seq_len, self.vocab, self.buggy = n, seq_len, vocab, buggy
        def __len__(self): return self.n
        def __getitem__(self, idx):
            if self.buggy:
                return torch.full((self.seq_len,), self.vocab + 100, dtype=torch.int64)
            return torch.randint(0, self.vocab, (self.seq_len,), dtype=torch.int64)

    torch.manual_seed(0)
    vocab = 50000
    model = nn.Embedding(vocab + 200, 8)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    dl = DataLoader(Ds(buggy=buggy, vocab=vocab), batch_size=2)

    losses, grad_norms, grad_nan_step = [], [], None
    for step, batch in enumerate(dl):
        opt.zero_grad()
        loss = model(batch).pow(2).sum()
        loss.backward()
        gn, gn_nan = _grad_stats(model)
        losses.append(loss.item())
        grad_norms.append(gn)
        if gn_nan and grad_nan_step is None:
            grad_nan_step = step
        opt.step()
    return losses, grad_norms, grad_nan_step


# T1 bugs: synthetic_runners emits fake events instead of real training, so
# Naïve has no metric stream to monitor. We honestly report CLEAN with a note.
T1_BUGS = {"B1", "B2", "B3", "B8",
           "B13", "M-012", "M-NEW-5", "M-024", "M-020", "OC-NEW-3"}


_NAIVE_RUNNERS = {
    "B11": _naive_run_b11,
    "B12": _naive_run_b12,
    "M-014": _naive_run_m014,
    "O-NEW-1": _naive_run_o_new_1,
    "OC-NEW-2": _naive_run_oc_new_2,
    "O-005": _naive_run_o_005,
    "O-NEW-9": _naive_run_o_new_9,
}


# ---- Driver ----------------------------------------------------------------


def run_synthetic(bug_id: str, phase: str) -> Dict[str, Any]:
    t0 = time.time()
    buggy = (phase == "buggy")
    if bug_id in T1_BUGS:
        return {
            "bug_id": bug_id, "phase": phase,
            "verdict": "CLEAN",
            "trigger_signal": "",
            "trigger_step": -1,
            "duration_s": time.time() - t0,
            "note": "T1-tier surrogate emits framework metadata events, not "
                     "metric stream — naive monitor has nothing to observe",
        }
    runner = _NAIVE_RUNNERS.get(bug_id)
    if runner is None:
        return {
            "bug_id": bug_id, "phase": phase,
            "verdict": "FAIL",
            "trigger_signal": "",
            "trigger_step": -1,
            "duration_s": time.time() - t0,
            "note": f"no naive runner for {bug_id}",
        }
    try:
        losses, grad_norms, grad_nan_step = runner(buggy)
        verdict, trig_sig, trig_step = _naive_detector(
            losses, grad_norms, grad_nan_step)
        return {
            "bug_id": bug_id, "phase": phase,
            "verdict": verdict,
            "trigger_signal": trig_sig,
            "trigger_step": trig_step,
            "duration_s": time.time() - t0,
            "note": (f"steps={len(losses)}; loss[0]={losses[0]:.3g} "
                      f"loss[-1]={losses[-1]:.3g} max_gn={max(grad_norms or [0]):.3g}"),
        }
    except Exception as e:  # noqa: BLE001
        import traceback
        return {
            "bug_id": bug_id, "phase": phase,
            "verdict": "FAIL",
            "trigger_signal": "",
            "trigger_step": -1,
            "duration_s": time.time() - t0,
            "note": f"runner crashed: {type(e).__name__}: {e}; "
                      f"{traceback.format_exc()[:300]}",
        }


def run_subset(subset_path: Path, manifest_path: Path,
                set_label: str) -> List[Dict[str, Any]]:
    subset = json.loads(subset_path.read_text())
    bug_ids = subset["bugs"]
    manifest_data = json.loads(manifest_path.read_text())
    manifest_rows = manifest_data.get("bugs", manifest_data)
    manifest = {r["bug_id"]: r for r in manifest_rows}

    rows: List[Dict[str, Any]] = []
    for bug_id in bug_ids:
        meta = manifest.get(bug_id, {})
        framework = meta.get("framework", "?")
        category = meta.get("category", "?")
        for phase in ("buggy", "fixed"):
            res = run_synthetic(bug_id, phase)
            rows.append({
                "bug_id": res["bug_id"],
                "framework": framework,
                "category": category,
                "phase": phase,
                "set": set_label,
                "verdict": res["verdict"],
                "trigger_signal": res["trigger_signal"],
                "trigger_step": res["trigger_step"],
                "duration_s": f"{res['duration_s']:.2f}",
                "note": res["note"],
            })
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(HARNESS_ROOT / "synthetic_17.json"))
    ap.add_argument("--manifest", default=str(HARNESS_ROOT / "manifest_v2.json"))
    ap.add_argument("--out",
                    default=str(HARNESS_ROOT / "baseline_naive_results.csv"))
    args = ap.parse_args()

    rows = run_subset(Path(args.subset), Path(args.manifest), set_label="surrogate")
    write_csv(rows, Path(args.out))
    n_buggy = sum(1 for r in rows if r["phase"] == "buggy")
    n_buggy_det = sum(1 for r in rows
                       if r["phase"] == "buggy" and r["verdict"] == "DETECTED")
    n_fixed = sum(1 for r in rows if r["phase"] == "fixed")
    n_fixed_clean = sum(1 for r in rows
                         if r["phase"] == "fixed" and r["verdict"] == "CLEAN")
    print(f"-> {args.out}")
    print(f"buggy detected: {n_buggy_det}/{n_buggy}")
    print(f"fixed clean (1-FP): {n_fixed_clean}/{n_fixed}")


if __name__ == "__main__":
    main()
