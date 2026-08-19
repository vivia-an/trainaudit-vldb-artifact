"""In-process bug surrogate runners.

Each runner builds a real PyTorch model, optionally injects a bug pattern
that matches the corresponding paper bug fingerprint, runs trainaudit,
and emits the per-bug contract line:
  [BUG_ID] BUG DETECTED: <rule_id>: <message>
  [BUG_ID] CLEAN: <message>

Used by run_all.py --mode synthetic. The same bugs are also covered by
tests/integration/test_real_bug_detection.py, but here we expose them as
named runners so D1 can produce results.csv on a CPU-only machine.

Each runner returns a list of (bug_id, verdict, message) tuples — typically
two: one for the buggy run, one for the clean / fixed run.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure trainaudit is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_TA_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "trainaudit"))
if _TA_ROOT not in sys.path:
    sys.path.insert(0, _TA_ROOT)

import trainaudit  # noqa: E402


def _summary(rule_id: str, results) -> Tuple[str, str]:
    """Return (verdict, message) given the rule_id of interest."""
    for r in results:
        if r.rule_id == rule_id and r.violated:
            return ("BUG DETECTED", f"{rule_id}: {r.message}")
    return ("CLEAN", f"{rule_id} did not fire")


def _phase(bug_id: str, label: str, verdict: str, message: str) -> Tuple[str, str, str]:
    return (bug_id, verdict, f"[{label}] {message}"[:240])


# ---- B11 ----------------------------------------------------------------


def _b11_run(buggy: bool) -> Tuple[str, str]:
    import torch.nn.utils as _nn_utils
    real_clip = _nn_utils.clip_grad_norm_

    def buggy_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        params = [p for p in parameters if p is not None and p.grad is not None]
        if not params:
            return torch.tensor(0.0)
        return torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad.detach().float(), norm_type)
             for p in params]), norm_type)

    if buggy:
        _nn_utils.clip_grad_norm_ = buggy_clip
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                              db_path=os.path.join(tmp, "trace.duckdb"))
            try:
                torch.manual_seed(0)
                model = nn.Sequential(nn.Linear(8, 16), nn.GELU(),
                                       nn.Linear(16, 4))
                opt = optim.AdamW(model.parameters(), lr=1e-3)
                trainaudit.snapshot_build(model, opt)
                x = torch.randn(2, 8)
                (model(x).pow(2).sum() * 1e6).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                opt.step()
                results = trainaudit.run_rules()
                return _summary("T0-clip-grad-bounded", results)
            finally:
                trainaudit.disable()
    finally:
        _nn_utils.clip_grad_norm_ = real_clip


def run_b11() -> List[Tuple[str, str, str]]:
    bv, bm = _b11_run(buggy=True)
    fv, fm = _b11_run(buggy=False)
    return [_phase("B11", "buggy", bv, bm), _phase("B11", "fixed", fv, fm)]


# ---- O-NEW-1 ------------------------------------------------------------


class _BrokenRMSNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x) * 0.33


def _olmo_norm_run(buggy: bool) -> Tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            norm_cls = _BrokenRMSNorm if buggy else nn.LayerNorm
            model = nn.Sequential(nn.Linear(8, 16), norm_cls(16),
                                   nn.GELU(), nn.Linear(16, 4))
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            for _ in range(3):
                x = torch.randn(2, 8)
                model(x).pow(2).sum().backward()
                opt.step()
            return _summary("T0-norm-output-unit-rms", trainaudit.run_rules())
        finally:
            trainaudit.disable()


def run_o_new_1() -> List[Tuple[str, str, str]]:
    bv, bm = _olmo_norm_run(True)
    fv, fm = _olmo_norm_run(False)
    return [_phase("O-NEW-1", "buggy", bv, bm),
            _phase("O-NEW-1", "fixed", fv, fm)]


# ---- OC-NEW-2 -----------------------------------------------------------


class _FrozenStepAdamW(optim.AdamW):
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


def _ocn2_run(buggy: bool) -> Tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.GELU(),
                                   nn.Linear(16, 4))
            opt_cls = _FrozenStepAdamW if buggy else optim.AdamW
            opt = opt_cls(model.parameters(), lr=1e-3)
            if buggy:
                # The frozen AdamW only touches state['step'] — pre-populate
                # so the trainaudit hook always has it to capture.
                for p in model.parameters():
                    opt.state[p]["step"] = torch.tensor(0.0)
            # Healthy AdamW initialises full state lazily on first step()
            trainaudit.snapshot_build(model, opt)
            for _ in range(4):
                opt.zero_grad()  # clear stale grads BEFORE backward
                x = torch.randn(2, 8)
                model(x).pow(2).sum().backward()
                opt.step()
            return _summary("T0-optim-step-counter-monotonic",
                             trainaudit.run_rules())
        finally:
            trainaudit.disable()


def run_oc_new_2() -> List[Tuple[str, str, str]]:
    bv, bm = _ocn2_run(True)
    fv, fm = _ocn2_run(False)
    return [_phase("OC-NEW-2", "buggy", bv, bm),
            _phase("OC-NEW-2", "fixed", fv, fm)]


# ---- M-014 (degenerate softmax) ----------------------------------------


class _DegenerateRouter(nn.Module):
    def __init__(self, hidden, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts)

    def forward(self, x):
        scores = self.linear(x)
        topk_scores, _ = scores.topk(1, dim=-1)
        return torch.softmax(topk_scores, dim=-1)


class _CleanRouter(nn.Module):
    def __init__(self, hidden, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


def _m014_run(buggy: bool) -> Tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            router_cls = _DegenerateRouter if buggy else _CleanRouter
            model = nn.Sequential(nn.Linear(8, 16), router_cls(16))
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            for _ in range(2):
                x = torch.randn(2, 8)
                model(x).pow(2).sum().backward()
                opt.step()
            # name the routers so the rule precondition matches '%Router%'
            return _summary("T0-softmax-degenerate", trainaudit.run_rules())
        finally:
            trainaudit.disable()


def run_m_014() -> List[Tuple[str, str, str]]:
    bv, bm = _m014_run(True)
    fv, fm = _m014_run(False)
    return [_phase("M-014", "buggy", bv, bm),
            _phase("M-014", "fixed", fv, fm)]


# ---- B12 ----------------------------------------------------------------


def _b12_run(buggy: bool) -> Tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            if buggy:
                for g in opt.param_groups:
                    g.pop("initial_lr", None)
            else:
                # healthy resume path: optimizer config sets initial_lr
                for g in opt.param_groups:
                    g.setdefault("initial_lr", g["lr"])
            try:
                _ = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=100, last_epoch=10)
            except KeyError:
                pass
            return _summary("T0-initial-lr-present", trainaudit.run_rules())
        finally:
            trainaudit.disable()


def run_b12() -> List[Tuple[str, str, str]]:
    bv, bm = _b12_run(True)
    fv, fm = _b12_run(False)
    return [_phase("B12", "buggy", bv, bm),
            _phase("B12", "fixed", fv, fm)]


# ---- registry ----------------------------------------------------------


_RUNNERS = {
    "B11":      run_b11,
    "B12":      run_b12,
    "O-NEW-1":  run_o_new_1,
    "OC-NEW-2": run_oc_new_2,
    "M-014":    run_m_014,
}


# ---- T1-tier bugs: emit adapter-probe events directly ===================
# These bugs depend on framework adapters' active probes (residual.probe,
# decay.probe, jitter.probe) or semantic labelling (semantic.is_router,
# semantic.expert_bias_dtype). The real adapters live under
# trainaudit.adapters.{megatron,olmo,olmo_core}; for the synthetic harness
# we emit the same probe / build-snapshot event payloads directly so the
# rule path is the same as on a live framework run.


def _emit_and_check(bug_id: str, rule_id: str,
                     buggy_payloads, clean_payloads,
                     hookpoints) -> List[Tuple[str, str, str]]:
    """Run trainaudit twice (buggy then clean), emit synthetic events,
    summarise via run_rules. Used by every T1 surrogate below."""
    out: List[Tuple[str, str, str]] = []
    for label, payloads in (("buggy", buggy_payloads),
                              ("fixed", clean_payloads)):
        with tempfile.TemporaryDirectory() as tmp:
            store = trainaudit.enable(tier=trainaudit.Tier.T1_FW_METADATA,
                                       db_path=os.path.join(tmp, "trace.duckdb"))
            try:
                # Always emit a build.snapshot first so rules with
                # framework_invariants preconditions can fire
                if any(hp == "build.snapshot" for hp, _ in payloads):
                    pass  # caller already includes one
                else:
                    store.emit("build.snapshot", {
                        "model": {"n_parameters": 1, "n_modules": 1},
                        "framework_invariants": {},
                    })
                for hp, p in payloads:
                    store.emit(hp, p)
                store.flush()
                v, m = _summary(rule_id, trainaudit.run_rules())
                out.append(_phase(bug_id, label, v, m))
            finally:
                trainaudit.disable()
    return out


def run_b13() -> List[Tuple[str, str, str]]:
    """B13/O-002 surrogate: OLMo residual block clobbered the residual
    variable so output is closer to the normalized input than to the
    original input."""
    buggy = [("residual.probe", {
        "block_class": "OLMoSequentialBlock",
        "d_to_original_input": 5.0,
        "d_to_normed_input": 0.5,  # output closer to normed → bug
    })]
    clean = [("residual.probe", {
        "block_class": "OLMoSequentialBlock",
        "d_to_original_input": 0.4,
        "d_to_normed_input": 4.0,  # output closer to original → healthy
    })]
    return _emit_and_check("B13", "T1-residual-stream-preserved",
                            buggy, clean,
                            ["residual.probe"])


def run_m_012() -> List[Tuple[str, str, str]]:
    """M-012 surrogate: TopKRouter.expert_bias silently demoted to bf16."""
    buggy = [("module.fwd.post", {
        "module_class": "TopKRouter",
        "module_id": 1, "module_name": "router",
        "is_normalizer": False, "training": True,
        "semantic": {"is_router": True, "expert_bias_dtype": "bfloat16"},
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 1.0},
    })]
    clean = [("module.fwd.post", {
        "module_class": "TopKRouter",
        "module_id": 1, "module_name": "router",
        "is_normalizer": False, "training": True,
        "semantic": {"is_router": True, "expert_bias_dtype": "float32"},
        "output": {"dtype": "torch.float32", "shape": [4, 8],
                   "l2_norm": 1.0},
    })]
    return _emit_and_check("M-012", "T1-expert-bias-fp32",
                            buggy, clean, ["module.fwd.post"])


def run_m_new_5() -> List[Tuple[str, str, str]]:
    """M-NEW-5 surrogate: MoE Router missing calculate_per_token_loss attr.
    Precondition: --calculate-per-token-loss enabled in framework config.
    """
    common_bs = ("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {
            "megatron": {"calculate_per_token_loss": True},
        },
    })
    buggy = [
        common_bs,
        ("module.fwd.post", {
            "module_class": "TopKRouter",
            "module_id": 1, "module_name": "router",
            "is_normalizer": False, "training": True,
            "semantic": {"is_router": True,
                          "has_calculate_per_token_loss": False},
            "output": {"dtype": "torch.float32", "shape": [4, 8],
                       "l2_norm": 1.0},
        }),
    ]
    clean = [
        common_bs,
        ("module.fwd.post", {
            "module_class": "TopKRouter",
            "module_id": 1, "module_name": "router",
            "is_normalizer": False, "training": True,
            "semantic": {"is_router": True,
                          "has_calculate_per_token_loss": True},
            "output": {"dtype": "torch.float32", "shape": [4, 8],
                       "l2_norm": 1.0},
        }),
    ]
    return _emit_and_check("M-NEW-5",
                            "T1-router-has-calculate-per-token-loss",
                            buggy, clean, ["module.fwd.post"])


def run_m_024() -> List[Tuple[str, str, str]]:
    """M-024 surrogate: Megatron apply_input_jitter promotes bf16 → fp32
    via torch.distributions.Uniform."""
    buggy = [("jitter.probe", {
        "module_class": "TopKRouter",
        "input_dtype": "bfloat16",
        "output_dtype": "float32",  # promoted → bug
        "dtypes_match": False,
    })]
    clean = [("jitter.probe", {
        "module_class": "TopKRouter",
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "dtypes_match": True,
    })]
    return _emit_and_check("M-024", "T1-jitter-preserves-dtype",
                            buggy, clean, ["jitter.probe"])


def run_oc_new_3() -> List[Tuple[str, str, str]]:
    """OC-NEW-3 surrogate: OLMo-core _sqrt_decay direction inverted —
    decay is slow-then-fast instead of fast-then-slow.

    The rule fits slope at progress=0 (low) vs progress=1 (high). For
    a correct schedule, |slope at start of training| (high progress
    in our convention) >> |slope at end|. For the bug, the opposite
    holds. Emit 5+ probe events spanning the progress axis."""
    # progress is step_from_end / decay; progress=1 = start, progress=0 = end
    buggy = [
        ("decay.probe", {"kind": "sqrt_decay", "progress": p,
                          "result": r, "initial_lr": 1.0,
                          "decay_min_lr": 0.0})
        for p, r in [(0.0, 0.0), (0.1, 0.32), (0.3, 0.55),
                      (0.6, 0.77), (0.9, 0.95), (1.0, 1.0)]
        # buggy curve: result = sqrt(progress) — slow then fast; slope at
        # high progress is shallow, slope at low progress is steep
    ]
    clean = [
        ("decay.probe", {"kind": "sqrt_decay", "progress": p,
                          "result": r, "initial_lr": 1.0,
                          "decay_min_lr": 0.0})
        for p, r in [(0.0, 0.0), (0.1, 0.05), (0.3, 0.18),
                      (0.6, 0.45), (0.9, 0.78), (1.0, 1.0)]
        # correct curve: 1 - sqrt(1 - progress) — fast then slow over real
        # time; slope at progress=1 (start) is steep, slope at progress=0
        # (end) is shallow
    ]
    return _emit_and_check("OC-NEW-3", "T1-sqrt-decay-front-loaded",
                            buggy, clean, ["decay.probe"])


def run_m_020() -> List[Tuple[str, str, str]]:
    """M-020 surrogate: PP layer count silently dropped by integer
    division — declared num_layers % pp_size != 0."""
    buggy = [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {
            "megatron": {
                "num_layers": 24,
                "pipeline_model_parallel_size": 5,  # 24 % 5 = 4
                "n_transformer_layers_in_local_module": 4,  # 4*5=20 ≠ 24
            },
        },
    })]
    clean = [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {
            "megatron": {
                "num_layers": 24,
                "pipeline_model_parallel_size": 4,  # divides
                "n_transformer_layers_in_local_module": 6,  # 6*4=24 ✓
            },
        },
    })]
    return _emit_and_check("M-020", "T1-layer-count-strict",
                            buggy, clean, ["build.snapshot"])


def run_b1() -> List[Tuple[str, str, str]]:
    """B1/M-005 surrogate: replica param diverged across replica group."""
    buggy = [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {},
        "cross_rank_cksums": [{
            "name": "router.weight",
            "group_size": 4,
            "local_cksum": 100,
            "gathered_cksums": [100, 100, 999, 100],  # rank 2 outlier
            "all_equal": False,
        }],
    })]
    clean = [("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {},
        "cross_rank_cksums": [{
            "name": "router.weight",
            "group_size": 4,
            "local_cksum": 100,
            "gathered_cksums": [100, 100, 100, 100],
            "all_equal": True,
        }],
    })]
    return _emit_and_check("B1", "T1-replica-cksum-equal",
                            buggy, clean, ["build.snapshot"])


_RUNNERS.update({
    "B1":       run_b1,
    "B13":      run_b13,
    "M-012":    run_m_012,
    "M-NEW-5":  run_m_new_5,
    "M-024":    run_m_024,
    "OC-NEW-3": run_oc_new_3,
    "M-020":    run_m_020,
})


# ---- O-005: torch.utils.checkpoint preserve_rng_state=False with Dropout ====


def _o005_run(buggy: bool) -> Tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            from torch.utils import checkpoint as _cp

            class BlockWithDropout(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(8, 8)
                    self.dropout = nn.Dropout(p=0.5)
                def forward(self, x):
                    return self.dropout(self.linear(x))

            class TopModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Linear(8, 8)
                    self.blk = BlockWithDropout()
                    self.head = nn.Linear(8, 4)
                def forward(self, x):
                    h = self.embed(x)
                    # buggy: preserve_rng_state=False with Dropout present
                    # fixed:  preserve_rng_state=True
                    h = _cp.checkpoint(self.blk, h,
                                        use_reentrant=False,
                                        preserve_rng_state=(not buggy))
                    return self.head(h)

            model = TopModel()
            opt = optim.AdamW(model.parameters(), lr=1e-3)
            trainaudit.snapshot_build(model, opt)
            opt.zero_grad()
            x = torch.randn(2, 8, requires_grad=True)
            model(x).pow(2).sum().backward()
            opt.step()
            return _summary("T0-checkpoint-preserve-rng", trainaudit.run_rules())
        finally:
            trainaudit.disable()


def run_o_005() -> List[Tuple[str, str, str]]:
    bv, bm = _o005_run(True)
    fv, fm = _o005_run(False)
    return [_phase("O-005", "buggy", bv, bm),
            _phase("O-005", "fixed", fv, fm)]


# ---- O-NEW-9: DataLoader yields truncated token ids =========================


def _onew9_run(buggy: bool) -> Tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH,
                          db_path=os.path.join(tmp, "trace.duckdb"))
        try:
            torch.manual_seed(0)
            from torch.utils.data import DataLoader, Dataset

            class TruncatingTokenDataset(Dataset):
                def __init__(self, n=4, seq_len=16, vocab=50_000,
                             buggy_truncate: bool = False):
                    self.n = n
                    self.seq_len = seq_len
                    self.vocab = vocab
                    # buggy: silently overflow int range to ~5_000_000
                    self.buggy_truncate = buggy_truncate
                def __len__(self):
                    return self.n
                def __getitem__(self, idx):
                    if self.buggy_truncate:
                        return torch.full((self.seq_len,), 5_000_000,
                                          dtype=torch.int64)
                    return torch.randint(0, self.vocab, (self.seq_len,),
                                          dtype=torch.int64)

            ds = TruncatingTokenDataset(buggy_truncate=buggy)
            dl = DataLoader(ds, batch_size=2)
            for batch in dl:
                _ = batch  # iterate to trigger dataloader hook
            return _summary("T0-token-id-in-vocab", trainaudit.run_rules())
        finally:
            trainaudit.disable()


def run_o_new_9() -> List[Tuple[str, str, str]]:
    bv, bm = _onew9_run(True)
    fv, fm = _onew9_run(False)
    return [_phase("O-NEW-9", "buggy", bv, bm),
            _phase("O-NEW-9", "fixed", fv, fm)]


_RUNNERS.update({
    "O-005":    run_o_005,
    "O-NEW-9":  run_o_new_9,
})


# ---- B2: TP frozen-weight grad missing all-reduce ===========================
# In TP > 1, LinearWithFrozenWeight's backward should all-reduce the input
# grad across TP rank. Buggy commit skips this → upstream replica params
# end up with different grads on different TP ranks. We synthesise the
# observable signature: cross_rank_grad_cksums entries with all_equal=False
# at optim.step.pre. Detection: T1-grad-replica-cksum-equal.


def run_b2() -> List[Tuple[str, str, str]]:
    buggy = [
        ("optim.step.pre", {
            "kind": "optim_step", "optimizer_class": "AdamW",
            "param_groups": [], "n_params": 4,
            "total_grad_l2": 12.4, "total_param_l2": 100.0,
            "cross_rank_grad_cksums": [{
                "name": "embed.weight", "group_size": 4,
                "local_cksum": 12345,
                "gathered_cksums": [12345, 12345, 99999, 12345],
                "all_equal": False,
            }, {
                "name": "head.bias", "group_size": 4,
                "local_cksum": 100, "gathered_cksums": [100, 100, 100, 100],
                "all_equal": True,
            }],
        }),
    ]
    clean = [
        ("optim.step.pre", {
            "kind": "optim_step", "optimizer_class": "AdamW",
            "param_groups": [], "n_params": 4,
            "total_grad_l2": 12.4, "total_param_l2": 100.0,
            "cross_rank_grad_cksums": [{
                "name": "embed.weight", "group_size": 4,
                "local_cksum": 12345,
                "gathered_cksums": [12345, 12345, 12345, 12345],
                "all_equal": True,
            }, {
                "name": "head.bias", "group_size": 4,
                "local_cksum": 100, "gathered_cksums": [100, 100, 100, 100],
                "all_equal": True,
            }],
        }),
    ]
    return _emit_and_check("B2", "T1-grad-replica-cksum-equal",
                            buggy, clean, ["optim.step.pre"])


_RUNNERS.update({
    "B2": run_b2,
})


# ---- B8: DeepSpeed EP comm group initialised with wrong size ===============


def run_b8() -> List[Tuple[str, str, str]]:
    buggy = [
        ("build.snapshot", {
            "model": {"n_parameters": 100, "n_modules": 5},
            "framework_invariants": {
                "deepspeed": {
                    "ep_size_declared": 2,    # user said EP=2
                    "ep_size_actual":   8,    # but code used num_experts=8
                    "dp_size_declared": 4,
                    "dp_size_actual":   4,
                },
            },
        }),
    ]
    clean = [
        ("build.snapshot", {
            "model": {"n_parameters": 100, "n_modules": 5},
            "framework_invariants": {
                "deepspeed": {
                    "ep_size_declared": 2,
                    "ep_size_actual":   2,
                    "dp_size_declared": 4,
                    "dp_size_actual":   4,
                },
            },
        }),
    ]
    return _emit_and_check("B8", "T1-process-group-size-correct",
                            buggy, clean, ["build.snapshot"])


_RUNNERS.update({"B8": run_b8})


# ---- B3: DeepSpeed BF16 declared but comm uses FP16 ========================


def run_b3() -> List[Tuple[str, str, str]]:
    common_bs = ("build.snapshot", {
        "model": {"n_parameters": 100, "n_modules": 5},
        "framework_invariants": {
            "deepspeed": {"training_dtype": "bfloat16"},
        },
    })
    buggy = [
        common_bs,
        ("comm.post", {
            "kind": "comm", "op": "all_reduce", "group_size": 4,
            "tensor_post": {"dtype": "float16", "shape": [128],
                             "l2_norm": 1.0, "has_nan": False,
                             "has_inf": False, "abs_max": 1.0,
                             "max": 1.0, "min": -1.0, "mean": 0.0},
        }),
    ]
    clean = [
        common_bs,
        ("comm.post", {
            "kind": "comm", "op": "all_reduce", "group_size": 4,
            "tensor_post": {"dtype": "bfloat16", "shape": [128],
                             "l2_norm": 1.0, "has_nan": False,
                             "has_inf": False, "abs_max": 1.0,
                             "max": 1.0, "min": -1.0, "mean": 0.0},
        }),
    ]
    return _emit_and_check("B3", "T1-comm-dtype-matches-training",
                            buggy, clean, ["build.snapshot", "comm.post"])


_RUNNERS.update({"B3": run_b3})


def run_synthetic_bug(bug_id: str) -> List[Tuple[str, str, str]]:
    runner = _RUNNERS.get(bug_id)
    if runner is None:
        return [(bug_id, "FAIL",
                 f"no synthetic runner for {bug_id}; available: "
                 f"{sorted(_RUNNERS)}")]
    try:
        return runner()
    except Exception as e:  # noqa: BLE001
        import traceback
        return [(bug_id, "FAIL",
                 f"runner crashed: {type(e).__name__}: {e}\n"
                 f"{traceback.format_exc()[:800]}")]


def available_bugs() -> List[str]:
    return sorted(_RUNNERS)
