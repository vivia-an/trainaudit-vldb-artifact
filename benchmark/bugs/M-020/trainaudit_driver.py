"""M-020 via TrainAudit T1: actual layer count matches declared num_layers."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "trainaudit_pkg"))
import trainaudit
trainaudit.enable(tier=trainaudit.Tier.T1_FW_METADATA, db_path=":memory:")

# M-020 commit doesn't natively support --mock-data; patch the data loader path
try:
    from megatron.training import utils as _mutils
    _orig_get_blend = _mutils.get_blend_and_blend_per_split
    def _mock_get_blend(args):
        if getattr(args, "mock_data", False):
            return None, None
        return _orig_get_blend(args)
    _mutils.get_blend_and_blend_per_split = _mock_get_blend
except Exception:
    pass

_ts_mod = None
for _name in ("megatron.training.training", "megatron.training"):
    try:
        _ts_mod = __import__(_name, fromlist=["train_step"])
        if hasattr(_ts_mod, "train_step"):
            break
        _ts_mod = None
    except ImportError:
        _ts_mod = None

_orig_ts = _ts_mod.train_step
_done = [False]

def _patched_ts(*args, **kwargs):
    if not _done[0]:
        try:
            model_arg = args[2] if len(args) > 2 else kwargs.get("model")
            m = model_arg[0] if isinstance(model_arg, list) else model_arg
            while hasattr(m, "module"):
                m = m.module
            trainaudit.snapshot_build(m, None)
            # Run rules immediately after snapshot — bug manifests at build time
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                results = trainaudit.run_rules()
                print(trainaudit.summarize(results))
                violated = [r for r in results if r.violated]
                if violated:
                    print(f"\n[M-020/trainaudit-prebuild] BUG DETECTED via {len(violated)} rule(s):")
                    for v in violated:
                        print(f"   - {v.rule_id}: {v.message}")
                        if v.evidence: print(f"     evidence: {v.evidence}")
                else:
                    print("\n[M-020/trainaudit-prebuild] CLEAN at build time")
                _done[0] = True  # don't re-run after train_step
        except Exception as e:
            print(f"[M-020] snapshot_build failed: {e}")
    result = _orig_ts(*args, **kwargs)
    if not _done[0]:
        _done[0] = True
        import torch
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            results = trainaudit.run_rules()
            print(trainaudit.summarize(results))
            violated = [r for r in results if r.violated]
            if violated:
                print(f"\n[M-020/trainaudit] BUG DETECTED via {len(violated)} rule(s):")
                for v in violated:
                    print(f"   - {v.rule_id}: {v.message}")
                    if v.evidence: print(f"     evidence: {v.evidence}")
            else:
                print("\n[M-020/trainaudit] CLEAN: no rule violations")
    return result

_ts_mod.train_step = _patched_ts
print("[M-020] hooked train_step + trainaudit T1")
exec(open("pretrain_gpt.py").read())
