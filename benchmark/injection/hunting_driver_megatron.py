"""Megatron hunting driver — wraps pretrain_gpt.py with trainaudit.

Reads pretrain_gpt.py from the Megatron-LM checkout (cwd at run time),
hooks train_step so the first iteration triggers snapshot_build + rule
evaluation. Configuration is steered through Megatron CLI flags passed
in argv (the run.sh in benchmark/eval/hunt.py builds those flags from
the matrix env vars).

Env knobs:
  HUNT_TIER  T0_PYTORCH | T1_FW_METADATA  (default T1_FW_METADATA)

Required CLI args (passed by hunt.py launcher): --num-layers, --hidden-size,
--num-attention-heads, --seq-length, --max-position-embeddings, etc. —
see hunt.py::megatron_args for the matrix.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "trainaudit_pkg"))
import trainaudit  # noqa: E402


def _tier(name: str):
    name = (name or "T1_FW_METADATA").upper()
    if hasattr(trainaudit.Tier, name):
        return getattr(trainaudit.Tier, name)
    return trainaudit.Tier.T1_FW_METADATA


trainaudit.enable(tier=_tier(os.environ.get("HUNT_TIER", "T1_FW_METADATA")),
                   db_path=":memory:")

_ts_mod = None
for _name in ("megatron.training.training", "megatron.training"):
    try:
        _ts_mod = __import__(_name, fromlist=["train_step"])
        if hasattr(_ts_mod, "train_step"):
            break
        _ts_mod = None
    except ImportError:
        _ts_mod = None
if _ts_mod is None:
    raise RuntimeError("Megatron train_step not importable from cwd")

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
        except Exception as e:
            print(f"[HUNT] snapshot_build failed: {e}")

    result = _orig_ts(*args, **kwargs)

    if not _done[0]:
        _done[0] = True
        import torch
        rank = (torch.distributed.get_rank()
                 if torch.distributed.is_initialized() else 0)
        if rank == 0:
            results = trainaudit.run_rules()
            print(trainaudit.summarize(results))
            violated = [r for r in results if r.violated]
            if violated:
                print(f"\n[HUNT/trainaudit] BUG DETECTED via "
                      f"{len(violated)} rule(s):")
                for v in violated:
                    print(f"   - {v.rule_id}: {v.message}")
                    if v.evidence:
                        print(f"     evidence: {v.evidence}")
            else:
                print("\n[HUNT/trainaudit] CLEAN: no rule violations")
    return result


_ts_mod.train_step = _patched_ts
print("[HUNT] hooked Megatron train_step + trainaudit enabled")

exec(open("pretrain_gpt.py").read())
