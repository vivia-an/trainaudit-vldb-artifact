"""B1 via TrainAudit Phase 2 (T1 framework metadata).

Trigger: Megatron pretrain with TP=2 + num-experts=2. Router weight is
initialized via torch.nn.Linear default RNG → diverges across TP ranks
on buggy commit; goes through get_data_parallel_rng_tracker_name() fork
on fixed → equal across TP ranks.

T1 detection: Megatron adapter labels each non-TP-shard param as
'replica' (replica_group = TP group). build_snapshot does an all_gather
of cksums in the replica group. T1-replica-cksum-equal compares.
"""
import atexit
import os
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "trainaudit_pkg"))

import trainaudit

_db_path_template = os.environ.get("TRAINAUDIT_DB_PATH", ":memory:")
if _db_path_template != ":memory:":
    _rank = int(os.environ.get("RANK", "0"))
    _db_path = _db_path_template.replace("{rank}", str(_rank))
    os.makedirs(os.path.dirname(_db_path) or ".", exist_ok=True)
else:
    _db_path = _db_path_template
trainaudit.enable(tier=trainaudit.Tier.T1_FW_METADATA, db_path=_db_path)
atexit.register(lambda: trainaudit.is_enabled() and trainaudit.disable())

# Hook train_step to run rules + report after first step
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
    # Snapshot model once per first invocation (the model is now built)
    if not _done[0]:
        try:
            model_arg = args[2] if len(args) > 2 else kwargs.get("model")
            m = model_arg[0] if isinstance(model_arg, list) else model_arg
            while hasattr(m, "module"):
                m = m.module
            trainaudit.snapshot_build(m, None)
        except Exception as e:
            print(f"[B1] snapshot_build failed: {e}")

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
                print(f"\n[B1/trainaudit] BUG DETECTED via {len(violated)} rule(s):")
                for v in violated:
                    print(f"   - {v.rule_id}: {v.message}")
                    if v.evidence:
                        print(f"     evidence: {v.evidence}")
            else:
                print("\n[B1/trainaudit] CLEAN: no rule violations")
    return result


_ts_mod.train_step = _patched_ts
print("[B1] hooked train_step + trainaudit T1 enabled")

exec(open("pretrain_gpt.py").read())
