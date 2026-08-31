"""TrainAudit driver for M-NEW-33 (Megatron-LM commit 1505db4cc).

Bug:   `ChainedOptimizer.load_state_dict` calls each chained optimizer's
       `load_state_dict` but does not rebuild `self.param_groups`. After a
       checkpoint resume with EP > 1 the parent ChainedOptimizer keeps the
       pre-load param_groups (stale) while children point at the loaded
       state — downstream code that iterates `self.param_groups` (e.g. LR
       scheduler) reads stale references.
Fix:   After the loop, rebuild
           self.param_groups = []
           for optimizer in self.chained_optimizers:
               self.param_groups += optimizer.param_groups

Detection:
  Mock two chained sub-optimizers whose `load_state_dict` mutates their
  `param_groups` (simulating a checkpoint resume). Call
  `ChainedOptimizer.load_state_dict` on a mock self. Inspect
  `self.param_groups` after the call.
  - parent param_groups identity preserved (stale) -> BUG DETECTED
  - parent param_groups == concatenation of children -> CLEAN

Buggy commit: 1505db4cc~1
Fixed commit: 1505db4cc4e9e94ee22583c76f7e425ea34f5aea
"""
from __future__ import annotations
import os
import sys
import traceback


def _emit(verdict: str, message: str = "") -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    line = f"[M-NEW-33] {verdict}: {message}" if message else f"[M-NEW-33] {verdict}"
    print(line, flush=True)


def main() -> None:
    MEGATRON_DIR = os.environ.get("MEGATRON_DIR", "")
    if MEGATRON_DIR:
        sys.path.insert(0, MEGATRON_DIR)

    try:
        from megatron.core.optimizer.optimizer import ChainedOptimizer
    except Exception as e:
        _emit("FAIL", f"megatron_import: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    method = getattr(ChainedOptimizer, "load_state_dict", None)
    if method is None:
        _emit("FAIL", "method_missing: ChainedOptimizer.load_state_dict not present")
        return

    # Mock two sub-optimizers. Each one's load_state_dict swaps its
    # param_groups to a fresh list (simulating checkpoint reload).
    class _MockSubOpt:
        def __init__(self, original_pg):
            self.param_groups = original_pg

        def load_state_dict(self, state):
            self.param_groups = state["param_groups"]

    sub_a = _MockSubOpt([{"params": [], "lr": 0.0, "tag": "stale_A"}])
    sub_b = _MockSubOpt([{"params": [], "lr": 0.0, "tag": "stale_B"}])

    parent_initial_pg = sub_a.param_groups + sub_b.param_groups

    class _MockChained:
        chained_optimizers = [sub_a, sub_b]
        param_groups = parent_initial_pg

    mock = _MockChained()

    new_pg_a = [{"params": [], "lr": 1e-3, "tag": "fresh_A"}]
    new_pg_b = [{"params": [], "lr": 2e-3, "tag": "fresh_B"}]
    state_dict = [
        {"param_groups": new_pg_a},
        {"param_groups": new_pg_b},
    ]

    try:
        method(mock, state_dict)
    except Exception as e:
        _emit("FAIL", f"method_call: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    expected_fresh = new_pg_a + new_pg_b
    parent_tags = [pg.get("tag") for pg in mock.param_groups]
    fresh_tags = [pg.get("tag") for pg in expected_fresh]
    stale_tags = [pg.get("tag") for pg in parent_initial_pg]

    msg_common = (
        f"chained_optimizers=2, parent.param_groups tags after load_state_dict={parent_tags}, "
        f"fresh_tags={fresh_tags}, stale_tags={stale_tags}"
    )

    if parent_tags == fresh_tags:
        _emit("CLEAN", "ChainedOptimizer.load_state_dict rebuilds self.param_groups: " + msg_common)
    elif parent_tags == stale_tags:
        _emit(
            "BUG DETECTED",
            "chained_optimizer_param_groups_resync_invariant: " + msg_common
            + " — parent.param_groups remained stale after load_state_dict; downstream LR scheduler sees pre-resume groups",
        )
    else:
        _emit("FAIL", f"unexpected_param_groups_state: parent_tags={parent_tags}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        _emit("FAIL", f"toplevel: {type(e).__name__}: {e}")
        raise
