"""SC1 surrogate (buggy): TP=2 checkpoint.save only writes rank-0 file.

Triggers P14 Sharded State Completeness: under TP=N, save must produce a file
covering every TP rank in [0, N). Buggy short-circuits to rank 0 only.
"""
import os
import shutil
import tempfile


def fake_save(model_state, save_dir, tp_size, mode):
    os.makedirs(save_dir, exist_ok=True)
    if mode == "buggy":
        # buggy: only rank 0 saves (early return short-circuit)
        path = os.path.join(save_dir, "mp_rank_00_model_states.pt")
        with open(path, "w") as f:
            f.write("rank-0 state")
    else:
        # fixed: every rank saves
        for r in range(tp_size):
            path = os.path.join(save_dir, f"mp_rank_{r:02d}_model_states.pt")
            with open(path, "w") as f:
                f.write(f"rank-{r} state")


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    tp_size = 2
    save_dir = tempfile.mkdtemp(prefix="sc1_buggy_")

    fake_save({"weight": "stub"}, save_dir, tp_size, mode="buggy")
    saved_files = sorted(os.listdir(save_dir))
    expected_files = sorted([f"mp_rank_{r:02d}_model_states.pt" for r in range(tp_size)])
    missing = set(expected_files) - set(saved_files)
    print(f"[SC1_buggy] tp_size={tp_size}")
    print(f"[SC1_buggy] saved   : {saved_files}")
    print(f"[SC1_buggy] expected: {expected_files}")
    print(f"[SC1_buggy] missing : {sorted(missing)}  (P14 violation: {len(missing)} ranks not saved)")

    shutil.rmtree(save_dir)
    return len(missing)


if __name__ == "__main__":
    main()
