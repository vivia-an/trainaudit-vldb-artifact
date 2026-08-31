"""SC1 surrogate (fixed): every TP rank writes its own checkpoint shard."""
import os
import shutil
import tempfile


def fake_save(model_state, save_dir, tp_size, mode):
    os.makedirs(save_dir, exist_ok=True)
    if mode == "buggy":
        path = os.path.join(save_dir, "mp_rank_00_model_states.pt")
        with open(path, "w") as f:
            f.write("rank-0 state")
    else:
        for r in range(tp_size):
            path = os.path.join(save_dir, f"mp_rank_{r:02d}_model_states.pt")
            with open(path, "w") as f:
                f.write(f"rank-{r} state")


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    tp_size = 2
    save_dir = tempfile.mkdtemp(prefix="sc1_fixed_")

    fake_save({"weight": "stub"}, save_dir, tp_size, mode="fixed")
    saved_files = sorted(os.listdir(save_dir))
    expected_files = sorted([f"mp_rank_{r:02d}_model_states.pt" for r in range(tp_size)])
    missing = set(expected_files) - set(saved_files)
    print(f"[SC1_fixed] tp_size={tp_size}")
    print(f"[SC1_fixed] saved   : {saved_files}")
    print(f"[SC1_fixed] expected: {expected_files}")
    print(f"[SC1_fixed] missing : {sorted(missing)}")

    shutil.rmtree(save_dir)
    return len(missing)


if __name__ == "__main__":
    main()
