"""PE1 surrogate (fixed): RoPE position resets at each packed-doc boundary."""
import torch


def make_positions(cu_doc_lens, seq_len, mode):
    positions = torch.zeros(seq_len, dtype=torch.long)
    if mode == "buggy":
        positions = torch.arange(seq_len)
    else:
        ptr = 0
        for end in cu_doc_lens:
            length = end - ptr
            positions[ptr:end] = torch.arange(length)
            ptr = end
    return positions


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    torch.manual_seed(0)
    seq_len = 64
    cu_doc_lens = [16, 40, 64]

    pos = make_positions(cu_doc_lens, seq_len, mode="fixed")
    boundary_violations = 0
    for k, end in enumerate(cu_doc_lens[:-1]):
        if pos[end].item() != 0:
            boundary_violations += 1
    print(f"[PE1_fixed] cu_doc_lens={cu_doc_lens}")
    print(f"[PE1_fixed] positions sample (first 20): {pos[:20].tolist()}")
    print(f"[PE1_fixed] boundary violations: {boundary_violations}/{len(cu_doc_lens)-1}")
    return boundary_violations


if __name__ == "__main__":
    main()
