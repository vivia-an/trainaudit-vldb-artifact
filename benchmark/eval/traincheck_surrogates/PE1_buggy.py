"""PE1 surrogate (buggy): RoPE position not reset at packed-document boundary.

Triggers P11 Position Encoding & Doc Boundary Integrity: positions[i] should
restart at 0 at each cu_doc_lens[k] boundary. Buggy: continues monotonic across docs.
"""
import torch
import torch.nn as nn


def make_positions(cu_doc_lens, seq_len, mode):
    positions = torch.zeros(seq_len, dtype=torch.long)
    if mode == "buggy":
        positions = torch.arange(seq_len)  # never resets
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
    cu_doc_lens = [16, 40, 64]  # 3 docs of length 16, 24, 24

    pos = make_positions(cu_doc_lens, seq_len, mode="buggy")
    # Check P11 invariant: pos[cu_doc_lens[k]] should == 0 for k > 0
    boundary_violations = 0
    for k, end in enumerate(cu_doc_lens[:-1]):
        if pos[end].item() != 0:
            boundary_violations += 1
    print(f"[PE1_buggy] cu_doc_lens={cu_doc_lens}")
    print(f"[PE1_buggy] positions sample (first 20): {pos[:20].tolist()}")
    print(f"[PE1_buggy] boundary violations: {boundary_violations}/{len(cu_doc_lens)-1}")
    return boundary_violations


if __name__ == "__main__":
    main()
