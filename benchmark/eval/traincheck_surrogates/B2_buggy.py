"""B2 surrogate (buggy): TP frozen-weight LinearWithFrozenWeight skipped the
input grad all-reduce, so upstream replica params end up with diverged grads
across DP ranks. Mirrors Megatron `3c637fc0d` regression.
"""
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    torch.manual_seed(0)
    rank0 = nn.Sequential(nn.Embedding(100, 8), nn.Linear(8, 4))
    rank1 = nn.Sequential(nn.Embedding(100, 8), nn.Linear(8, 4))
    rank1.load_state_dict(rank0.state_dict())

    opt0 = optim.AdamW(rank0.parameters(), lr=1e-3)
    opt1 = optim.AdamW(rank1.parameters(), lr=1e-3)

    for step in range(8):
        # buggy: ranks see different inputs (no input all-reduce)
        x0 = torch.randint(0, 100, (2, 4))
        x1 = torch.randint(0, 100, (2, 4)) + 50
        opt0.zero_grad(); opt1.zero_grad()
        rank0(x0).pow(2).sum().backward()
        rank1(x1).pow(2).sum().backward()
        # buggy: NO grad all-reduce, ranks diverge
        opt0.step(); opt1.step()


if __name__ == "__main__":
    main()
