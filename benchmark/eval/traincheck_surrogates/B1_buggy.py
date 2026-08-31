"""B1 surrogate (buggy): replica params silently diverged across DP ranks.
Mirrors Megatron embedding-init regression where one rank kept the random
init, another did not.
"""
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    torch.manual_seed(0)
    rank0 = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    rank1 = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    # buggy: rank1 NOT mirrored — diverged init
    with torch.no_grad():
        rank1[0].weight.add_(torch.randn_like(rank1[0].weight) * 0.5)

    opt0 = optim.AdamW(rank0.parameters(), lr=1e-3)
    opt1 = optim.AdamW(rank1.parameters(), lr=1e-3)

    for _ in range(8):
        x = torch.randn(2, 8)
        opt0.zero_grad(); opt1.zero_grad()
        out0 = rank0(x)
        out1 = rank1(x)
        loss0 = out0.pow(2).sum()
        loss1 = out1.pow(2).sum()
        loss0.backward(); loss1.backward()
        opt0.step(); opt1.step()


if __name__ == "__main__":
    main()
