"""B1 surrogate (fixed): replica params identical across simulated DP ranks.
Reference run for TrainCheck.

We don't have a real distributed setup here; the surrogate exercises the
observable invariant: identical weights → identical forward outputs.
"""
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    torch.manual_seed(0)
    rank0 = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    rank1 = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    # fixed: rank1 mirrors rank0
    rank1.load_state_dict(rank0.state_dict())

    opt0 = optim.AdamW(rank0.parameters(), lr=1e-3)
    opt1 = optim.AdamW(rank1.parameters(), lr=1e-3)

    for _ in range(8):
        x = torch.randn(2, 8)
        opt0.zero_grad(); opt1.zero_grad()
        out0 = rank0(x)
        out1 = rank1(x)
        loss0 = out0.pow(2).sum()
        loss1 = out1.pow(2).sum()
        # identical replicas → identical losses
        loss0.backward(); loss1.backward()
        opt0.step(); opt1.step()


if __name__ == "__main__":
    main()
