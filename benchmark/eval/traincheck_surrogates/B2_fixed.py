"""B2 surrogate (fixed): TP-replica gradients all-reduced across rank, so
embed.weight grad checksum is identical across simulated DP ranks. Reference
run for TrainCheck.

We don't have real distributed; simulate two ranks via two model copies with
gradients explicitly synced post-backward.
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

    for _ in range(8):
        x = torch.randint(0, 100, (2, 4))
        opt0.zero_grad(); opt1.zero_grad()
        rank0(x).pow(2).sum().backward()
        rank1(x).pow(2).sum().backward()
        # fixed: explicit grad all-reduce (avg) across ranks
        with torch.no_grad():
            for p0, p1 in zip(rank0.parameters(), rank1.parameters()):
                if p0.grad is not None and p1.grad is not None:
                    avg = (p0.grad + p1.grad) / 2
                    p0.grad.copy_(avg)
                    p1.grad.copy_(avg)
        opt0.step(); opt1.step()


if __name__ == "__main__":
    main()
