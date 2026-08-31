"""OC-NEW-3 surrogate (fixed): _sqrt_decay schedules lr as
1 - sqrt(1 - progress) — fast at start, slow at end. Reference run.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim


def sqrt_decay(progress, initial_lr=1e-3, decay_min_lr=0.0):
    # fixed: 1 - sqrt(1 - progress); progress in [0, 1]
    factor = 1.0 - math.sqrt(max(0.0, 1.0 - progress))
    return decay_min_lr + factor * (initial_lr - decay_min_lr)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    opt = optim.SGD(model.parameters(), lr=1e-3)

    for step in range(20):
        progress = step / 20.0
        for g in opt.param_groups:
            g["lr"] = sqrt_decay(progress)
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
