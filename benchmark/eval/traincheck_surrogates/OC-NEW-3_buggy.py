"""OC-NEW-3 surrogate (buggy): _sqrt_decay direction inverted to
sqrt(progress) — slow at start, fast at end. Mirrors OLMo-core `f34e7ddc`
regression.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim


def sqrt_decay(progress, initial_lr=1e-3, decay_min_lr=0.0):
    # buggy: sqrt(progress) — slope inverted
    factor = math.sqrt(max(0.0, progress))
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
