"""B12 surrogate (fixed): AdamW state preserves 'initial_lr' so resume works.

Reference run for TrainCheck: optimizer ships with `initial_lr` populated in
each param group, so CosineAnnealingLR resume succeeds.
"""
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, last_epoch=10)

    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()
        sched.step()


if __name__ == "__main__":
    main()
