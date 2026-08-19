"""B12 surrogate (buggy): AdamW state lost 'initial_lr', mirrors OLMo-core
optimizer-resume regression at 6e330ba2 — the scheduler resume path silently
falls back to current lr instead of original initial_lr.
"""
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    for g in opt.param_groups:
        g.pop("initial_lr", None)
    try:
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, last_epoch=10)
    except KeyError:
        sched = None

    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()
        if sched is not None:
            sched.step()


if __name__ == "__main__":
    main()
