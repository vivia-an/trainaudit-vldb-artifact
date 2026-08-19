"""O-NEW-1 surrogate (fixed): LayerNorm output preserves unit RMS scale.

Reference run for TrainCheck.
"""
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16),
                          nn.GELU(), nn.Linear(16, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
