"""B13 surrogate (buggy): OLMo residual block clobbered the residual variable
so output is closer to normalized input than to original input.
Mirrors `562c0fe0` regression.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class BrokenResidualBlock(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        # buggy: residual added to NORMED input, not original
        normed = self.norm(x)
        return normed + self.proj(normed)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), BrokenResidualBlock(8),
                          BrokenResidualBlock(8), nn.Linear(8, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
