"""B13 surrogate (fixed): OLMo residual block preserves the residual stream
y = x + f(norm(x)), keeping output close to original input. Reference run.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class ResidualBlock(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        # fixed: residual = original x
        return x + self.proj(self.norm(x))


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), ResidualBlock(8),
                          ResidualBlock(8), nn.Linear(8, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
