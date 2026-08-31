"""B8 surrogate (fixed): expert count matches declared EP size.
Reference run for TrainCheck.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class MoEBlock(nn.Module):
    def __init__(self, hidden=8, n_experts=2, declared_ep_size=2):
        super().__init__()
        # fixed: actual experts == declared ep size
        assert n_experts == declared_ep_size, (
            f"EP size mismatch: {n_experts} != {declared_ep_size}")
        self.experts = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(n_experts)])
        self.n_experts = n_experts

    def forward(self, x):
        out = sum(expert(x) for expert in self.experts) / self.n_experts
        return out


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), MoEBlock(8, n_experts=2,
                                                   declared_ep_size=2),
                          nn.Linear(8, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
