"""B8 surrogate (buggy): user declared EP=2 but code spawned 8 experts —
the EP communicator group is sized for 2, so 6 experts silently never get
gradients. Mirrors DeepSpeed MoE config-validation regression.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class MoEBlock(nn.Module):
    def __init__(self, hidden=8, n_experts=8, declared_ep_size=2):
        super().__init__()
        # buggy: actual experts != declared ep size
        self.experts = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(n_experts)])
        self.declared_ep_size = declared_ep_size  # exposed but ignored
        self.n_experts = n_experts

    def forward(self, x):
        # buggy: only first declared_ep_size experts receive gradients
        # (the rest are dead weight in the EP comm topology)
        active = self.experts[:self.declared_ep_size]
        out = sum(expert(x) for expert in active) / len(active)
        return out


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), MoEBlock(8, n_experts=8,
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
