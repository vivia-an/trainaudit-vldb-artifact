"""M-NEW-5 surrogate (buggy): Router missing calculate_per_token_loss
attribute. Mirrors Megatron `87d9d2506` regression where the attribute
was renamed and the loss tracker silently fell back to grouped aggregation.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class TopKRouter(nn.Module):
    def __init__(self, hidden=8, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts)
        # buggy: attribute removed

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), TopKRouter(8, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
