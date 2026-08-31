"""M-012 surrogate (fixed): TopK Router holds an FP32 `expert_bias` parameter.
Real torch forward exercises Linear + bias add so TrainCheck can observe the
parameter dtype + tensor stats.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class TopKRouter(nn.Module):
    def __init__(self, hidden=8, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts)
        # fixed: bias kept as float32 (rule precondition)
        self.expert_bias = nn.Parameter(torch.zeros(n_experts, dtype=torch.float32))

    def forward(self, x):
        return self.linear(x) + self.expert_bias


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
