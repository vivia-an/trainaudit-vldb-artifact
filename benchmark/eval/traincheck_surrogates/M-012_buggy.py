"""M-012 surrogate (buggy): TopK Router silently demoted `expert_bias` to bf16,
so router add runs in low precision while linear is fp32. Mirrors Megatron
`a58768725f` regression.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class TopKRouter(nn.Module):
    def __init__(self, hidden=8, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts)
        # buggy: bias silently bf16
        self.expert_bias = nn.Parameter(torch.zeros(n_experts, dtype=torch.bfloat16))

    def forward(self, x):
        return self.linear(x) + self.expert_bias.to(x.dtype)


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
