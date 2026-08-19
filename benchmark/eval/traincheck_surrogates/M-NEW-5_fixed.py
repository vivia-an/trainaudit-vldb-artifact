"""M-NEW-5 surrogate (fixed): TopK Router exposes
calculate_per_token_loss flag so MoE loss tracker knows how to aggregate.
Reference run for TrainCheck.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class TopKRouter(nn.Module):
    def __init__(self, hidden=8, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts)
        self.calculate_per_token_loss = True  # fixed: attribute present

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
