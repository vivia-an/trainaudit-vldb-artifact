"""M-014 surrogate (fixed): MoE router applies softmax over the full expert
dimension. Reference run for TrainCheck.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class CleanRouter(nn.Module):
    def __init__(self, hidden, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), CleanRouter(16))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
