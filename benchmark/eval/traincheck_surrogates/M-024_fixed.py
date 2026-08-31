"""M-024 surrogate (fixed): apply_input_jitter preserves bf16 dtype after
adding noise. Reference run for TrainCheck.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class JitterRouter(nn.Module):
    def __init__(self, hidden=8, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts).to(torch.bfloat16)

    def forward(self, x):
        x_bf = x.to(torch.bfloat16)
        # fixed: scale-only jitter, no dtype promotion
        jitter = torch.empty_like(x_bf).uniform_(-0.01, 0.01)
        return self.linear(x_bf + jitter)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), JitterRouter(8, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
