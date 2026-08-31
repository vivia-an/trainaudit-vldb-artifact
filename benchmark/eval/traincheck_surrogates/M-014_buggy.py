"""M-014 surrogate (buggy): Megatron MoE router applies softmax AFTER topk
selection — softmax over a 1-element vector is always [1.0], destroying expert
weighting. Mirrors `5153efea0` regression.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class DegenerateRouter(nn.Module):
    def __init__(self, hidden, n_experts=4):
        super().__init__()
        self.linear = nn.Linear(hidden, n_experts)

    def forward(self, x):
        scores = self.linear(x)
        topk_scores, _ = scores.topk(1, dim=-1)
        return torch.softmax(topk_scores, dim=-1)  # softmax over 1-d → all 1.0


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), DegenerateRouter(16))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
