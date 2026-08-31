"""M-024 surrogate (buggy): apply_input_jitter uses torch.distributions.Uniform
which silently promotes bf16 → fp32 → router input dtype mismatch.
Mirrors Megatron `20b395424d` regression.
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
        # buggy: torch.distributions.Uniform.sample → fp32, then promote x_bf
        dist = torch.distributions.Uniform(
            torch.tensor(-0.01), torch.tensor(0.01))
        jitter = dist.sample(x_bf.shape)
        promoted = x_bf + jitter  # promoted to fp32
        return self.linear(promoted.to(torch.bfloat16))


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
