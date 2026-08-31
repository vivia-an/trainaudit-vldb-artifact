"""O-NEW-1 surrogate (buggy): RMSNorm output silently scaled by 0.33,
mirrors OLMo `67c9e315` regression where the gain was applied twice / dropped.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class BrokenRMSNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x) * 0.33


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), BrokenRMSNorm(16),
                          nn.GELU(), nn.Linear(16, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
