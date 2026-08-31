"""M-020 surrogate (buggy): pipeline-parallel layer count not divisible by
pp_size, integer division silently drops layers (24 // 5 → 4 per stage,
total 20 instead of 24). Mirrors Megatron `99f999a466` regression.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class TransformerBlock(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        return self.norm(x + self.linear(x))


class PipelineModel(nn.Module):
    def __init__(self, hidden=8, num_layers=24, pp_size=5):
        super().__init__()
        # buggy: integer division drops 4 layers
        per_stage = num_layers // pp_size  # 24/5 = 4
        self.blocks = nn.ModuleList([TransformerBlock(hidden)
                                       for _ in range(per_stage)])
        self.head = nn.Linear(hidden, 4)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), PipelineModel(8, 24, 5))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
