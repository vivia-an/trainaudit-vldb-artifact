"""M-020 surrogate (fixed): pipeline-parallel layer count divides cleanly.
Reference run for TrainCheck — model has the declared number of transformer
blocks, every block runs.
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
    def __init__(self, hidden=8, num_layers=24, pp_size=4):
        super().__init__()
        per_stage = num_layers // pp_size  # 24/4 = 6 (clean)
        self.blocks = nn.ModuleList([TransformerBlock(hidden)
                                       for _ in range(per_stage)])
        self.head = nn.Linear(hidden, 4)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), PipelineModel(8, 24, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
