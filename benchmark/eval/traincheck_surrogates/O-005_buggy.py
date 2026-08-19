"""O-005 surrogate (buggy): preserve_rng_state=False with Dropout, mirrors
OLMo `204ad53c` regression — recompute draws different masks → silent gradient
divergence vs forward.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import checkpoint as cp


class BlockWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.dropout(self.linear(x))


class TopModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, 8)
        self.blk = BlockWithDropout()
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        h = self.embed(x)
        h = cp.checkpoint(self.blk, h,
                          use_reentrant=False,
                          preserve_rng_state=False)  # buggy
        return self.head(h)


def main():
    torch.manual_seed(0)
    model = TopModel()
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(4):
        opt.zero_grad()
        x = torch.randn(2, 8, requires_grad=True)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
