"""AV1 surrogate (fixed): fused layernorm includes eps, equivalent to unfused."""
import torch
import torch.nn as nn


class FusedLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        # fixed: eps included
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class ReferenceLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    torch.manual_seed(0)
    fused = FusedLayerNorm(64)
    ref = ReferenceLayerNorm(64)
    with torch.no_grad():
        ref.weight.copy_(fused.weight); ref.bias.copy_(fused.bias)

    diffs = []
    for step in range(20):
        x = torch.randn(4, 64)
        rel_diff = (fused(x) - ref(x)).abs().max() / ref(x).abs().max()
        diffs.append(rel_diff.item())

    print(f"[AV1_fixed] avg fused-vs-ref rel_diff = {sum(diffs)/len(diffs):.6e}")
    print(f"[AV1_fixed] max rel_diff = {max(diffs):.6e}  (P12 threshold: 1e-5)")
    return diffs


if __name__ == "__main__":
    main()
