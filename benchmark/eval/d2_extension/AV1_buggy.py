"""AV1 surrogate (buggy): fused layernorm impl differs from unfused by ~1e-3.

Triggers P12 Algorithm Variant / Formula Equivalence: fused and unfused
implementations of the same operator should produce numerically equivalent
output (within fp32 tolerance ~1e-6). Buggy fused path drops the eps term.
"""
import torch
import torch.nn as nn


class BuggyFusedLayerNorm(nn.Module):
    """Buggy fused impl: drops eps, uses 1/std instead of 1/sqrt(var+eps)."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        # buggy: eps dropped (or set to 0)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)  # no +eps
        return self.weight * (x - mean) / std + self.bias


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
    torch.manual_seed(0)
    fused = BuggyFusedLayerNorm(64)
    ref = ReferenceLayerNorm(64)
    # Sync params
    with torch.no_grad():
        ref.weight.copy_(fused.weight); ref.bias.copy_(fused.bias)

    diffs = []
    for step in range(20):
        x = torch.randn(4, 64)
        out_fused = fused(x)
        out_ref = ref(x)
        rel_diff = (out_fused - out_ref).abs().max() / out_ref.abs().max()
        diffs.append(rel_diff.item())

    print(f"[AV1_buggy] avg fused-vs-ref rel_diff = {sum(diffs)/len(diffs):.6e}")
    print(f"[AV1_buggy] max rel_diff = {max(diffs):.6e}  (P12 threshold: 1e-5)")
    return diffs


if __name__ == "__main__":
    main()
