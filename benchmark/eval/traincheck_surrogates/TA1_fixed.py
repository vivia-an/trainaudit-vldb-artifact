"""TA1 surrogate (fixed): cached_norm refreshed after each param update."""
import torch
import torch.nn as nn


class FreshCachedNorm:
    def __init__(self, param):
        self.param = param
        self._cached = None
        self._param_version = -1
    def refresh(self):
        # fixed: recompute from current param data
        self._cached = self.param.data.norm()
        self._param_version = self.param._version
    def get(self):
        if self._cached is None or self._param_version != self.param._version:
            self.refresh()
        return self._cached


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    torch.manual_seed(0)
    layer = nn.Linear(32, 64)
    cache = FreshCachedNorm(layer.weight)
    initial_cached = cache.get().item()

    opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    for step in range(20):
        x = torch.randn(4, 32)
        loss = layer(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        cache.refresh()  # fixed: explicit refresh after mutation

    final_cached = cache.get().item()
    final_actual = layer.weight.data.norm().item()
    print(f"[TA1_fixed] initial cached_norm = {initial_cached:.6f}")
    print(f"[TA1_fixed] final cached_norm   = {final_cached:.6f}")
    print(f"[TA1_fixed] actual final norm   = {final_actual:.6f}")
    print(f"[TA1_fixed] stale-vs-actual diff = {abs(final_cached - final_actual):.6f}")


if __name__ == "__main__":
    main()
