"""TA1 surrogate (buggy): cached_norm shares storage with param.data, never refreshes after optim.step.

Triggers P13 Tensor Aliasing & Stale State: when cached_state.data_ptr() ==
underlying_param.data_ptr(), cache must be invalidated after the param mutates.
Buggy: the cached_norm pointer is never refreshed, so reading it after optim.step
gives stale value.
"""
import torch
import torch.nn as nn


class StaleCachedNorm:
    def __init__(self, param):
        self.param = param
        # buggy: store a *view* (shares storage); never explicitly refresh
        self._cached = param.data.norm()
    def get(self):
        return self._cached  # buggy: stale, never recomputed


def main():
    torch.manual_seed(0)
    layer = nn.Linear(32, 64)
    cache = StaleCachedNorm(layer.weight)
    initial_cached = cache.get().item()

    opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    for step in range(20):
        x = torch.randn(4, 32)
        loss = layer(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        # P13 check: cache value should equal current param.norm() after each update
        actual_norm = layer.weight.data.norm().item()
        cached_value = cache.get().item()

    final_cached = cache.get().item()
    final_actual = layer.weight.data.norm().item()
    print(f"[TA1_buggy] initial cached_norm = {initial_cached:.6f}")
    print(f"[TA1_buggy] final cached_norm   = {final_cached:.6f}")
    print(f"[TA1_buggy] actual final norm   = {final_actual:.6f}")
    print(f"[TA1_buggy] stale-vs-actual diff = {abs(final_cached - final_actual):.6f}  (P13: should be ~0)")


if __name__ == "__main__":
    main()
