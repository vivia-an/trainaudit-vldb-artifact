"""ID1 surrogate (buggy): linear weight initialized at std=0.5 vs declared std=0.02.

Triggers P9 Init Distribution Consistency: param.std() should match declared init
distribution within [×0.5, ×1.5] band. Buggy std is 25× the declared, far outside
band. Fixed uses normal_(mean=0, std=0.02) per declared spec.
"""
import torch
import torch.nn as nn


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    torch.manual_seed(0)
    declared_std = 0.02
    layer = nn.Linear(64, 256)
    # buggy: re-initialize with wrong std
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.5)  # 25x declared

    actual = layer.weight.std().item()
    print(f"[ID1_buggy] declared_std={declared_std}  actual_std={actual:.4f} (target [{declared_std*0.5}, {declared_std*1.5}])")

    # Run a few training steps to make trace non-trivial
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    for step in range(20):
        x = torch.randn(8, 64)
        loss = layer(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[ID1_buggy] final loss={loss.item():.4f}")
    return actual


if __name__ == "__main__":
    main()
