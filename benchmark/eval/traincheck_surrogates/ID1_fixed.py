"""ID1 surrogate (fixed): linear weight initialized at declared std=0.02."""
import torch
import torch.nn as nn


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    torch.manual_seed(0)
    declared_std = 0.02
    layer = nn.Linear(64, 256)
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=declared_std)  # matches declared

    actual = layer.weight.std().item()
    print(f"[ID1_fixed] declared_std={declared_std}  actual_std={actual:.4f}")

    opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    for step in range(20):
        x = torch.randn(8, 64)
        loss = layer(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[ID1_fixed] final loss={loss.item():.4f}")
    return actual


if __name__ == "__main__":
    main()
