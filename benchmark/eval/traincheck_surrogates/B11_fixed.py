"""B11 surrogate (fixed): grad clipping correctly bounds the gradient norm.

Reference run for TrainCheck invariant inference. Trains a tiny MLP for 8 steps
with healthy `torch.nn.utils.clip_grad_norm_`. After clipping, the post-clip
grad norm should always be <= max_norm.
"""
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    max_norm = 0.1

    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        loss = (model(x).pow(2).sum() * 1e6)  # huge loss → grads explode
        loss.backward()
        # Healthy clip — grad_norm after returning is <= max_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        opt.step()


if __name__ == "__main__":
    main()
