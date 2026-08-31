"""B11 surrogate (buggy): grad clipping is replaced with a no-op that only
returns the norm without actually rescaling gradients.

Mirrors the DeepSpeed `ds_utils.clip_grad_norm_` regression at commit 005afe12
where the routine returned the L2 norm but never multiplied parameters by the
clip coefficient. After this "clip", grads remain unbounded.
"""
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim


def buggy_clip(parameters, max_norm, norm_type=2.0, **kwargs):
    params = [p for p in parameters if p is not None and p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(
        [torch.linalg.vector_norm(p.grad.detach().float(), norm_type)
         for p in params]), norm_type)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    max_norm = 0.1

    nn_utils.clip_grad_norm_ = buggy_clip  # silent regression

    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        loss = (model(x).pow(2).sum() * 1e6)
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        opt.step()


if __name__ == "__main__":
    main()
