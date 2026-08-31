"""OF1 surrogate (fixed): offload restore preserves original dtype.
"""
import torch
import torch.nn as nn


def fake_offload_then_restore(t, restore_dtype):
    cpu_copy = t.detach().clone()
    return cpu_copy.to(restore_dtype).to(t.dtype)


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    grad_norms = []
    for step in range(20):
        opt.zero_grad()
        x = torch.randn(4, 16)
        y = model(x).pow(2).sum()
        y.backward()

        gnorm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
        # fixed: preserve original dtype during offload round-trip
        gnorm_after_offload = fake_offload_then_restore(gnorm, restore_dtype=gnorm.dtype)
        grad_norms.append(gnorm_after_offload.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gnorm_after_offload.item() * 0.99)
        opt.step()

    final_param = next(model.parameters()).norm().item()
    avg_gnorm = sum(grad_norms) / len(grad_norms)
    print(f"[OF1_fixed] avg grad_norm seen by clip = {avg_gnorm:.6f}")
    print(f"[OF1_fixed] final param norm = {final_param:.6f}")
    return avg_gnorm, final_param


if __name__ == "__main__":
    main()
