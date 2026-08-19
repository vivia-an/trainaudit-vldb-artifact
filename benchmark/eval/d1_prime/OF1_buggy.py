"""OF1 surrogate (buggy): CPU-offload optimizer state restored at wrong dtype.

Blueprint: D-029 / DeepSpeed ZeRO + CPU offload family. The grad-norm tensor is
offloaded to CPU for memory savings, then restored to GPU before the norm computation.
The buggy code restores it as fp16 (half) instead of preserving the original fp32,
introducing a sub-percent drift in the clip-grad threshold and hence in the optimizer
update magnitude.

In our CPU-only surrogate, we model "offload" by serializing tensor through a
half-precision round-trip in the buggy path; fixed path preserves dtype.
"""
import torch
import torch.nn as nn


def fake_offload_then_restore(t, restore_dtype):
    """Simulate offload to CPU then bring-back, with a configurable restore dtype."""
    cpu_copy = t.detach().clone()  # "offload"
    # Buggy path passes torch.float16; fixed path passes original dtype.
    return cpu_copy.to(restore_dtype).to(t.dtype)  # round-trip


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

        # collect grad norm
        gnorm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
        # buggy: scaled_norm_tensor goes through fp16 round-trip during offload restore
        gnorm_after_offload = fake_offload_then_restore(gnorm, restore_dtype=torch.float16)
        grad_norms.append(gnorm_after_offload.item())

        # clip using the (corrupted) grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gnorm_after_offload.item() * 0.99)
        opt.step()

    final_param = next(model.parameters()).norm().item()
    avg_gnorm = sum(grad_norms) / len(grad_norms)
    print(f"[OF1_buggy] avg grad_norm seen by clip = {avg_gnorm:.6f}")
    print(f"[OF1_buggy] final param norm = {final_param:.6f}")
    return avg_gnorm, final_param


if __name__ == "__main__":
    main()
