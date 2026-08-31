"""B3 surrogate (buggy): bf16 training declared, but comm tensor silently cast
to fp16. Mirrors DeepSpeed `005afe12` precision-config regression.
"""
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4)).to(torch.bfloat16)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        out = model(x)
        # buggy: comm tensor cast to fp16 — narrower exponent, different range
        comm_tensor = out.detach().clone().to(torch.float16)
        comm_tensor.mul_(1.0)  # placeholder for all_reduce
        # write back to bf16 (mimics how engine consumes the all_reduce result)
        out.pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
