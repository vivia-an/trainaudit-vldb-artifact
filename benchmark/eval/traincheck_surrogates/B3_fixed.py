"""B3 surrogate (fixed): bf16 training keeps comm tensors in bf16 — dtype
matches training_dtype. Reference run for TrainCheck.
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
        # fixed: comm tensor matches training dtype
        comm_tensor = out.detach().clone().to(torch.bfloat16)
        comm_tensor.mul_(1.0)  # placeholder for all_reduce
        out.pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
