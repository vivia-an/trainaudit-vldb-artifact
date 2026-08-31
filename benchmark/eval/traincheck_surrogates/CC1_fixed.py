"""CC1 surrogate (fixed): zero_stage=2 properly couples to partitioned optimizer state."""
import torch
import torch.nn as nn


class FakeOptimWrapper:
    def __init__(self, optimizer, zero_stage):
        self.optimizer = optimizer
        self.zero_stage = zero_stage
        # fixed: derive has_partitioned_state from zero_stage
        self.has_partitioned_state = (zero_stage > 0)
    def step(self): self.optimizer.step()
    def zero_grad(self): self.optimizer.zero_grad()


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    torch.manual_seed(0)
    config = {"zero_stage": 2, "tp_size": 1}
    model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 32))
    inner_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt = FakeOptimWrapper(inner_opt, zero_stage=config["zero_stage"])

    print(f"[CC1_fixed] config.zero_stage={config['zero_stage']}  "
          f"optimizer.has_partitioned_state={opt.has_partitioned_state}  ⟹ P10 holds")

    for step in range(20):
        x = torch.randn(4, 64)
        loss = model(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[CC1_fixed] final loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
