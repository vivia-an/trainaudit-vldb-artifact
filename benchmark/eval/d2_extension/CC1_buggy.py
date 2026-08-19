"""CC1 surrogate (buggy): config says zero_stage=2 but optimizer state isn't partitioned.

Triggers P10 Config-Implied Coupling: config.zero_stage > 0 implies
optimizer.has_partitioned_state. Buggy keeps full state on every rank.
"""
import torch
import torch.nn as nn


class FakeOptimWrapper:
    def __init__(self, optimizer, zero_stage):
        self.optimizer = optimizer
        self.zero_stage = zero_stage
        # P10 invariant: zero_stage > 0 ⟹ partitioned_state == True
        self.has_partitioned_state = False  # buggy: never partitions despite zero_stage=2
    def step(self): self.optimizer.step()
    def zero_grad(self): self.optimizer.zero_grad()


def main():
    torch.manual_seed(0)
    config = {"zero_stage": 2, "tp_size": 1}
    model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 32))
    inner_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt = FakeOptimWrapper(inner_opt, zero_stage=config["zero_stage"])

    print(f"[CC1_buggy] config.zero_stage={config['zero_stage']}  "
          f"optimizer.has_partitioned_state={opt.has_partitioned_state}  "
          f"⟹ P10 violation: zero>{0} but state not partitioned")

    for step in range(20):
        x = torch.randn(4, 64)
        loss = model(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[CC1_buggy] final loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
