"""OC-NEW-2 surrogate (buggy): a frozen AdamW.step() forgets to increment
state['step']. Mirrors OLMo-core `2b6cf996` regression where a custom optimizer
override skipped state bookkeeping.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class FrozenStepAdamW(optim.AdamW):
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0)
                # buggy: no state['step'] += 1
                p.data.add_(p.grad, alpha=-group["lr"])
        return None


def main():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
    opt = FrozenStepAdamW(model.parameters(), lr=1e-3)
    for p in model.parameters():
        opt.state[p]["step"] = torch.tensor(0.0)

    for step in range(8):
        opt.zero_grad()
        x = torch.randn(2, 8)
        model(x).pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
