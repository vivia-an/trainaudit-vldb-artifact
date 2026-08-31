"""CF1 surrogate (fixed): MoE aux-loss tracker guarded by torch.is_grad_enabled().

Recompute pass under activation checkpointing runs without grad-enabled, so the guard
suppresses the second tracker call. calls_per_step == 1 invariant holds.
"""
import torch
import torch.nn as nn


class AuxLossTracker:
    def __init__(self):
        self.calls_per_step = 0
        self.total = 0.0

    def save_to_aux_losses_tracker(self, value):
        self.calls_per_step += 1
        self.total += float(value)

    def end_step(self):
        c = self.calls_per_step
        self.calls_per_step = 0
        return c


class MoELayer(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.gate = nn.Linear(8, 4)
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        self.tracker = tracker

    def forward(self, x):
        scores = self.gate(x).softmax(dim=-1)
        aux = (scores * scores.log().clamp(min=-10)).sum()
        # fixed: only accumulate when grad is enabled (skip recompute pass)
        if torch.is_grad_enabled():
            self.tracker.save_to_aux_losses_tracker(aux.item())
        weighted = sum(s.unsqueeze(-1) * e(x)
                       for s, e in zip(scores.unbind(-1), self.experts))
        return weighted


def main():
    torch.manual_seed(0)
    tracker = AuxLossTracker()
    moe = MoELayer(tracker)
    model = moe

    final_aux = 0.0
    for step in range(16):
        x = torch.randn(2, 8, requires_grad=True)
        out_first = moe(x)
        with torch.no_grad():
            _ = moe(x)  # recompute — guard prevents double accumulation
        loss = out_first.pow(2).sum()
        loss.backward()
        calls = tracker.end_step()
        if step == 0:
            print(f"[CF1_fixed] step 0 calls_per_step={calls} (expected 1)")
        final_aux = tracker.total

    print(f"[CF1_fixed] final aux_loss accumulator = {final_aux:.4f}")
    return final_aux


if __name__ == "__main__":
    main()
