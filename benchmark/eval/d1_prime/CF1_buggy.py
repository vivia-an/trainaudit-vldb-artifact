"""CF1 surrogate (buggy): MoE aux-loss double-counted under activation checkpointing.

Blueprint: M-010. The aux_loss tracker function is called twice per step because the
recomputation pass under activation checkpointing re-runs the MoE forward without a
grad-enabled guard, so save_to_aux_losses_tracker() executes once during the original
forward AND once during recompute.

Effect: aux_loss accumulator value at end-of-step is 2x the intended scale, slowly
biasing the auxiliary loss term. The total loss only differs by sub-percent because
aux loss is weighted by ~0.01 in the total — exactly the kind of bias Naïve monitoring
won't see in any single step.
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
        # buggy: no torch.is_grad_enabled() guard; recompute under checkpointing
        # re-enters this path and accumulates aux_loss twice
        aux = (scores * scores.log().clamp(min=-10)).sum()
        self.tracker.save_to_aux_losses_tracker(aux.item())
        weighted = sum(s.unsqueeze(-1) * e(x)
                       for s, e in zip(scores.unbind(-1), self.experts))
        return weighted


def main():
    torch.manual_seed(0)
    tracker = AuxLossTracker()
    moe = MoELayer(tracker)

    final_aux = 0.0
    for step in range(16):
        x = torch.randn(2, 8, requires_grad=True)
        # buggy: simulate activation checkpointing — forward runs once, then a
        # recompute pass calls forward again. No grad-enabled guard, so the
        # tracker accumulates twice every step.
        out_first = moe(x)
        with torch.no_grad():
            _ = moe(x)  # recompute — also accumulates
        loss = out_first.pow(2).sum()
        loss.backward()
        calls = tracker.end_step()
        if step == 0:
            print(f"[CF1_buggy] step 0 calls_per_step={calls} (expected 1)")
        final_aux = tracker.total

    print(f"[CF1_buggy] final aux_loss accumulator = {final_aux:.4f}")
    return final_aux


if __name__ == "__main__":
    main()
