"""CW1 surrogate (buggy): step counter uses int8, overflows at step 128.

Triggers P15 Counter Width Adequacy: counter width must be sufficient for the
training-step horizon. Buggy uses int8 (max 127); after step 128 it wraps to -128.
Surrogate uses int8 to make overflow visible in 200 steps.
"""
import torch


class Int8Counter:
    """Simulates an int32-overflow situation by using int8 for the counter."""
    def __init__(self):
        self.value = torch.tensor(0, dtype=torch.int8)
    def increment(self):
        # buggy: blindly += 1, will overflow silently
        self.value = (self.value + 1).to(torch.int8)
    def get(self):
        return self.value.item()


def main():
    torch.manual_seed(0)
    counter = Int8Counter()
    history = []
    overflows = 0
    last = -1
    for step in range(200):
        counter.increment()
        v = counter.get()
        history.append(v)
        # P15 check: counter must monotonically increase (overflow breaks this)
        if v < last:
            overflows += 1
        last = v
    print(f"[CW1_buggy] counter type: int8")
    print(f"[CW1_buggy] step 100 → counter={history[99]}")
    print(f"[CW1_buggy] step 128 → counter={history[127]}")
    print(f"[CW1_buggy] step 130 → counter={history[129]}  (P15 violation: monotonic broken)")
    print(f"[CW1_buggy] step 199 → counter={history[198]}")
    print(f"[CW1_buggy] total overflow events: {overflows}")
    return overflows


if __name__ == "__main__":
    main()
