"""CW1 surrogate (fixed): step counter uses int64, no overflow within horizon."""
import torch


class Int64Counter:
    def __init__(self):
        self.value = torch.tensor(0, dtype=torch.int64)
    def increment(self):
        self.value = self.value + 1  # int64, no overflow within reasonable horizon
    def get(self):
        return self.value.item()


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    torch.manual_seed(0)
    counter = Int64Counter()
    history = []
    overflows = 0
    last = -1
    for step in range(200):
        counter.increment()
        v = counter.get()
        history.append(v)
        if v < last:
            overflows += 1
        last = v
    print(f"[CW1_fixed] counter type: int64")
    print(f"[CW1_fixed] step 100 → counter={history[99]}")
    print(f"[CW1_fixed] step 128 → counter={history[127]}")
    print(f"[CW1_fixed] step 199 → counter={history[198]}")
    print(f"[CW1_fixed] total overflow events: {overflows}")
    return overflows


if __name__ == "__main__":
    main()
