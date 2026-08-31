"""
OC-NEW-2: Adam step counter not incremented — bias correction stuck.

Bug: `step.add_(step_factor)` was commented out in adamw_step (used by SkipStepAdamW),
     so the step counter stays at initial value. Adam's bias correction terms never update.
Fix: Uncomment `step.add_(step_factor)`.

Detection: Build SkipStepAdamW, run several steps, check if state['step'] increments.
"""
import os, sys, torch, torch.nn as nn

OLMO_CORE_DIR = os.environ.get("OLMO_CORE_DIR", "")
if OLMO_CORE_DIR:
    sys.path.insert(0, os.path.join(OLMO_CORE_DIR, "src"))

from olmo_core.optim.adamw import SkipStepAdamWConfig


def main():
    torch.manual_seed(42)
    model = nn.Linear(64, 64)

    optimizer_config = SkipStepAdamWConfig(lr=1e-3, betas=(0.9, 0.999))
    optimizer = optimizer_config.build(model)

    # Run 5 training steps
    for step in range(5):
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Check step counter in optimizer state
    step_values = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                if 'step' in state:
                    sv = state['step']
                    step_values.append(sv.item() if hasattr(sv, 'item') else float(sv))

    print(f"\n{'='*60}")
    print(f"[OC-NEW-2] After 5 training steps (SkipStepAdamW):")
    if step_values:
        print(f"  Step counter values: {step_values[:3]}...")
        avg_step = sum(step_values) / len(step_values)
        if avg_step < 1.0:
            print(f"[OC-NEW-2] BUG DETECTED: step counter stuck at {avg_step:.1f} (should be ~5.0)")
            print(f"  Adam bias correction permanently wrong — effective LR biased")
        else:
            print(f"[OC-NEW-2] CLEAN: step counter = {avg_step:.1f} (correctly incremented)")
    else:
        print(f"[OC-NEW-2] WARNING: no step state found in optimizer")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
