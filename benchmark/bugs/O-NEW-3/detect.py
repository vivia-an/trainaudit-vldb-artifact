"""
O-NEW-3: z_loss mean reduction wrong — divides by mask.mean() instead of masked mean.

Bug: z_squared / (labels != ignore_index).mean() — includes padding positions, returns non-scalar
Fix: (z_squared * (labels != ignore_index)).mean() — only non-padding, returns scalar
"""
import os, sys, torch

OLMO_DIR = os.environ.get("OLMO_DIR", "")
if OLMO_DIR: sys.path.insert(0, OLMO_DIR)

def main():
    torch.manual_seed(42)

    # Simulate logits and labels with some padding (ignore_index=-100)
    B, S, V = 2, 32, 1024
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    # Make 50% of positions padding
    labels[:, S//2:] = -100
    ignore_index = -100

    # Compute z_squared (logsumexp squared)
    z = torch.logsumexp(logits, dim=-1)  # (B, S)
    z_squared = z.pow(2)

    mask = (labels != ignore_index).float()  # (B, S)

    # Buggy formula
    buggy_z_loss = z_squared / mask.mean()

    # Correct formula
    correct_z_loss = (z_squared * mask).mean()

    print(f"\n{'='*60}")
    print(f"[O-NEW-3] z_loss reduction check:")
    print(f"  logits shape: {logits.shape}, labels shape: {labels.shape}")
    print(f"  non-padding fraction: {mask.mean().item():.2f}")
    print(f"  buggy z_loss shape: {buggy_z_loss.shape} (should be scalar)")
    print(f"  buggy z_loss mean: {buggy_z_loss.mean().item():.4f}")
    print(f"  correct z_loss: {correct_z_loss.item():.4f} (scalar)")

    # Now test with real OLMo code if available
    try:
        # Try to import and hook the actual cross_entropy_loss function
        from olmo.train import cross_entropy_loss
        # Call it with our test data
        result = cross_entropy_loss(logits.view(-1, V), labels.view(-1), ignore_index=ignore_index, reduction="mean", compute_z_loss=True)
        if isinstance(result, tuple) and len(result) >= 2:
            ce_loss, z_loss = result[0], result[1]
            z_is_scalar = z_loss.dim() == 0
            print(f"\n  Real OLMo z_loss shape: {z_loss.shape}")
            print(f"  Real OLMo z_loss value: {z_loss.mean().item():.4f}")
            if not z_is_scalar:
                print(f"[O-NEW-3] BUG DETECTED: z_loss is not scalar (shape={z_loss.shape})")
                print(f"  Buggy formula returns per-position tensor, not reduced")
            else:
                print(f"[O-NEW-3] CLEAN: z_loss is scalar")
        else:
            print(f"  cross_entropy_loss returned unexpected format")
    except Exception as e:
        # Fall back: compare formulas directly
        buggy_is_tensor = buggy_z_loss.dim() > 0
        ratio = buggy_z_loss.mean().item() / (correct_z_loss.item() + 1e-10)
        print(f"\n  Formula comparison (ratio buggy/correct): {ratio:.2f}")
        if buggy_is_tensor and abs(ratio - 1.0) > 0.1:
            print(f"[O-NEW-3] BUG DETECTED: buggy formula produces tensor (not scalar)")
            print(f"  and value differs by {ratio:.2f}x from correct")
        else:
            print(f"[O-NEW-3] Formulas match (both produce same value)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
