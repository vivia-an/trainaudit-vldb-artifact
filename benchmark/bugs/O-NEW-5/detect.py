"""
O-NEW-5: Embedding initialization std too small — missing sqrt(d_model) factor.

Bug: Embedding init uses config.init_std directly, should be config.init_std * sqrt(d_model).
     With d_model=256 and init_std=0.02, embeddings get std=0.02 instead of 0.32.
Fix: Multiply init_std by sqrt(d_model).

Detection: Build model, check embedding weight std against expected value.
"""
import os, sys, torch, math

OLMO_DIR = os.environ.get("OLMO_DIR", "")
if OLMO_DIR:
    sys.path.insert(0, OLMO_DIR)

def main():
    from olmo.config import ModelConfig
    try:
        from olmo.model import Olmo
    except ImportError:
        from olmo.model import OLMo as Olmo

    config = ModelConfig(
        d_model=256,
        n_heads=4,
        n_layers=2,
        vocab_size=50304,
        embedding_size=50304,
        init_std=0.02,
    )

    torch.manual_seed(42)
    model = Olmo(config)

    # Check embedding weight statistics
    emb_weight = model.transformer.wte.weight
    actual_std = emb_weight.std().item()
    expected_correct = config.init_std * math.sqrt(config.d_model)  # 0.02 * 16 = 0.32
    expected_buggy = config.init_std  # 0.02

    print(f"\n{'='*60}")
    print(f"[O-NEW-5] Embedding initialization check:")
    print(f"  config.init_std = {config.init_std}")
    print(f"  config.d_model = {config.d_model}")
    print(f"  embedding weight std = {actual_std:.4f}")
    print(f"  expected (correct): {expected_correct:.4f} (init_std * sqrt(d_model))")
    print(f"  expected (buggy):   {expected_buggy:.4f} (init_std only)")

    # Check which one it's closer to
    dist_correct = abs(actual_std - expected_correct)
    dist_buggy = abs(actual_std - expected_buggy)

    if dist_buggy < dist_correct:
        print(f"[O-NEW-5] BUG DETECTED: embedding std={actual_std:.4f} ≈ init_std={expected_buggy}")
        print(f"  Missing sqrt(d_model) factor — embeddings {math.sqrt(config.d_model):.0f}x too small")
    else:
        print(f"[O-NEW-5] CLEAN: embedding std={actual_std:.4f} ≈ init_std*sqrt(d)={expected_correct:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
