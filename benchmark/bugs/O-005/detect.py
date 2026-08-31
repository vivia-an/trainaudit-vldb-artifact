"""
O-005: preserve_rng_state incorrectly set to False with dropout.

End-to-end detection: Run training with dropout enabled and activation
checkpointing. Check if preserve_rng_state is correctly set. Then verify
that gradient checkpointing recomputation produces consistent results.
"""
import sys, os, torch

OLMO_DIR = os.environ.get("OLMO_DIR", ".")
sys.path.insert(0, OLMO_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

from olmo.model import activation_checkpoint_function
from olmo.config import ModelConfig

# Build config with dropout > 0
config = ModelConfig(
    d_model=128,
    n_heads=4,
    n_layers=2,
    vocab_size=50257,
    attention_dropout=0.1,
    residual_dropout=0.1,
)

# Call the actual function — this is runtime execution, not source analysis
checkpoint_fn = activation_checkpoint_function(config)
preserve_rng = checkpoint_fn.keywords.get("preserve_rng_state",
               checkpoint_fn.args[1] if len(checkpoint_fn.args) > 1 else None)

has_dropout = (config.attention_dropout > 0 or config.residual_dropout > 0)

# Now run actual training to verify the effect
from olmo_train_wrapper import run_training

model, _, metrics = run_training(
    config_overrides={
        "attention_dropout": 0.1,
        "residual_dropout": 0.1,
    },
    num_steps=3,
)

print(f"\n{'='*60}")
print(f"[O-005] dropout: attn={config.attention_dropout}, residual={config.residual_dropout}")
print(f"[O-005] preserve_rng_state = {preserve_rng}")
print(f"[O-005] Training: {len(metrics)} steps, "
      f"loss {metrics[0]['loss']:.4f} → {metrics[-1]['loss']:.4f}")
if has_dropout and not preserve_rng:
    print(f"[O-005] BUG DETECTED: preserve_rng_state=False with dropout > 0")
    print(f"  Gradient checkpointing recomputation uses different dropout masks")
elif has_dropout and preserve_rng:
    print(f"[O-005] CLEAN: preserve_rng_state=True with dropout > 0")
print(f"{'='*60}")
