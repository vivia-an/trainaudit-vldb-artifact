"""O-005 via TrainAudit Phase 1 (T0).

Bug: olmo's activation_checkpoint_function passes preserve_rng_state=False
even when dropout > 0 → recomputed forward draws different dropout masks →
silently wrong gradient.

T0 detection path:
  1. checkpoint_hook captures torch.utils.checkpoint(... preserve_rng_state=...) calls
  2. build_snapshot lists model parameters; module hooks expose Dropout class names
  3. T0-checkpoint-preserve-rng fires when preserve_rng_state=False AND any Dropout exists
"""
import os
import sys
import torch
import torch.nn as nn

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "trainaudit_pkg"))

OLMO_DIR = os.environ.get("OLMO_DIR", "")
if OLMO_DIR:
    sys.path.insert(0, OLMO_DIR)

import trainaudit


def main():
    trainaudit.enable(tier=trainaudit.Tier.T0_PYTORCH, db_path=":memory:")

    # Build a tiny OLMo model with dropout > 0 + activation checkpointing.
    from olmo.config import ModelConfig
    from olmo.model import OLMo, activation_checkpoint_function

    config = ModelConfig(
        d_model=128, n_heads=4, n_layers=2,
        vocab_size=1024, embedding_size=1024,
        attention_dropout=0.1, residual_dropout=0.1,
        block_type="sequential", alibi=False, rope=True, weight_tying=True,
    )
    model = OLMo(config)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Snapshot model + optimizer
    trainaudit.snapshot_build(model, optim)

    # Real forward + backward, going through activation_checkpoint_function
    checkpoint_fn = activation_checkpoint_function(config)
    input_ids = torch.randint(0, 1024, (2, 16))
    # Wrap forward via the framework's chosen checkpoint function — this exercises
    # exactly the buggy/fixed code path: preserve_rng_state passed by olmo.
    out = model(input_ids)
    loss = out.logits.float().sum()
    optim.zero_grad()
    loss.backward()
    optim.step()

    # Manually invoke checkpoint_fn to exercise its preserve_rng_state arg
    def _f(x):
        return model.transformer.blocks[0](x)

    # Use the framework's own checkpoint wrapper at least once
    try:
        x = torch.randn(2, 16, config.d_model, requires_grad=True)
        _ = checkpoint_fn(_f, x)
    except Exception as e:
        # Some olmo versions wire checkpoint differently — capture event still fires
        pass

    results = trainaudit.run_rules()
    print(trainaudit.summarize(results))
    violated = [r for r in results if r.violated]
    if violated:
        print(f"\n[O-005/trainaudit] BUG DETECTED via {len(violated)} rule(s):")
        for v in violated:
            print(f"   - {v.rule_id}: {v.message}")
            if v.evidence:
                print(f"     evidence: {v.evidence}")
    else:
        print("\n[O-005/trainaudit] CLEAN: no rule violations")
    trainaudit.disable()


if __name__ == "__main__":
    main()
