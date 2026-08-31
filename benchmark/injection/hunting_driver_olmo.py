"""OLMo hunting driver — clean train step under trainaudit.

Builds a small OLMo Sequential including the chosen norm type (rms /
default), runs forward + backward + optim.step, lets every trainaudit
rule observe.

Env knobs:
  HUNT_NORM       rms | default        (default rms)
  HUNT_LAYERS     int                  (default 2)
  HUNT_STEPS      int                  (default 5)
  HUNT_TIER       T0_PYTORCH | T1_FW_METADATA   (default T1_FW_METADATA)
  HUNT_CKPT       0 | 1 — exercise checkpoint hook                (default 0)
  OLMO_DIR        OLMo checkout root, prepended to PYTHONPATH
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "trainaudit_pkg"))

OLMO_DIR = os.environ.get("OLMO_DIR", "")
if OLMO_DIR:
    sys.path.insert(0, OLMO_DIR)

import trainaudit  # noqa: E402

# Try a couple of import paths — OLMo's tree changed across releases.
_RMS = None
_CONFIG = None
for _path in ("olmo.model", "olmo.modeling.olmo"):
    try:
        m = __import__(_path, fromlist=["RMSLayerNorm"])
        if hasattr(m, "RMSLayerNorm"):
            _RMS = m.RMSLayerNorm
            break
    except Exception:
        continue
try:
    from olmo.config import ModelConfig as _CONFIG  # type: ignore
except Exception:
    _CONFIG = None


def _tier(name: str):
    name = (name or "T1_FW_METADATA").upper()
    if hasattr(trainaudit.Tier, name):
        return getattr(trainaudit.Tier, name)
    return trainaudit.Tier.T1_FW_METADATA


def main():
    norm_type = os.environ.get("HUNT_NORM", "rms").lower()
    n_layers = int(os.environ.get("HUNT_LAYERS", "2"))
    n_steps = int(os.environ.get("HUNT_STEPS", "5"))
    use_ckpt = bool(int(os.environ.get("HUNT_CKPT", "0")))
    tier = _tier(os.environ.get("HUNT_TIER", "T1_FW_METADATA"))

    torch.manual_seed(42)
    trainaudit.enable(tier=tier, db_path=":memory:")

    d_model = 256
    if _RMS is not None and _CONFIG is not None and norm_type == "rms":
        cfg = _CONFIG(
            d_model=d_model, n_heads=4, n_layers=n_layers,
            vocab_size=50304, embedding_size=50304,
            layer_norm_type="rms")
        norm_layer = _RMS(cfg, size=d_model)
    else:
        norm_layer = nn.LayerNorm(d_model)

    layers = []
    for _ in range(n_layers):
        layers.append(nn.Linear(d_model, d_model))
        layers.append(norm_layer.__class__(*norm_layer.__init__.__defaults__)
                       if False else norm_layer)
        layers.append(nn.GELU())
    # Avoid sharing state across blocks: rebuild norm copies
    blocks = nn.ModuleList()
    for _ in range(n_layers):
        if _RMS is not None and _CONFIG is not None and norm_type == "rms":
            n = _RMS(cfg, size=d_model)
        else:
            n = nn.LayerNorm(d_model)
        blocks.append(nn.Sequential(nn.Linear(d_model, d_model), n, nn.GELU()))
    model = nn.Sequential(*blocks, nn.Linear(d_model, d_model))

    if use_ckpt:
        # Pretend a ckpt event happened so the checkpoint hook has data.
        ckpt_path = "/tmp/hunt_olmo_ckpt.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)
        torch.load(ckpt_path)

    opt = optim.AdamW(model.parameters(), lr=1e-3)
    trainaudit.snapshot_build(model, opt)

    for step in range(n_steps):
        trainaudit.set_step(step)
        x = torch.randn(2, 32, d_model)
        loss = model(x).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

    results = trainaudit.run_rules()
    print(trainaudit.summarize(results))
    violated = [r for r in results if r.violated]
    if violated:
        print(f"\n[HUNT/trainaudit] BUG DETECTED via {len(violated)} rule(s):")
        for v in violated:
            print(f"   - {v.rule_id}: {v.message}")
            if v.evidence:
                print(f"     evidence: {v.evidence}")
    else:
        print(f"\n[HUNT/trainaudit] CLEAN: no rule violations "
              f"(norm={norm_type} layers={n_layers} ckpt={use_ckpt})")
    trainaudit.disable()


if __name__ == "__main__":
    main()
