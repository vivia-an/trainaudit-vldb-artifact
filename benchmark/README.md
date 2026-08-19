# Silent Error Reproduction Benchmark

Reproducible silent errors (silent data corruption bugs) in distributed LLM training frameworks.

Each bug includes:
- **`config.json`** — bug metadata, commits, trigger conditions
- **`detect.py`** — self-contained detection script (no external dependencies beyond PyTorch + Megatron)
- **`run.sh`** — training launcher with trigger parameters
- **`reproduce.sh`** — one-click reproduction: checkout buggy → detect → checkout fixed → verify

## Prerequisites

- 2+ NVIDIA GPUs with CUDA
- PyTorch >= 2.0
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) repository (cloned with full history)

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
export MEGATRON_DIR=/path/to/Megatron-LM
```

## Quick Start

```bash
# Reproduce a single bug
cd bugs/M-020
MEGATRON_DIR=/path/to/Megatron-LM bash reproduce.sh
```

## Bug Catalog

| Bug | Category | Parallel | Detection Method | Description |
|-----|----------|----------|-----------------|-------------|
| [M-010](bugs/M-010/) | control_flow | — | function call counting | Activation checkpointing causes aux_loss to be accumulated twice |
| [M-012](bugs/M-012/) | dtype | — | dtype invariant | expert_bias silently downcast from fp32 to bf16 by Float16Module |
| [M-014](bugs/M-014/) | numerical | — | value invariant | topk=1 with post-softmax gives trivial probs (all 1.0), zero router gradient |
| [M-020](bugs/M-020/) | config_validation | PP | structural invariant | Integer division silently drops layers when num_layers % pp_size != 0 |
| [M-024](bugs/M-024/) | dtype | — | dtype invariant | Router input jitter silently promotes bf16 input to fp32 |
| [M-033](bugs/M-033/) | numerical | DP | value scaling check | Global aux_loss gradient missing dp_size scaling factor |

## Detection Methods

### 1. Cross-Rank Equality (M-005)
Compare tensor checksums across ranks in a parallel group. Parameters that should be identical (e.g., non-sharded router weights in TP) are hashed and all-gathered for comparison.

### 2. Function Call Counting (M-010)
Hook framework-internal functions and count invocations per training step. Detects when a function is called more times than expected (e.g., aux_loss saved twice due to recompute).

### 3. Structural Invariant (M-020)
Inspect the model architecture at runtime and compare against configuration. Detects when the actual model structure doesn't match what was configured (e.g., fewer layers than specified).

## How Detection Works

Each `detect.py` follows the same pattern:

```python
# 1. Import and patch megatron's train_step
import megatron.training.training as mtt
_orig = mtt.train_step

def _patched(*args, **kwargs):
    result = _orig(*args, **kwargs)
    # ... check invariant ...
    return result

mtt.train_step = _patched

# 2. Execute Megatron's own entry point
exec(open("pretrain_gpt.py").read())
```

No modifications to Megatron source code — detection is done via monkey-patching at import time.

## Directory Structure

```
benchmark/
├── README.md
├── tools/
│   └── gen_fake_data.py     # Generate fake data for old Megatron versions
└── bugs/
    └── M-XXX/
        ├── config.json      # Bug metadata
        ├── detect.py        # Detection script (entry point for torchrun)
        ├── run.sh           # Training launcher
        └── reproduce.sh     # One-click reproduction
```

## Adding a New Bug

1. Create `bugs/M-XXX/` directory
2. Write `config.json` with buggy/fixed commits and trigger conditions
3. Write `detect.py` that patches megatron internals and checks an invariant
4. Write `run.sh` with the minimal training arguments that trigger the bug
5. Write `reproduce.sh` following the template from existing bugs
6. Test: buggy version should print `BUG DETECTED`, fixed version should print `CLEAN` or crash with an assertion

## Notes

- Scripts temporarily checkout specific commits in the Megatron repo and restore `main` afterward
- Each run creates a timestamped directory under `bugs/M-XXX/runs/` for logs
- Some old Megatron versions don't support `--mock-data`; use `tools/gen_fake_data.py` to generate fake training data
- `CUDA_DEVICE_MAX_CONNECTIONS=1` is required by Megatron's async gradient all-reduce
- Old Megatron versions need `tools/__init__.py` created after checkout (handled automatically)
