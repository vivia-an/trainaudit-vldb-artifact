# `loss.compute.post` Hookpoint — Adapter Spec

> Per brief 32 §3.3: 1 NEW hookpoint to add (the only addition; other 7 reuse existing).
> Patch should land in 4 framework adapters.

## Hookpoint signature

```python
def _loss_compute_post(self,
                      loss_components: Dict[str, Tensor],
                      divisors: Dict[str, int],
                      step: int) -> None:
    """
    Capture per-component loss numerator + divisor for P16 verification.

    Args:
        loss_components: {component_name: scalar_tensor of summed contributions}
        divisors: {component_name: divisor used (token count / micro-batch / etc.)}
        step: current training step
    """
    for name, value in loss_components.items():
        self.collector.emit_event({
            "hook": "loss.compute.post",
            "component_name": name,
            "sum": float(value.detach().sum()),
            "divisor": divisors.get(name, 1),
            "dtype": str(value.dtype),
            "mask_count": ...,  # if applicable
            "step": step,
        })
```

## Per-framework integration points (where to register the hook)

| Framework | Integration point | Estimated LoC |
|-----------|-------------------|---------------|
| Megatron-LM | `pretrain_*.py` end of `forward_step()`, after `loss = ...` and before `return loss` | ~30 |
| DeepSpeed | `engine.py::_compute_loss()` post; or wrap `model.module.compute_loss` | ~50 |
| OLMo | `train.py::_train_microbatch()` after `loss.backward()` invocation site | ~40 |
| OLMo-core | `trainer.py::compute_loss()` post-hook | ~30 |

Total: ~150 LoC across 4 adapters.

## Smoke-test plan

After adapter integration:

```bash
# Per framework, run a clean 200-step training (no bug) with full P1-P16 checks active.
# Expect: 0 false positives across all 16 patterns.

# Pseudo:
for fw in megatron-lm deepspeed olmo olmo-core; do
    ssh $GPU_HOST "cd $repo && \
        TRAINAUDIT_ACTIVE_PATTERNS=P1,P2,...,P16 \
        python3 -m $fw.pretrain --config clean_config.yaml --max_steps 200" 2>&1 | tee smoke_$fw.log
    grep "VIOLATION" smoke_$fw.log  # should be empty for clean run
done
```

## Decision: failure threshold

Per brief §9 失败处理:
- 0 false positives → ✓ deploy approved
- 1-3 false positives → tighten precond_rho on offending pattern
- ≥4 false positives → revisit Phase 1 prompt design

## Implementation status

⏸️ **Not implemented** in this experiment iteration. The 4-framework collector adapter changes are paper engineering work (~150 LoC + 200-step smoke test runs) that the paper team owns.

The **spec above is sufficient** for the paper team to implement. All P9-P16 inline rule checks already work on the surrogates (see `benchmark/eval/d2_extension/trainaudit_inline_d2.py` 8/8 ✓).
