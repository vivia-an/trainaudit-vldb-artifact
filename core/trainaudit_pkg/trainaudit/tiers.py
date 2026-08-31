"""Observability tiers for the portability-versus-coverage axis."""
from enum import IntEnum


class Tier(IntEnum):
    T0_PYTORCH      = 0  # torch.distributed + nn.Module + optim base + build snapshot
    T1_FW_METADATA  = 1  # + read framework attrs (param.tensor_model_parallel, etc.)
    T2_FW_PRIMITIVE = 2  # + hook framework primitives (parallel_state, engine props)
    T3_FW_SPECIFIC  = 3  # + hook framework specific methods (finish_grad_sync, etc.)
    T4_INSTANCE     = 4  # + per-bug instance detectors (regression baseline)
