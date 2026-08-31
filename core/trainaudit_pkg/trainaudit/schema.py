"""Stable attribute-key namespace used by adapters and guarded rules."""

SCHEMA_VERSION = 1

# ---- Hookpoint names (stable) ----
HP_BUILD_SNAPSHOT     = "build.snapshot"
HP_COMM_PRE           = "comm.pre"
HP_COMM_POST          = "comm.post"
HP_MODULE_FWD_PRE     = "module.fwd.pre"
HP_MODULE_FWD_POST    = "module.fwd.post"
HP_MODULE_BWD         = "module.bwd"
HP_OPTIM_STEP_PRE     = "optim.step.pre"
HP_OPTIM_STEP_POST    = "optim.step.post"
HP_CLIP_GRAD_PRE      = "utils.clip_grad.pre"
HP_CLIP_GRAD_POST     = "utils.clip_grad.post"
HP_DATALOADER_BATCH   = "dataloader.batch"
HP_LR_SCHED_INIT      = "scheduler.init"

# ---- Tensor attribute keys (stable) ----
T_DTYPE      = "dtype"
T_SHAPE      = "shape"
T_DEVICE     = "device"
T_L2_NORM    = "l2_norm"
T_ABS_MAX    = "abs_max"
T_HAS_NAN    = "has_nan"
T_HAS_INF    = "has_inf"
T_CKSUM      = "cksum"          # optional, sampled

# ---- Call attribute keys (stable) ----
C_KIND       = "kind"           # forward | backward | comm | optim_step | clip_grad | init
C_OP         = "op"             # for comm: all_reduce | broadcast | reduce_scatter | all_gather | all_to_all
C_FN_QUAL    = "fn"             # qualified name
C_GROUP_SIZE = "group_size"

# ---- Group attribute keys (T1+ when adapter sets these) ----
G_KIND       = "group.kind"     # tp | dp | pp | ep | cp | composite | world
G_MEMBERS    = "group.members"
