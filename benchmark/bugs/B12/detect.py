"""
B12 / OLMo-core PR 27: AdamWConfig.build forgets to set initial_lr on param_groups.

Bug at olmo_core/optim/__init__.py: AdamWConfig.build(model) returns
torch.optim.AdamW(self.build_groups(model), **kwargs) without copying lr
into 'initial_lr'. Standard PyTorch LRSchedulers require 'initial_lr' in
each param_group when last_epoch >= 0 (i.e. on restart).

Fix sets group['initial_lr'] = self.lr after constructing the optimizer.

Detection (end-to-end):
  - Build a small real model + AdamWConfig optimizer.
  - Run real forward + backward + optim.step on GPU.
  - After step, attempt to construct an LRScheduler with last_epoch=0 (resume).
  - Inspect optim.param_groups for 'initial_lr'.
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn

# Initialize distributed (single GPU torchrun also works).
dist.init_process_group(backend="nccl")
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

from olmo_core.optim import AdamWConfig

torch.manual_seed(0)
model = nn.Sequential(
    nn.Embedding(1024, 128),
    nn.Linear(128, 1024),
).to(device)
model.train()

config = AdamWConfig(lr=3e-4)
optim = config.build(model)

# Real backward + step
input_ids = torch.randint(0, 1024, (2, 16), device=device)
for step in range(3):
    optim.zero_grad()
    logits = model(input_ids)
    loss = logits.float().sum()
    loss.backward()
    optim.step()

# --- Detection ---
groups = optim.param_groups
n_groups = len(groups)
n_with_initial = sum(1 for g in groups if "initial_lr" in g)

# Try the real failure mode: simulate restart by constructing a scheduler
# with last_epoch != -1. PyTorch raises KeyError if 'initial_lr' is missing.
restart_error = None
try:
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda e: 1.0, last_epoch=0)
    _ = sched.get_last_lr()
except Exception as e:  # noqa: BLE001
    restart_error = f"{type(e).__name__}: {e}"

if dist.get_rank() == 0:
    print(f"[DETECT] num_param_groups = {n_groups}")
    print(f"[DETECT] num_with_initial_lr = {n_with_initial}")
    print(f"[DETECT] restart_scheduler_error = {restart_error}")
    print(f"[DETECT] sample group keys = {sorted(groups[0].keys())}")

    if n_with_initial == 0 or restart_error is not None:
        print(f"[BUG] AdamWConfig.build did not set initial_lr on {n_groups - n_with_initial}/{n_groups} groups")
        print("[RESULT] BUG DETECTED")
    else:
        print("[RESULT] CLEAN: all param_groups carry initial_lr")

dist.barrier()
dist.destroy_process_group()
