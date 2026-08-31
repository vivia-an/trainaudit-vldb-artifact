"""Per-step telemetry for the long-run experiment (Fig 2 latency curves).

Side-effect import: patches megatron.training.training.train_step to write a
JSONL record per rank per step. detect.py imports this module when LONG_RUN=1.

Records (rank 0 carries loss/grad_norm; every PP-stage rank that owns a tied
word_embeddings replica also carries an embedding-projection checksum):
    step, rank, pp_rank, pp_size, embed_name,
    embed_proj_checksum, embed_norm,
    loss (rank 0), grad_norm (rank 0)

The projection checksum is a deterministic scalar hash:
    rng   = torch.Generator(seed=0xCAFE)
    proj  = randn(hidden_size, generator=rng).to(device, dtype)
    ck    = (embed.float() @ proj.float()).sum()
Two PP ranks holding bit-equal embeddings produce equal ck up to FP32 rounding;
divergence in ck across ranks is the bug's physical signature.

Activation:
    LONG_RUN=1
    STEP_LOG_PATH=/path/to/steps.jsonl    # files written as .rank{N}
"""
import json
import os
import sys

import megatron.training.training as mtt

_step_log_path = os.environ.get(
    "STEP_LOG_PATH", "/tmp/m_new_muon_mtp_steps.jsonl"
)
_orig_train_step = getattr(mtt, "train_step", None)
_step_counter = [0]
_proj_cache = {}
_log_handle = [None]


def _find_tied_embedding(model_chunks):
    chunks = model_chunks if isinstance(model_chunks, list) else [model_chunks]
    for chunk in chunks:
        try:
            named = list(chunk.named_parameters())
        except Exception:
            continue
        for name, p in named:
            if "word_embeddings.weight" in name:
                return name, p
    return None, None


def _embed_checksum(embed):
    import torch
    key = (embed.shape[-1], str(embed.device), str(embed.dtype))
    if key not in _proj_cache:
        rng = torch.Generator().manual_seed(0xCAFE)
        proj = torch.randn(
            embed.shape[-1], generator=rng, dtype=torch.float32
        ).to(embed.device)
        _proj_cache[key] = proj
    proj = _proj_cache[key]
    with torch.no_grad():
        e = embed.detach().to(torch.float32)
        ck = (e @ proj).sum().item()
        norm = e.norm().item()
    return ck, norm


def _capture_step(loss_dict, grad_norm, model_chunks):
    import torch
    from megatron.core import mpu
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    try:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
    except Exception:
        pp_rank, pp_size = rank, 1

    name, embed = _find_tied_embedding(model_chunks)
    if embed is not None:
        ck, norm = _embed_checksum(embed)
    else:
        ck, norm = None, None

    record = {
        "step": _step_counter[0],
        "rank": rank,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "embed_name": name,
        "embed_proj_checksum": ck,
        "embed_norm": norm,
    }
    if rank == 0:
        if isinstance(loss_dict, dict):
            for k in ("lm loss", "loss", "lm_loss"):
                if k in loss_dict:
                    v = loss_dict[k]
                    record["loss"] = float(v.item() if hasattr(v, "item") else v)
                    break
        record["grad_norm"] = (
            float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            if grad_norm is not None else None
        )

    if _log_handle[0] is None:
        path = f"{_step_log_path}.rank{rank}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _log_handle[0] = open(path, "a", buffering=1)
        _log_handle[0].write(
            json.dumps({"_meta": "M-NEW-MUON-MTP telemetry", "rank": rank}) + "\n"
        )
    _log_handle[0].write(json.dumps(record) + "\n")


def _patched_train_step(*args, **kwargs):
    result = _orig_train_step(*args, **kwargs)
    try:
        loss_dict = result[0] if len(result) > 0 else None
        grad_norm = result[5] if len(result) > 5 else None
        model_chunks = args[2] if len(args) > 2 else kwargs.get("model")
        if model_chunks is not None:
            _capture_step(loss_dict, grad_norm, model_chunks)
        _step_counter[0] += 1
    except Exception as e:
        print(
            f"[telemetry] step {_step_counter[0]} capture failed: {e!r}",
            file=sys.stderr, flush=True,
        )
    return result


if _orig_train_step is not None:
    mtt.train_step = _patched_train_step
    print(
        f"[telemetry] LONG_RUN telemetry active -> {_step_log_path}.rank*",
        flush=True,
    )
else:
    print(
        "[telemetry] WARN: megatron.training.training.train_step not found",
        file=sys.stderr, flush=True,
    )
