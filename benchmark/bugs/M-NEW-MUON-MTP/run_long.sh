#!/bin/bash
# M-NEW-MUON-MTP long-run experiment for §2.1 Fig 2 latency curves.
#
# Goal: produce buggy-vs-fixed time series of {loss, grad_norm,
# embed_proj_checksum} over ~1500-2000 steps so we can show that
# TrainAudit fires at step 0 (init-time tag check) and at step 1
# (P3 drift query) while loss/grad-norm stay visually identical to a
# clean run for hundreds of steps.
#
# Critical: Muon optimizer MUST be enabled, else there is no Muon-vs-AdamW
# split and no drift. Default MUON_FLAGS below matches upstream
# arguments.py:2574: --optimizer muon turns it on; --use-distributed-optimizer
# is recommended (the legacy --optimizer dist_muon is deprecated).
# Defaults for --muon-momentum / --muon-scale-mode / --muon-ns-steps /
# --muon-scalar-optimizer are sane (the latter is adam by default, which is
# precisely how the bug is triggered: tagged params -> adam, untagged ->
# Muon Newton-Schulz; on the fixed commit both replicas are tagged so both
# go to adam and remain bit-identical).
#
# Usage:
#   MEGATRON_DIR=/path/to/megatron-muon-mtp \
#   STEP_LOG_PATH=/path/to/run_buggy/steps.jsonl \
#   bash run_long.sh
#
# Run buggy then fixed by checking out the corresponding commit between
# invocations (or use reproduce.sh's pattern of doing both back-to-back).
set -euo pipefail

MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$SCRIPT_DIR:$MEGATRON_DIR:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LONG_RUN=1
export STEP_LOG_PATH="${STEP_LOG_PATH:-/tmp/m_new_muon_mtp_steps.jsonl}"

# Larger config than detect.py default: hidden+layers chosen so the bug's
# silent-burn phase is long enough to show non-trivial detection latency.
# Adjust to your GPU budget.
cd "$MEGATRON_DIR"
torchrun --nproc_per_node=${NPROC:-2} --master_port=${MASTER_PORT:-29501} \
    "$SCRIPT_DIR/detect.py" \
    --num-layers ${NUM_LAYERS:-12} \
    --hidden-size ${HIDDEN:-1024} \
    --num-attention-heads ${NHEAD:-16} \
    --seq-length ${SEQLEN:-512} \
    --max-position-embeddings ${SEQLEN:-512} \
    --micro-batch-size ${MBS:-1} \
    --global-batch-size ${GBS:-8} \
    --train-iters ${NUM_STEPS:-1500} \
    --lr ${LR:-1e-4} \
    --min-lr ${MIN_LR:-1e-5} \
    --lr-warmup-iters ${WARMUP:-50} \
    --lr-decay-iters ${NUM_STEPS:-1500} \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-interval 100000 \
    --eval-iters 0 \
    --bf16 \
    --no-save-optim \
    --no-save-rng \
    --tokenizer-type NullTokenizer \
    --vocab-size 50304 \
    --mock-data \
    --pipeline-model-parallel-size ${PP_SIZE:-2} \
    --tensor-model-parallel-size ${TP_SIZE:-1} \
    --position-embedding-type rope \
    --mtp-num-layers ${MTP_NUM_LAYERS:-1} \
    --mtp-loss-scaling-factor ${MTP_LOSS_SCALE:-0.1} \
    ${MUON_FLAGS:---optimizer muon --use-distributed-optimizer} \
    ${EXTRA_FLAGS:-} \
    2>&1 | tee "${STEP_LOG_PATH}.train.log"
