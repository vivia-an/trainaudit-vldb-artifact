#!/bin/bash
# M-020: Training launcher
# Trigger: PP=2 + num_layers not divisible by pp_size (5 % 2 != 0)
set -euo pipefail

MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM clone}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$SCRIPT_DIR:$MEGATRON_DIR:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd "$MEGATRON_DIR"

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT:-29501} \
    "$SCRIPT_DIR/detect.py" \
    --num-layers 5 \
    --hidden-size 128 \
    --num-attention-heads 4 \
    --seq-length 64 \
    --max-position-embeddings 64 \
    --micro-batch-size 2 \
    --global-batch-size 4 \
    --train-iters ${NUM_STEPS:-3} \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-warmup-iters 1 \
    --lr-decay-iters ${NUM_STEPS:-3} \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --bf16 \
    --no-save-optim \
    --no-save-rng \
    --tokenizer-type NullTokenizer \
    --vocab-size 50304 \
    --mock-data \
    --pipeline-model-parallel-size 2 \
    2>&1
