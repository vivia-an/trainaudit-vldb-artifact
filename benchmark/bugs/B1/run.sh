#!/bin/bash
set -euo pipefail
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to Megatron-LM repo root}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../../tools" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$MEGATRON_DIR:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/.data}"
mkdir -p "$DATA_DIR"
DATA_PREFIX="$DATA_DIR/fake_document"
if [ ! -f "${DATA_PREFIX}.bin" ]; then
    python "$TOOLS_DIR/gen_fake_data.py" --output-prefix "$DATA_PREFIX"
fi

cd "$MEGATRON_DIR"
[ -f tools/__init__.py ] || touch tools/__init__.py

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT:-29564} \
    "$SCRIPT_DIR/detect.py" \
    --num-layers 2 --hidden-size 128 --num-attention-heads 4 \
    --seq-length 64 --max-position-embeddings 64 \
    --micro-batch-size 2 --global-batch-size 4 \
    --train-iters ${NUM_STEPS:-2} --lr 1e-4 --min-lr 1e-5 \
    --lr-warmup-iters 1 --lr-decay-iters ${NUM_STEPS:-2} --lr-decay-style cosine \
    --log-interval 1 --eval-interval 1000 --eval-iters 0 \
    --bf16 --no-save-optim --no-save-rng \
    --tokenizer-type NullTokenizer --vocab-size 50304 \
    --data-path "$DATA_PREFIX" --split 100,0,0 \
    --tensor-model-parallel-size 2 \
    --num-experts 2 \
    --sequence-parallel \
    --transformer-impl local \
    2>&1
