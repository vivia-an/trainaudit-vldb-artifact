#!/bin/bash
set -euo pipefail
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../../tools" && pwd)"

BUGGY_COMMIT="3c637fc0d~1"
FIXED_COMMIT="3c637fc0d"

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/.data}"
mkdir -p "$DATA_DIR"
DATA_PREFIX="$DATA_DIR/fake_document"
if [ ! -f "${DATA_PREFIX}.bin" ]; then
    python "$TOOLS_DIR/gen_fake_data.py" --output-prefix "$DATA_PREFIX"
fi

cd "$MEGATRON_DIR"
ORIG=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)
trap 'cd "$MEGATRON_DIR" && git checkout -q "$ORIG" 2>/dev/null || true' EXIT

run_one() {
    local label="$1" rev="$2"
    echo "===== TrainAudit T1 on $label ($rev) ====="
    cd "$MEGATRON_DIR"
    rm -f tools/__init__.py 2>/dev/null || true
    git checkout -q "$rev"
    [ -f tools/__init__.py ] || touch tools/__init__.py
    export PYTHONPATH="$SCRIPT_DIR:$MEGATRON_DIR:${PYTHONPATH:-}"
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    if [ -n "${TRAINAUDIT_TRACE_DIR:-}" ]; then
        export TRAINAUDIT_DB_PATH="$TRAINAUDIT_TRACE_DIR/B1_${label}_rank{rank}.duckdb"
        mkdir -p "$TRAINAUDIT_TRACE_DIR"
    fi
    set +e
    torchrun --nproc_per_node=2 --master_port=$((29590 + RANDOM % 1000)) \
        "$SCRIPT_DIR/trainaudit_driver.py" \
        --num-layers 2 --hidden-size 128 --num-attention-heads 4 \
        --seq-length 64 --max-position-embeddings 64 \
        --micro-batch-size 2 --global-batch-size 4 \
        --train-iters 2 --lr 1e-4 --min-lr 1e-5 \
        --lr-warmup-iters 1 --lr-decay-iters 2 --lr-decay-style cosine \
        --log-interval 1 --eval-interval 1000 --eval-iters 0 \
        --bf16 --no-save-optim --no-save-rng \
        --tokenizer-type NullTokenizer --vocab-size 50304 \
        --data-path "$DATA_PREFIX" --split 100,0,0 \
        --tensor-model-parallel-size 2 \
        --num-experts 2 \
        --sequence-parallel \
        --transformer-impl local \
        2>&1
    set -e
    echo
}

run_one "BUGGY" "$BUGGY_COMMIT"
run_one "FIXED" "$FIXED_COMMIT"
