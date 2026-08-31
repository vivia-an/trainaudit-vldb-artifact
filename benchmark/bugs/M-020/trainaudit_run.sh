#!/bin/bash
set -euo pipefail
MEGATRON_DIR="${MEGATRON_DIR:?}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUGGY_COMMIT="64d816a39c41c17628eee49838a59c9693b6b036"
FIXED_COMMIT="99f999a46670359ecbba4ad82265901009722d8c"
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
    set +e
    torchrun --nproc_per_node=2 --master_port=$((29650 + RANDOM % 1000)) \
        "$SCRIPT_DIR/trainaudit_driver.py" \
        --num-layers 5 --hidden-size 128 --num-attention-heads 4 \
        --seq-length 64 --max-position-embeddings 64 \
        --micro-batch-size 2 --global-batch-size 4 \
        --train-iters 2 --lr 1e-4 --min-lr 1e-5 \
        --lr-warmup-iters 1 --lr-decay-iters 2 --lr-decay-style cosine \
        --log-interval 1 --eval-interval 1000 --eval-iters 0 \
        --bf16 --no-save-optim --no-save-rng \
        --tokenizer-type NullTokenizer --vocab-size 50304 --mock-data \
        --pipeline-model-parallel-size 2 \
        2>&1
    set -e
    echo
}
run_one "BUGGY" "$BUGGY_COMMIT"
run_one "FIXED" "$FIXED_COMMIT"
