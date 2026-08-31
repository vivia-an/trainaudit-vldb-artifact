#!/bin/bash
set -euo pipefail
DS_DIR="${DS_DIR:?Set DS_DIR to DeepSpeed repo root}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../../tools" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$TOOLS_DIR:$DS_DIR:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT:-29559} \
    "$SCRIPT_DIR/detect.py" \
    2>&1
