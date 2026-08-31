#!/bin/bash
set -euo pipefail
OLMO_CORE_DIR="${OLMO_CORE_DIR:?Set OLMO_CORE_DIR to OLMo-core repo root}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$OLMO_CORE_DIR"
export PYTHONPATH="$OLMO_CORE_DIR/src:${PYTHONPATH:-}"
torchrun --nproc_per_node=1 --master_port=${MASTER_PORT:-29551} \
    "$SCRIPT_DIR/detect.py" \
    2>&1
