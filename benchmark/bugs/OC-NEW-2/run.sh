#!/bin/bash
set -euo pipefail
OLMO_CORE_DIR="${OLMO_CORE_DIR:?Set OLMO_CORE_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$OLMO_CORE_DIR/src:${PYTHONPATH:-}"
python3 "$SCRIPT_DIR/detect.py"
