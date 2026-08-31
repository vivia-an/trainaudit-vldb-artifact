#!/bin/bash
set -euo pipefail
OLMO_DIR="${OLMO_DIR:?Set OLMO_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$OLMO_DIR:${PYTHONPATH:-}"
python3 "$SCRIPT_DIR/detect.py"
