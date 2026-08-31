#!/bin/bash
set -euo pipefail
OLMO_DIR="${OLMO_DIR:?Set OLMO_DIR to OLMo repo root}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$OLMO_DIR:${PYTHONPATH:-}"
# Must checkout to buggy commit: git -C $OLMO_DIR checkout a57f3803~1
python3 "$SCRIPT_DIR/detect.py"
