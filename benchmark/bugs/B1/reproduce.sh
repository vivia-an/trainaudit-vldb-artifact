#!/bin/bash
set -euo pipefail
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to Megatron-LM repo root}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUGGY_COMMIT="3c637fc0d~1"
FIXED_COMMIT="3c637fc0d"

cd "$MEGATRON_DIR"
ORIG=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)
trap 'cd "$MEGATRON_DIR" && git checkout -q "$ORIG" 2>/dev/null || true' EXIT

run_one() {
    local label="$1" rev="$2"
    echo "===== $label ($rev) ====="
    cd "$MEGATRON_DIR"
    rm -f tools/__init__.py 2>/dev/null || true
    git checkout -q "$rev"
    [ -f tools/__init__.py ] || touch tools/__init__.py
    set +e
    bash "$SCRIPT_DIR/run.sh"
    set -e
    echo
}

run_one "BUGGY" "$BUGGY_COMMIT"
run_one "FIXED" "$FIXED_COMMIT"
