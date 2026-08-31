#!/bin/bash
# M-020: Pipeline parallel silently trains with fewer layers than configured
# Issue: https://github.com/NVIDIA/Megatron-LM/issues/468
#
# Usage:
#   MEGATRON_DIR=/path/to/Megatron-LM bash reproduce.sh
#
# Prerequisites:
#   - 2+ GPUs with CUDA
#   - PyTorch >= 2.0
#   - Megatron-LM repo (will checkout specific commits)
#
# Expected behavior:
#   - Buggy version: trains successfully but with 4 layers instead of 5
#   - Fixed version: crashes with assertion (correct — rejects bad config)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM clone}"

BUGGY_COMMIT="64d816a39c41c17628eee49838a59c9693b6b036"
FIXED_COMMIT="99f999a46670359ecbba4ad82265901009722d8c"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SCRIPT_DIR/runs/$TS"
mkdir -p "$RUN_DIR"

echo "=== M-020 Reproduction ==="
echo "Run: $RUN_DIR"

checkout() {
    cd "$MEGATRON_DIR"
    git stash -q 2>/dev/null || true
    rm -f tools/__init__.py tools/retro/__init__.py 2>/dev/null || true
    git checkout -fq "$1" 2>/dev/null || git checkout -f "$1" 2>/dev/null
    find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    [ -d tools ] && [ ! -f tools/__init__.py ] && touch tools/__init__.py || true
    [ -d tools/retro ] && [ ! -f tools/retro/__init__.py ] && touch tools/retro/__init__.py || true
}

# --- [1/2] Buggy version ---
echo ""
echo "[1/2] Running buggy version..."
checkout "$BUGGY_COMMIT"
set +eo pipefail
MEGATRON_DIR="$MEGATRON_DIR" \
    bash "$SCRIPT_DIR/run.sh" 2>&1 | tee "$RUN_DIR/buggy.log"
set -eo pipefail

if grep -q "BUG DETECTED" "$RUN_DIR/buggy.log"; then
    echo ">>> Buggy: BUG DETECTED (expected)"
else
    echo ">>> Buggy: not detected (unexpected)"
fi

# --- [2/2] Fixed version ---
echo ""
echo "[2/2] Running fixed version..."
checkout "$FIXED_COMMIT"
set +eo pipefail
MEGATRON_DIR="$MEGATRON_DIR" MASTER_PORT=29502 \
    bash "$SCRIPT_DIR/run.sh" > "$RUN_DIR/fixed.log" 2>&1
set -eo pipefail

if grep -q "AssertionError" "$RUN_DIR/fixed.log" || \
   grep -q "Number of layers should be divisible" "$RUN_DIR/fixed.log"; then
    echo ">>> Fixed: ASSERT fired (expected — bug prevented)"
elif grep -q "CLEAN" "$RUN_DIR/fixed.log"; then
    echo ">>> Fixed: CLEAN (expected)"
else
    echo ">>> Fixed: unexpected result — check $RUN_DIR/fixed.log"
fi

# --- Cleanup ---
cd "$MEGATRON_DIR"
git checkout main -q 2>/dev/null || git checkout master -q 2>/dev/null || true

echo ""
echo "=== Results ==="
echo "Logs: $RUN_DIR/"
