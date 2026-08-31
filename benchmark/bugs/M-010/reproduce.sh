#!/bin/bash
# M-010: MoE aux_loss accumulated twice with activation checkpointing
# Issue: https://github.com/NVIDIA/Megatron-LM/issues/784
#
# Usage:
#   MEGATRON_DIR=/path/to/Megatron-LM bash reproduce.sh
#
# Prerequisites:
#   - 2+ GPUs with CUDA
#   - PyTorch >= 2.0
#   - Megatron-LM repo (will checkout specific commits)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM clone}"

BUGGY_COMMIT="892c4c85448c6f81ee8bf8347c430ecba528d634"
FIXED_COMMIT="e6d56d6828c0773f55772b92b2ec0eed5639665e"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SCRIPT_DIR/runs/$TS"
mkdir -p "$RUN_DIR"

echo "=== M-010 Reproduction ==="
echo "Run: $RUN_DIR"

checkout() {
    cd "$MEGATRON_DIR"
    git stash -q 2>/dev/null || true
    rm -f tools/__init__.py tools/retro/__init__.py 2>/dev/null || true
    git checkout -q "$1" 2>/dev/null || git checkout "$1" 2>/dev/null
    [ -d tools ] && [ ! -f tools/__init__.py ] && touch tools/__init__.py || true
    [ -d tools/retro ] && [ ! -f tools/retro/__init__.py ] && touch tools/retro/__init__.py || true
}

# --- [1/2] Buggy version ---
echo ""
echo "[1/2] Running buggy version..."
checkout "$BUGGY_COMMIT"
set +eo pipefail
MEGATRON_DIR="$MEGATRON_DIR" DETECT_OUTPUT_DIR="$RUN_DIR" \
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
MEGATRON_DIR="$MEGATRON_DIR" DETECT_OUTPUT_DIR="$RUN_DIR" MASTER_PORT=29502 \
    bash "$SCRIPT_DIR/run.sh" 2>&1 | tee "$RUN_DIR/fixed.log"
set -eo pipefail

if grep -q "CLEAN" "$RUN_DIR/fixed.log"; then
    echo ">>> Fixed: CLEAN (expected)"
else
    echo ">>> Fixed: unexpected result"
fi

# --- Cleanup ---
cd "$MEGATRON_DIR"
git checkout main -q 2>/dev/null || git checkout master -q 2>/dev/null || true

echo ""
echo "=== Results ==="
echo "Logs: $RUN_DIR/"
