#!/bin/bash
# M-NEW-MUON-MTP long-run reproduction harness.
# Runs both buggy and fixed commits with telemetry enabled, ~1500 steps each,
# producing per-step JSONL of {loss, grad_norm, embed_proj_checksum} for the
# Fig 2 latency curves.
#
# Usage:
#   MEGATRON_DIR=/path/to/megatron-muon-mtp \
#   BUGGY_COMMIT=<sha> FIXED_COMMIT=<sha> \
#       bash reproduce_long.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM clone}"
BUGGY_COMMIT="${BUGGY_COMMIT:?Set BUGGY_COMMIT (parent of PR #4642 fix)}"
FIXED_COMMIT="${FIXED_COMMIT:?Set FIXED_COMMIT (PR #4642 fix or descendant)}"
MUON_FLAGS="${MUON_FLAGS:---optimizer muon --use-distributed-optimizer}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SCRIPT_DIR/runs/long_$TS"
mkdir -p "$RUN_DIR/buggy" "$RUN_DIR/fixed"

echo "=== M-NEW-MUON-MTP Long-Run Reproduction ==="
echo "Run: $RUN_DIR"
echo "Muon flags: $MUON_FLAGS"

checkout() {
    cd "$MEGATRON_DIR"
    git stash -q 2>/dev/null || true
    git checkout -q "$1" 2>/dev/null || git checkout "$1"
}

run_phase() {
    local phase=$1; local commit=$2; local port=$3
    echo
    echo "[$phase] Running $commit ..."
    checkout "$commit"
    set +eo pipefail
    MEGATRON_DIR="$MEGATRON_DIR" \
    DETECT_OUTPUT_DIR="$RUN_DIR/$phase" \
    MUON_FLAGS="$MUON_FLAGS" \
    STEP_LOG_PATH="$RUN_DIR/$phase/steps.jsonl" \
    MASTER_PORT="$port" \
        bash "$SCRIPT_DIR/run_long.sh" 2>&1 | tee "$RUN_DIR/$phase/run.log"
    set -eo pipefail
}

run_phase buggy "$BUGGY_COMMIT" 29501
run_phase fixed "$FIXED_COMMIT" 29502

cd "$MEGATRON_DIR"
git checkout main -q 2>/dev/null || git checkout master -q 2>/dev/null || true

echo
echo "=== Done. Logs: $RUN_DIR/ ==="
echo "Buggy verdict:"  ; grep -E "BUG DETECTED|CLEAN|INCONCLUSIVE" "$RUN_DIR/buggy/run.log" || true
echo "Fixed verdict:"  ; grep -E "BUG DETECTED|CLEAN|INCONCLUSIVE" "$RUN_DIR/fixed/run.log" || true

echo
echo "=== Sanity check: fixed run rank0 vs rank1 embed checksum ==="
python3 "$SCRIPT_DIR/sanity_fixed_consistency.py" "$RUN_DIR/fixed/steps.jsonl" \
    || { echo "SANITY FAILED — telemetry implementation bug, not a model bug"; exit 3; }
