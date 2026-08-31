#!/bin/bash
# Drive a single TrainCheck check on one D1 surrogate bug.
#
# Usage: run_one.sh <bug_id>
#
#   - Runs traincheck-collect on bug_id_fixed.py to build the reference trace
#   - Runs traincheck-infer on the reference trace → invariants.json
#   - Runs traincheck-collect on bug_id_buggy.py to build the buggy trace
#   - Runs traincheck-check on the buggy trace against invariants.json
#   - Emits the contract line:
#       [<bug_id>] BUG DETECTED: traincheck:<N>/<M> failed invariants
#       [<bug_id>] CLEAN: traincheck:0/<M> failed
#       [<bug_id>] FAIL: <stage>: <reason>

set -uo pipefail
BUG_ID="${1:?usage: run_one.sh <bug_id>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${TC_WORK_DIR:-/tmp/tc_$BUG_ID}"
PY="${TC_PYTHON:-python3}"

FIXED="$HERE/${BUG_ID}_fixed.py"
BUGGY="$HERE/${BUG_ID}_buggy.py"

if [[ ! -f "$FIXED" || ! -f "$BUGGY" ]]; then
    echo "[$BUG_ID] FAIL: missing_surrogate: need $FIXED and $BUGGY"
    exit 0
fi

rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK"

# 1) Reference trace (fixed)
"$PY" -m traincheck.collect_trace -p "$FIXED" --output-dir trace_fixed \
    --models-to-track model > collect_fixed.log 2>&1
if [[ ! -d trace_fixed ]] || ! ls trace_fixed/proxy_log.json* trace_fixed/trace_* > /dev/null 2>&1; then
    echo "[$BUG_ID] FAIL: collect_fixed: $(tail -3 collect_fixed.log | tr '\n' ' ')"
    exit 0
fi

# 2) Infer invariants
"$PY" -m traincheck.infer_engine -f trace_fixed > infer.log 2>&1
INV_FILE="$(ls invariants*.json 2>/dev/null | head -1)"
if [[ -z "$INV_FILE" ]]; then
    echo "[$BUG_ID] FAIL: infer: $(tail -3 infer.log | tr '\n' ' ')"
    exit 0
fi
N_INV="$(wc -l < "$INV_FILE" 2>/dev/null || echo 0)"

# 3) Buggy trace
"$PY" -m traincheck.collect_trace -p "$BUGGY" --output-dir trace_buggy \
    --models-to-track model > collect_buggy.log 2>&1
if [[ ! -d trace_buggy ]] || ! ls trace_buggy/proxy_log.json* trace_buggy/trace_* > /dev/null 2>&1; then
    echo "[$BUG_ID] FAIL: collect_buggy: $(tail -3 collect_buggy.log | tr '\n' ' ')"
    exit 0
fi

# 4) Check
"$PY" -m traincheck.checker -f trace_buggy -i "$INV_FILE" > check.log 2>&1
FAILED_LINE="$(grep -E "Total failed invariants:" check.log | tail -1)"
N_FAILED="$(echo "$FAILED_LINE" | grep -oE "[0-9]+/[0-9]+" | head -1 | cut -d/ -f1)"
N_TOTAL="$(echo "$FAILED_LINE" | grep -oE "[0-9]+/[0-9]+" | head -1 | cut -d/ -f2)"

if [[ -z "$N_FAILED" ]]; then
    echo "[$BUG_ID] FAIL: check: $(tail -3 check.log | tr '\n' ' ')"
    exit 0
fi

if [[ "$N_FAILED" -gt 0 ]]; then
    echo "[$BUG_ID] BUG DETECTED: traincheck:$N_FAILED/$N_TOTAL failed invariants"
else
    echo "[$BUG_ID] CLEAN: traincheck:0/$N_TOTAL failed"
fi
