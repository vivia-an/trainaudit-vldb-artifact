#!/usr/bin/env bash
# Phase-2: re-mine constraint pool with adversarial verification disabled (theta_conf=0).
# Writes to config/lib_no_adversarial_mined.json (separate from Phase-1 proxy lib_no_adversarial.json).
#
# Estimated: ~350 iterations, ~8h, ~$70 API cost (see paper §5.2).
#
# Usage:
#   cd sdccheck/agents && ../scripts/ablation/run_mining_no_adversarial.sh [max_iterations]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AGENTS="${ROOT}/agents"
OUT_JSON="${ROOT}/config/lib_no_adversarial_mined.json"
SRC_JSON="${ROOT}/config/predefined_constraints.json"
MAX_IT="${1:-350}"

export SDC_ABLATION_NO_ADVERSARIAL=1
export SDC_MIN_CONF=0
export SDC_CONSTRAINTS_OUTPUT="$OUT_JSON"
export SDC_MAX_ITERATIONS="$MAX_IT"

mkdir -p "$(dirname "$OUT_JSON")"
cp -f "$SRC_JSON" "$OUT_JSON"

cd "$AGENTS"
echo "[mining] SDC_ABLATION_NO_ADVERSARIAL=1 max_iterations=$MAX_IT"
echo "[mining] output=$OUT_JSON"
echo "[mining] log=${ROOT}/logs/ablation_no_adv/_mining.log"

mkdir -p "${ROOT}/logs/ablation_no_adv"
nohup python3 -u run_self_reason_with_writeAgent.py \
  > "${ROOT}/logs/ablation_no_adv/_mining.log" 2>&1 &
echo "[mining] pid=$! — tail -f ${ROOT}/logs/ablation_no_adv/_mining.log"
