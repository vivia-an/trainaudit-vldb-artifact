#!/usr/bin/env bash
# Strict re-mining holdout: full pipeline (adversarial verification ON),
# with the 19 Real-SE evaluation cases withheld from all mining evidence.
#
# Category-round steering: one mining round per constraint category, so an
# empty starting library does not collapse into data_parallel-only output.
#
# Usage (run under nohup; the loop runs in the foreground of this script):
#   nohup ./scripts/ablation/run_mining_holdout.sh [iters_per_round] > logs/holdout_mining/_mining_rounds.log 2>&1 &
#
# Env:
#   SDC_FRESH=1          wipe output back to the empty seed before starting
#   SDC_HOLDOUT_CATS     override category list (space-separated)

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AGENTS="${ROOT}/agents"
OUT_JSON="${ROOT}/config/lib_holdout_mined.json"
PER_ROUND="${1:-60}"

CATS="${SDC_HOLDOUT_CATS:-tensor_parallel pipeline_parallel zero_optimization model_integrity training_progress data_parallel}"

export SDC_HOLDOUT_EXCLUDE="${ROOT}/logs/holdout_mining/EXCLUSION_LIST.json"
export SDC_CONSTRAINTS_OUTPUT="$OUT_JSON"
unset SDC_ABLATION_NO_ADVERSARIAL SDC_MIN_CONF 2>/dev/null || true

mkdir -p "$(dirname "$OUT_JSON")" "${ROOT}/logs/holdout_mining"
# Seed only on a fresh start; a restart must not wipe already-mined rules.
if [ ! -s "$OUT_JSON" ] || [ "${SDC_FRESH:-0}" = "1" ]; then
  cp -f "${ROOT}/config/lib_holdout_seed.json" "$OUT_JSON"
  echo "[mining-holdout] seeded fresh library"
fi

cd "$AGENTS"
echo "[mining-holdout] exclusions=$SDC_HOLDOUT_EXCLUDE per_round=$PER_ROUND cats=($CATS)"
echo "[mining-holdout] output=$OUT_JSON"

for cat in $CATS; do
  export SDC_TARGET_CATEGORY="$cat"
  export SDC_MAX_ITERATIONS="$PER_ROUND"
  echo "[mining-holdout] $(date '+%F %T') === round: $cat (${PER_ROUND} iters) ==="
  python3 -u run_self_reason_with_writeAgent.py \
    >> "${ROOT}/logs/holdout_mining/_round_${cat}.log" 2>&1 \
    || echo "[mining-holdout] WARN: round $cat exited nonzero"
  python3 - <<PYEOF
import json
d=json.load(open("$OUT_JSON"))
print("[mining-holdout] after $cat:", {k: len(v) for k, v in d["constraints"].items()})
PYEOF
done
echo "[mining-holdout] $(date '+%F %T') all rounds done"
