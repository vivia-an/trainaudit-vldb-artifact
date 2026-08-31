#!/usr/bin/env bash
# Phase-1 template supplement mining (A + B-light). Does NOT wipe the holdout lib.
# Run after / alongside category mining finishes writing; uses SDC_TARGET_TEMPLATE.
#
# Usage:
#   nohup ./scripts/ablation/run_mining_template_supplement.sh 30 \
#     > logs/holdout_mining/_template_supplement.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AGENTS="${ROOT}/agents"
OUT_JSON="${ROOT}/config/lib_holdout_mined.json"
PER_ROUND="${1:-30}"
# Default Phase-1 templates from template_mining_spec.json
TMPLS="${SDC_TEMPLATE_IDS:-P6 P9 P5-RNG COMM-PG}"

export SDC_HOLDOUT_EXCLUDE="${ROOT}/logs/holdout_mining/EXCLUSION_LIST.json"
export SDC_CONSTRAINTS_OUTPUT="$OUT_JSON"
export SDC_REASONING_EFFORT="${SDC_REASONING_EFFORT:-high}"
unset SDC_TARGET_CATEGORY 2>/dev/null || true

mkdir -p "${ROOT}/logs/holdout_mining"
cd "$AGENTS"

echo "[tmpl-supplement] $(date '+%F %T') templates=($TMPLS) per_round=$PER_ROUND out=$OUT_JSON"

# Seed static/B-light rules once (idempotent)
python3 "${ROOT}/scripts/ablation/seed_phase1_template_rules.py" "$OUT_JSON"

for tid in $TMPLS; do
  export SDC_TARGET_TEMPLATE="$tid"
  export SDC_MAX_ITERATIONS="$PER_ROUND"
  # map template → write category for env consumers that still read category
  case "$tid" in
    P6|P9|P5-RNG|COMM-PG) export SDC_TARGET_CATEGORY="model_integrity" ;;
    *) export SDC_TARGET_CATEGORY="model_integrity" ;;
  esac
  echo "[tmpl-supplement] $(date '+%F %T') === template: $tid (${PER_ROUND} iters) ==="
  python3 -u run_self_reason_with_writeAgent.py \
    >> "${ROOT}/logs/holdout_mining/_round_tmpl_${tid}.log" 2>&1 \
    || echo "[tmpl-supplement] WARN: template $tid exited nonzero"
  python3 - <<PYEOF
import json
d=json.load(open("$OUT_JSON"))
print("[tmpl-supplement] after $tid:", {k: len(v) for k, v in d["constraints"].items()})
PYEOF
done
unset SDC_TARGET_TEMPLATE
echo "[tmpl-supplement] $(date '+%F %T') phase1 templates done"
