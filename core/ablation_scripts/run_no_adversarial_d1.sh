#!/usr/bin/env bash
# Leave-one-out row 4: lib_no_adversarial on D1 clean / key dbs.
# Phase-1 proxy: all applicable_conditions stripped (see make_ablation_libs.py).
#
# Usage:
#   ./scripts/ablation/run_no_adversarial_d1.sh [smoke|full]
#   smoke = 6 cells (5 clean normal + tp_router), full = 42 D1 test dbs

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import duckdb" 2>/dev/null; then
  echo "[error] $PYTHON missing duckdb — pip install duckdb or set PYTHON=..." >&2
  exit 1
fi
MEGATRON="${MEGATRON:-/volume/posttrain/users/lsk/sdc/lsk/Megatron-LM}"
OUT_DIR="${ROOT}/logs/ablation_no_adv"
LIB_PATH="${ROOT}/config/lib_no_adversarial.json"
MODE="${1:-smoke}"

PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-3600}"
BATCH_SIZE="${BATCH_SIZE:-6}"
PROVIDER="${PROVIDER:-deepseek}"
MASTER_LOG="${OUT_DIR}/_run_master.log"

mkdir -p "$OUT_DIR"
cd "$ROOT"

if [ ! -f "$LIB_PATH" ]; then
  echo "[build] missing $LIB_PATH — running make_ablation_libs.py"
  python3 scripts/ablation/make_ablation_libs.py
fi

declare -a JOBS=()

add_job() {
  local db_name="$1" dp="$2" tp="$3" pp="$4" merged="$5"
  JOBS+=("${db_name}|${dp}|${tp}|${pp}|${merged}")
}

if [ "$MODE" = "smoke" ]; then
  add_job dist_optimizer_normal 2 1 1 "${MEGATRON}/normal_db/dist_optimizer_normal/Collector/merged_coredump.db"
  add_job dp_normal 2 1 1 "${MEGATRON}/normal_db/dp_normal/Collector/merged_coredump.db"
  add_job dp_normal_db 2 1 1 "${MEGATRON}/dp_normal_db/Collector/merged_coredump.db"
  add_job mixed_precision_normal 2 1 1 "${MEGATRON}/normal_db/mixed_precision_normal/Collector/merged_coredump.db"
  add_job tp_normal 1 2 1 "${MEGATRON}/normal_db/tp_normal/Collector/merged_coredump.db"
  add_job tp_router_test_db 1 2 1 "${MEGATRON}/tp_router_test_db/Collector/merged_coredump.db"
else
  for db_dir in "${MEGATRON}"/*_test_db "${MEGATRON}"/normal_db/* "${MEGATRON}"/dp_normal_db; do
    [ -d "$db_dir" ] || continue
    db_name=$(basename "$db_dir")
    merged="${db_dir}/Collector/merged_coredump.db"
    [ -f "$merged" ] || continue
    sz=$(stat -c %s "$merged" 2>/dev/null || echo 0)
    [ "$sz" -lt 1024 ] && continue
    collector="${db_dir}/Collector"
    files=$(ls "$collector" 2>/dev/null | grep -E "^coredump_dp[0-9]+_tp[0-9]+_pp[0-9]+_cp[0-9]+\.db$" || true)
    if [ -n "$files" ]; then
      max_dp=$(echo "$files" | grep -oE "dp[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
      max_tp=$(echo "$files" | grep -oE "tp[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
      max_pp=$(echo "$files" | grep -oE "pp[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
      dp=$((max_dp + 1)); tp=$((max_tp + 1)); pp=$((max_pp + 1))
    else
      dp=2; tp=1; pp=1
    fi
    add_job "$db_name" "$dp" "$tp" "$pp" "$merged"
  done
fi

total=${#JOBS[@]}
{
  echo "===================================="
  echo "no-adversarial ablation ($MODE): $total cells"
  echo "LIB=$LIB_PATH OUT=$OUT_DIR batch=$BATCH_SIZE"
  echo "===================================="
} | tee -a "$MASTER_LOG"

run_cell() {
  local job="$1"
  IFS='|' read -r db_name dp tp pp merged <<<"$job"
  local tag="lib_no_adversarial__${db_name}"
  local reports="${OUT_DIR}/${tag}.json"
  local stdout_log="${OUT_DIR}/${tag}.stdout"
  if [ -f "$reports" ] && [ -s "$reports" ]; then
    n=$("$PYTHON" -c "import json; print(len(json.load(open('${reports}'))))" 2>/dev/null || echo 0)
    if [ "${n:-0}" -gt 5 ]; then
      echo "  [skip] ${tag} (${n} records)" | tee -a "$MASTER_LOG"
      return 0
    fi
  fi
  echo "  [launch] ${tag} DP=${dp} TP=${tp} PP=${pp}" | tee -a "$MASTER_LOG"
  timeout "$PER_RUN_TIMEOUT" "$PYTHON" -u -m sdccheck "$merged" \
    --dp "$dp" --tp "$tp" --pp "$pp" \
    --constraints-file "$LIB_PATH" \
    --provider "$PROVIDER" \
    --reports-out "$reports" \
    > "$stdout_log" 2>&1
  local rc=$?
  echo "  [done ] ${tag} rc=${rc}" | tee -a "$MASTER_LOG"
}

launched=0
for job in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$BATCH_SIZE" ]; do
    sleep 5
  done
  run_cell "$job" &
  launched=$((launched + 1))
done
wait

"$PYTHON" scripts/ablation/aggregate_fp.py \
  --reports-dir "$OUT_DIR" \
  --library-json "$LIB_PATH" \
  --csv-out "${OUT_DIR}/no_adversarial_results.csv" \
  | tee -a "$MASTER_LOG"

echo "=== finished ($MODE) — see ${OUT_DIR}/no_adversarial_results.csv ===" | tee -a "$MASTER_LOG"
