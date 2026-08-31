#!/usr/bin/env bash
# Phase 3: launch all remaining buggy D1 dbs × 3 libs in batched parallel cells.
#
# - Detects per-db topology from the per-rank filenames (max DP/TP/PP rank index).
# - Skips (db, lib) cells that already have a non-empty reports JSON.
# - Limits to BATCH_SIZE concurrent cells; waits for the batch to drain before launching the next.
# - Excludes tp_router_test_db (already done in phase 1).

set -uo pipefail

ROOT="${SDCCHECK_ROOT:-/volume/posttrain/users/lsk/sdc/lsk/sdccheck}"
MEGATRON="${MEGATRON:-/volume/posttrain/users/lsk/sdc/lsk/Megatron-LM}"
OUT_DIR="${ROOT}/logs/ablation_d1"
mkdir -p "$OUT_DIR"

BATCH_SIZE="${BATCH_SIZE:-12}"      # max concurrent cells
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-3600}"
SKIP_DBS=("tp_router_test_db")  # already done in phase 1

cd "$ROOT"

LIBS=(
  "lib_full:config/lib_full.json"
  "lib_no_topo:config/lib_no_topo.json"
  "lib_no_precond:config/lib_no_precond.json"
)

# Discover dbs + topology
declare -a JOBS=()  # each entry: "<db_name>|<dp>|<tp>|<pp>|<lib_name>|<lib_path>"
for db_dir in "${MEGATRON}"/*_test_db; do
  [ -d "$db_dir" ] || continue
  db_name=$(basename "$db_dir")
  # skip
  for s in "${SKIP_DBS[@]}"; do [ "$db_name" = "$s" ] && continue 2; done
  collector="$db_dir/Collector"
  merged="$collector/merged_coredump.db"
  [ -f "$merged" ] || { echo "  [skip] $db_name no merged db"; continue; }
  sz=$(stat -c %s "$merged" 2>/dev/null || echo 0)
  if [ "$sz" -lt 1024 ]; then
    echo "  [skip] $db_name merged db too small ($sz bytes)"; continue
  fi
  files=$(ls "$collector" 2>/dev/null | grep -E "^coredump_dp[0-9]+_tp[0-9]+_pp[0-9]+_cp[0-9]+\.db$")
  [ -z "$files" ] && { echo "  [skip] $db_name no per-rank dbs"; continue; }
  max_dp=$(echo "$files" | grep -oE "dp[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
  max_tp=$(echo "$files" | grep -oE "tp[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
  max_pp=$(echo "$files" | grep -oE "pp[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
  dp=$((max_dp+1)); tp=$((max_tp+1)); pp=$((max_pp+1))
  for lib_spec in "${LIBS[@]}"; do
    lib_name="${lib_spec%%:*}"
    lib_path="${lib_spec##*:}"
    out_json="$OUT_DIR/${lib_name}__${db_name}.json"
    if [ -f "$out_json" ] && [ -s "$out_json" ]; then
      n=$(python3 -c "import json,sys
try:
    print(len(json.load(open('$out_json'))))
except Exception:
    print(0)" 2>/dev/null)
      if [ -n "$n" ] && [ "$n" -gt 5 ]; then
        echo "  [skip] $lib_name × $db_name (already have $n records)"; continue
      fi
    fi
    JOBS+=("$db_name|$dp|$tp|$pp|$lib_name|$lib_path")
  done
done

total=${#JOBS[@]}
echo
echo "===================================="
echo "Phase 3 plan: $total cells, batch=$BATCH_SIZE"
echo "===================================="

run_cell() {
  local job="$1"
  IFS='|' read -r db_name dp tp pp lib_name lib_path <<<"$job"
  local merged="${MEGATRON}/${db_name}/Collector/merged_coredump.db"
  local out_base="${OUT_DIR}/${lib_name}__${db_name}"
  echo "  [launch] ${lib_name} × ${db_name} (DP=$dp TP=$tp PP=$pp)"
  nohup timeout "$PER_RUN_TIMEOUT" python3 -u -m sdccheck "$merged" \
    --dp "$dp" --tp "$tp" --pp "$pp" \
    --constraints-file "$lib_path" \
    --provider deepseek \
    --reports-out "${out_base}.json" \
    > "${out_base}.stdout" 2>&1 &
}

# Drive batches
launched=0
batch_pids=()
for i in "${!JOBS[@]}"; do
  run_cell "${JOBS[$i]}" &
  batch_pids+=($!)
  launched=$((launched+1))

  if (( ${#batch_pids[@]} >= BATCH_SIZE )) || (( launched == total )); then
    echo "  [batch] waiting for ${#batch_pids[@]} cells to drain (${launched}/${total} launched)..."
    # Wait for ALL python sdccheck children, not the bash wrappers
    while pgrep -f "timeout ${PER_RUN_TIMEOUT} python3 -u -m sdccheck" > /dev/null; do
      sleep 30
    done
    batch_pids=()
    echo "  [batch] drained at $(date '+%H:%M:%S')"
  fi
done

echo
echo "===================================="
echo "Phase 3 done: $launched cells launched."
echo "Aggregate with: python3 scripts/ablation/aggregate_fp.py --reports-dir logs/ablation_d1 --library-json config/lib_full.json --csv-out logs/ablation_d1/d1_results.csv"
