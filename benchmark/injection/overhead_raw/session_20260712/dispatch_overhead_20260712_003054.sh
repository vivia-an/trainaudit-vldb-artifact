#!/bin/bash
set -euo pipefail
cd /volume/posttrain/users/lsk/sdc/lsk/Megatron-LM
OUT=/volume/posttrain/users/lsk/sdc/lsk/Megatron-LM/logs_gpu_lsk32
mkdir -p "$OUT"
echo "[$(date -Is)] START overhead on $(hostname)" | tee "$OUT/overhead_dispatch.log"
# 1) baseline
echo "[$(date -Is)] === BASELINE ===" | tee -a "$OUT/overhead_dispatch.log"
DUMP=0 LOGDIR=/tmp/oh_off_lsk32 bash run_overhead_scaled.sh > "$OUT/overhead_baseline.log" 2>&1
echo "[$(date -Is)] baseline done" | tee -a "$OUT/overhead_dispatch.log"
# 2) FAST collect (paper 7.7x path)
echo "[$(date -Is)] === COLLECT FAST ===" | tee -a "$OUT/overhead_dispatch.log"
export VTIMELINE_FAST=1
DUMP=8 LOGDIR="$OUT/overhead_fast_db" bash run_overhead_scaled.sh > "$OUT/overhead_fast_collect.log" 2>&1
echo "[$(date -Is)] fast collect done" | tee -a "$OUT/overhead_dispatch.log"
# 3) FULL collect (sha256 path, for A/B)
echo "[$(date -Is)] === COLLECT FULL ===" | tee -a "$OUT/overhead_dispatch.log"
unset VTIMELINE_FAST
DUMP=8 LOGDIR="$OUT/overhead_full_db" bash run_overhead_scaled.sh > "$OUT/overhead_full_collect.log" 2>&1
echo "[$(date -Is)] DONE" | tee -a "$OUT/overhead_dispatch.log"
