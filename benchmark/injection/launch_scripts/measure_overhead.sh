#!/bin/bash
cd /volume/posttrain/users/lsk/sdc/lsk/Megatron-LM
echo "[`date +%H:%M:%S`] === BASELINE (collector OFF) ==="
DUMP=0 LOGDIR=/tmp/oh_off bash run_overhead_scaled.sh > overhead_baseline.log 2>&1
echo "[`date +%H:%M:%S`] === COLLECT (collector ON, dump every step) ==="
DUMP=8 LOGDIR=/volume/posttrain/users/lsk/sdc/lsk/Megatron-LM/overhead_run_db bash run_overhead_scaled.sh > overhead_collect.log 2>&1
echo "[`date +%H:%M:%S`] === DONE ==="
