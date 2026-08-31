#!/bin/bash
# Run Naïve baseline on the 23 real bugs (13 verified + 10 hunt) on eval-gpu-0.
# Each bug attempt: ≤ 8min. Per-bug verdict written to stdout in contract form.
#
# Usage (on eval-gpu-0):
#   bash sweep_naive_real.sh > naive_real_results.txt 2>&1
set -uo pipefail
ROOT="/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025"
PY="/volume/qscai/cqs/temp/venv-cu126/bin/python"
PATH="/volume/qscai/cqs/temp/venv-cu126/bin:$PATH"
export PATH

export MEGATRON_DIR="$ROOT/exp/frameworks/Megatron-LM"
export DS_DIR="$ROOT/exp/frameworks/DeepSpeed"
export OLMO_DIR="$ROOT/exp/frameworks/OLMo"
export OLMO_CORE_DIR="$ROOT/exp/frameworks/OLMo-core"

# 13 Phase 1-2 verified bugs (doc 22 §2.1)
VERIFIED_BUGS=(
    B11 B12 B13 M-012 M-014 M-020 M-024 M-NEW-5
    O-002 O-005 O-NEW-1 OC-NEW-2 OC-NEW-3
)

run_verified() {
    local bug_id="$1"
    echo ">>> $bug_id (verified)"
    if [[ ! -f "$ROOT/benchmark/bugs/$bug_id/trainaudit_run.sh" ]]; then
        echo "[$bug_id] FAIL: naive:no trainaudit_run.sh in benchmark/bugs/$bug_id"
        return 0
    fi
    timeout 480 "$PY" "$ROOT/benchmark/eval/naive_subprocess.py" \
        --bug-id "$bug_id" \
        --bugs-root "$ROOT/benchmark/bugs" \
        --timeout-s 420 2>&1 | tail -3
}

# 10 Hunt phase candidates have their own drivers under
# benchmark/eval/hunt_log/<CAND_ID>/dynamic_confirm_e2e.py.
# Skip "structural" ones which don't run real training (no metric stream).
HUNT_REAL=(
    CAND_OLMOCORE_RNGCKPT
    CAND_OLMOCORE_EVAL_NOZEROGRAD
    CAND_DEEPSPEED_BF16_BOUNDARY_GRAD_LEAK
    CAND_DEEPSPEED_ZERO_OFFLOAD_MULTI_BACKWARD
    CAND_DEEPSPEED_WARMUPCOSINE_MULTIGROUP
    CAND_DEEPSPEED_BF16_ZERO0_DUAL_BUG
)
HUNT_STRUCTURAL=(
    CAND_OLMOCORE_FSDP_EXPERTS
    CAND_MEGATRON_CUDAGRAPH_BUFFER_CORRUPTION
    CAND_OLMO_CKPT_SAVE_OVERWRITE_DROP
    CAND_OLMO_ADAPTIVE_CLIP_EMA_RESET
)

run_hunt_real() {
    local cand_id="$1"
    echo ">>> $cand_id (hunt-real)"
    local driver="$ROOT/benchmark/eval/hunt_log/$cand_id/dynamic_confirm_e2e.py"
    if [[ ! -f "$driver" ]]; then
        echo "[$cand_id] FAIL: naive:no dynamic_confirm_e2e.py at hunt_log/$cand_id"
        return 0
    fi
    local out
    out="$(timeout 480 "$PY" "$driver" 2>&1)"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        local err
        err="$(echo "$out" | tail -3 | tr '\n' ' ' | head -c 200)"
        echo "[$cand_id] FAIL: naive:driver_rc=$rc; $err"
        return 0
    fi
    # Apply naive detector to driver stdout
    echo "$out" | "$PY" "$ROOT/benchmark/eval/naive_stdin_check.py" "$cand_id"
}

echo "===== Phase 1-2 verified (13 bugs) ====="
for b in "${VERIFIED_BUGS[@]}"; do
    run_verified "$b"
done

echo
echo "===== Hunt E2E confirmed (6 real-run candidates) ====="
for c in "${HUNT_REAL[@]}"; do
    run_hunt_real "$c"
done

echo
echo "===== Hunt structural (4 candidates, no metric stream) ====="
for c in "${HUNT_STRUCTURAL[@]}"; do
    echo "[$c] N/A: naive:structural-only candidate (AST + runtime emulation), no real training metric stream"
done

echo
echo "DONE sweep_naive_real"
