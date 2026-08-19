#!/bin/bash
# Run B1 and B2 with TP=2, save trainaudit duckdb stores for offline π_topo
# ablation. Produces 8 files at $TRAINAUDIT_TRACE_DIR (default under this
# repo's benchmark/eval/_runs_gpu_ablation/):
#
#   B1_BUGGY_rank{0,1}.duckdb
#   B1_FIXED_rank{0,1}.duckdb
#   B2_BUGGY_rank{0,1}.duckdb
#   B2_FIXED_rank{0,1}.duckdb
#
# Required env:
#   MEGATRON_DIR — Megatron-LM repo root (must have both
#                  3c637fc0d{,~1} and 5fffdfc7{,~1} commits reachable)
#
# Optional env:
#   TRAINAUDIT_TRACE_DIR — output dir (default:
#       /volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025/benchmark/eval/_runs_gpu_ablation)
#   DATA_DIR — fake-data dir (default: /volume/qscai/cqs/temp/megatron-fake-data)
#
# Each run takes ~2-3 min on 2× A100. Total wallclock: ~15 min.
set -euo pipefail

if [ -z "${MEGATRON_DIR:-}" ]; then
    echo "ERROR: set MEGATRON_DIR to your Megatron-LM repo root" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export TRAINAUDIT_TRACE_DIR="${TRAINAUDIT_TRACE_DIR:-$REPO_ROOT/benchmark/eval/_runs_gpu_ablation}"
mkdir -p "$TRAINAUDIT_TRACE_DIR"

echo "=== GPU ablation: writing traces to $TRAINAUDIT_TRACE_DIR ==="
echo "=== MEGATRON_DIR = $MEGATRON_DIR ==="
nvidia-smi -L | head -4
echo

bash "$REPO_ROOT/benchmark/bugs/B1/trainaudit_run.sh"
echo
bash "$REPO_ROOT/benchmark/bugs/B2/trainaudit_run.sh"

echo
echo "=== Done. Produced files: ==="
ls -la "$TRAINAUDIT_TRACE_DIR"/*.duckdb 2>/dev/null
echo
echo "Next: run offline ablation against these traces from the paper directory."
