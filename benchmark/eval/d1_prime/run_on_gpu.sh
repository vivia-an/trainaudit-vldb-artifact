#!/bin/bash
# One-click runner: ship D1' new surrogates (CF1/CM1/OF1) to a GPU host via SSH,
# run all 6 _traincheck_*.py adapters under the project venv, and write traces
# to the shared GPFS path so they're visible locally without rsync.
#
# Usage:
#   bash run_on_gpu.sh                    # all 3 surrogates × buggy/fixed = 6 runs
#   bash run_on_gpu.sh CF1                # just CF1 buggy + fixed
#   bash run_on_gpu.sh CF1 OF1            # CF1 + OF1
#   HOST=beijing-olmo-gpu bash run_on_gpu.sh   # use a different SSH alias
#
# Environment overrides:
#   HOST              SSH alias (default: eval-gpu-0)
#   VENV              path to GPU venv with traincheck installed
#                     (default: /volume/qscai/cqs/temp/venv-cu126)
#   TRACE_ROOT        where on shared FS to write traces
#                     (default: /volume/qscai/cqs/temp/d1_prime_traces)
#   PROJECT_ROOT      paper repo root (default: this repo)
#   PER_RUN_TIMEOUT   seconds, default 300
#
# Outputs:
#   $TRACE_ROOT/<BUG>/<VARIANT>/{instrumentation_*.log, proxy_log.json}
#   results/run_on_gpu.log    (consolidated stdout/stderr from all runs)
#   results/run_on_gpu.summary.json    (per-(bug,variant) status + trace size)

set -euo pipefail

HOST="${HOST:-eval-gpu-0}"
VENV="${VENV:-/volume/qscai/cqs/temp/venv-cu126}"
TRACE_ROOT="${TRACE_ROOT:-/volume/qscai/cqs/temp/d1_prime_traces}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-300}"
PROJECT_ROOT="${PROJECT_ROOT:-/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR" "$TRACE_ROOT"

# Default to all 3 surrogates if no args
if [[ $# -eq 0 ]]; then
    BUGS=(CF1 CM1 OF1)
else
    BUGS=("$@")
fi

# Validate
for b in "${BUGS[@]}"; do
    case "$b" in
        CF1|CM1|OF1) ;;
        *) echo "ERROR: unknown surrogate '$b' (expected CF1 / CM1 / OF1)"; exit 2;;
    esac
done

LOGFILE="$RESULTS_DIR/run_on_gpu.log"
SUMMARYFILE="$RESULTS_DIR/run_on_gpu.summary.json"
echo "=== run_on_gpu.sh @ $(date) ===" | tee "$LOGFILE"
echo "host=$HOST  venv=$VENV  trace_root=$TRACE_ROOT" | tee -a "$LOGFILE"
echo "bugs=${BUGS[*]}" | tee -a "$LOGFILE"

# Quick connectivity probe
echo "" | tee -a "$LOGFILE"
echo "--- probing $HOST ---" | tee -a "$LOGFILE"
if ! ssh -o ConnectTimeout=10 "$HOST" "bash -l -c 'nvidia-smi -L | head -2'" 2>&1 | tee -a "$LOGFILE"; then
    echo "ERROR: SSH probe failed. Check ~/.ssh/config and key." | tee -a "$LOGFILE"
    exit 3
fi

# Build a remote script that runs all (bug,variant) pairs sequentially.
# Doing it in one SSH call avoids re-establishing connection 6× and re-loading the env.
REMOTE_SCRIPT="/volume/qscai/cqs/temp/run_d1prime_${$}.sh"
cat > "$REMOTE_SCRIPT" <<EOF
#!/bin/bash
set -uo pipefail   # NOT -e: keep going even if one bug fails
source /etc/shinit_v2 2>/dev/null || true
source "$VENV/bin/activate" 2>/dev/null || {
    echo "WARNING: venv at $VENV not found; falling back to system python"
}
python3 -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
echo "================================================="

cd "$PROJECT_ROOT"

EOF

declare -a SUMMARY_ROWS

for bug in "${BUGS[@]}"; do
    for variant in buggy fixed; do
        TDIR="$TRACE_ROOT/${bug}/${variant}"
        cat >> "$REMOTE_SCRIPT" <<EOF
echo ""
echo "===== $bug / $variant ====="
mkdir -p "$TDIR"
ML_DAIKON_OUTPUT_DIR="$TDIR" timeout $PER_RUN_TIMEOUT python3 \
    "$PROJECT_ROOT/benchmark/eval/d1_prime/_traincheck_${bug}_${variant}.py" \
    2>&1 | tail -30
echo "RC=\$?"
echo "trace files:"
ls -la "$TDIR" | tail -5

EOF
    done
done

chmod +x "$REMOTE_SCRIPT"

# Run remote
echo "" | tee -a "$LOGFILE"
echo "--- launching remote run on $HOST ---" | tee -a "$LOGFILE"
echo "remote script: $REMOTE_SCRIPT" | tee -a "$LOGFILE"
ssh "$HOST" "bash -l $REMOTE_SCRIPT" 2>&1 | tee -a "$LOGFILE"

# Local-side summary (read from shared FS)
echo "" | tee -a "$LOGFILE"
echo "--- summary ---" | tee -a "$LOGFILE"

python3 - <<PY | tee -a "$LOGFILE"
import json, os
from pathlib import Path

trace_root = Path("$TRACE_ROOT")
bugs = "${BUGS[*]}".split()
out = {"host": "$HOST", "venv": "$VENV", "trace_root": str(trace_root), "runs": []}
for b in bugs:
    for v in ["buggy", "fixed"]:
        d = trace_root / b / v
        files = list(d.glob("*")) if d.exists() else []
        size = sum(f.stat().st_size for f in files if f.is_file()) if files else 0
        instr = [f for f in files if f.name.startswith("instrumentation_")]
        proxy = [f for f in files if f.name == "proxy_log.json"]
        ok = bool(instr) and size > 1000  # heuristic: instrumentation log written
        out["runs"].append({
            "bug": b, "variant": v, "trace_dir": str(d),
            "n_files": len(files), "total_bytes": size,
            "has_instrumentation_log": bool(instr),
            "has_proxy_log": bool(proxy),
            "looks_ok": ok,
        })
        print(f"  {b}/{v:>5}: {len(files)} files, {size:,} bytes  -> {'OK' if ok else 'EMPTY/FAIL'}")

Path("$SUMMARYFILE").write_text(json.dumps(out, indent=2))
print()
print(f"Summary JSON: $SUMMARYFILE")
print(f"Full log    : $LOGFILE")
PY

# Cleanup remote staging script
rm -f "$REMOTE_SCRIPT"
