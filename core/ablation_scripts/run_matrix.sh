#!/usr/bin/env bash
# Run the leave-one-out ablation matrix:
#   3 libraries (full / no_topo / no_precond) × N databases
#
# Each cell writes its per-constraint reports to:
#   logs/ablation/<lib_name>__<db_basename>.json
# plus a stdout transcript to:
#   logs/ablation/<lib_name>__<db_basename>.stdout
#
# Usage: ./scripts/ablation/run_matrix.sh [TIMEOUT_SEC_PER_RUN]
# Default per-run timeout: 1800s (30 min).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="logs/ablation"
mkdir -p "$OUT_DIR"

PER_RUN_TIMEOUT="${1:-1800}"

LIBS=(
    "lib_full:config/lib_full.json"
    "lib_no_topo:config/lib_no_topo.json"
    "lib_no_precond:config/lib_no_precond.json"
)

# (db_path, dp, tp, pp) tuples
DBS=(
    "data/merged_coredump_1.db:2:2:1"
    "data/merged_coredump_4.db:2:2:1"
    "data/merged_coredump_5.db:2:2:1"
    "data/merged_coredump_dp2tp2pp2.db:2:2:2"
)

PROVIDER="${PROVIDER:-deepseek}"

echo "=== Ablation matrix run ==="
echo "ROOT=$ROOT"
echo "PROVIDER=$PROVIDER"
echo "PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT}s"
echo "OUT_DIR=$OUT_DIR"
echo

total=0
ok=0
fail=0
for lib in "${LIBS[@]}"; do
    lib_name="${lib%%:*}"
    lib_path="${lib##*:}"
    if [ ! -f "$lib_path" ]; then
        echo "SKIP $lib_name (missing $lib_path)"
        continue
    fi
    for dbe in "${DBS[@]}"; do
        IFS=':' read -r db dp tp pp <<<"$dbe"
        if [ ! -f "$db" ]; then
            echo "SKIP db $db not found"
            continue
        fi
        db_base="$(basename "$db" .db)"
        tag="${lib_name}__${db_base}"
        reports="${OUT_DIR}/${tag}.json"
        stdout_log="${OUT_DIR}/${tag}.stdout"

        if [ -f "$reports" ] && [ -s "$reports" ]; then
            echo "[skip] ${tag} (reports already exist at ${reports})"
            continue
        fi

        echo "[run ] ${tag}  (dp=$dp tp=$tp pp=$pp)"
        total=$((total+1))
        timeout "$PER_RUN_TIMEOUT" python3 -m sdccheck \
            "$db" --dp "$dp" --tp "$tp" --pp "$pp" \
            --constraints-file "$lib_path" \
            --provider "$PROVIDER" \
            --reports-out "$reports" \
            > "$stdout_log" 2>&1
        rc=$?
        if [ $rc -eq 0 ] && [ -s "$reports" ]; then
            ok=$((ok+1))
            n_pass=$(python3 -c "import json,sys; d=json.load(open('$reports')); print(sum(1 for r in d if r['status']=='pass'))" 2>/dev/null || echo "?")
            n_fail=$(python3 -c "import json,sys; d=json.load(open('$reports')); print(sum(1 for r in d if r['status']=='fail'))" 2>/dev/null || echo "?")
            n_err=$(python3 -c "import json,sys; d=json.load(open('$reports')); print(sum(1 for r in d if r['status']=='error'))" 2>/dev/null || echo "?")
            n_total=$(python3 -c "import json,sys; d=json.load(open('$reports')); print(len(d))" 2>/dev/null || echo "?")
            echo "       ok rc=$rc  total=$n_total pass=$n_pass fail=$n_fail err=$n_err"
        else
            fail=$((fail+1))
            echo "       FAIL rc=$rc  (last lines of $stdout_log:)"
            tail -5 "$stdout_log" | sed 's/^/         > /'
        fi
    done
done

echo
echo "=== summary ==="
echo "total=$total ok=$ok fail=$fail"
