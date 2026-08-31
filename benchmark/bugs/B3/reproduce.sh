#!/bin/bash
set -euo pipefail
DS_DIR="${DS_DIR:?Set DS_DIR to DeepSpeed repo root}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUGGY_COMMIT="d56268f3~1"
FIXED_COMMIT="d56268f3"

cd "$DS_DIR"
ORIG=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)
trap 'cd "$DS_DIR" && git checkout -q "$ORIG" 2>/dev/null || true' EXIT

run_one() {
    local label="$1" rev="$2"
    echo "===== $label ($rev) ====="
    cd "$DS_DIR"
    git reset --hard "$rev" 2>&1 | tail -2
    git clean -fdq -- deepspeed/ 2>/dev/null || true
    find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    set +e
    bash "$SCRIPT_DIR/run.sh"
    set -e
    echo
}

run_one "BUGGY" "$BUGGY_COMMIT"
run_one "FIXED" "$FIXED_COMMIT"
