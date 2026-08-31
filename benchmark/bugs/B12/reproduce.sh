#!/bin/bash
# B12 / OLMo-core PR 27 reproduction: peak LR not set on restart.

set -euo pipefail
OLMO_CORE_DIR="${OLMO_CORE_DIR:?Set OLMO_CORE_DIR to OLMo-core repo root}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUGGY_COMMIT="6e330ba2~1"
FIXED_COMMIT="6e330ba2"

cd "$OLMO_CORE_DIR"
ORIG=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)
trap 'cd "$OLMO_CORE_DIR" && git checkout -q "$ORIG" 2>/dev/null || true' EXIT

run_one() {
    local label="$1" rev="$2"
    echo "===== $label ($rev) ====="
    cd "$OLMO_CORE_DIR"
    git checkout -q "$rev"
    set +e
    bash "$SCRIPT_DIR/run.sh"
    set -e
    echo
}

run_one "BUGGY" "$BUGGY_COMMIT"
run_one "FIXED" "$FIXED_COMMIT"
