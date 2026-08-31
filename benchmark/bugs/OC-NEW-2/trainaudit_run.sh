#!/bin/bash
set -euo pipefail
OLMO_CORE_DIR="${OLMO_CORE_DIR:?Set OLMO_CORE_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUGGY_COMMIT="2b6cf996~1"
FIXED_COMMIT="2b6cf996"

cd "$OLMO_CORE_DIR"
ORIG=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)
trap 'cd "$OLMO_CORE_DIR" && git checkout -q "$ORIG" 2>/dev/null || true' EXIT

run_one() {
    local label="$1" rev="$2"
    echo "===== TrainAudit T0 on $label ($rev) ====="
    cd "$OLMO_CORE_DIR"
    git checkout -q "$rev"
    set +e
    OLMO_CORE_DIR="$OLMO_CORE_DIR" python3 "$SCRIPT_DIR/trainaudit_driver.py" 2>&1
    set -e
    echo
}

run_one "BUGGY" "$BUGGY_COMMIT"
run_one "FIXED" "$FIXED_COMMIT"
