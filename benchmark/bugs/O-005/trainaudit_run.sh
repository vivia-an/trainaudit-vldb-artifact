#!/bin/bash
set -euo pipefail
OLMO_DIR="${OLMO_DIR:?Set OLMO_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUGGY_COMMIT="0bc7f6c7"
FIXED_COMMIT="204ad53c"

cd "$OLMO_DIR"
ORIG=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)
trap 'cd "$OLMO_DIR" && git checkout -q "$ORIG" 2>/dev/null || true' EXIT

run_one() {
    local label="$1" rev="$2"
    echo "===== TrainAudit on $label ($rev) ====="
    cd "$OLMO_DIR"
    git checkout -q "$rev"
    set +e
    OLMO_DIR="$OLMO_DIR" PYTHONPATH="$OLMO_DIR" python3 "$SCRIPT_DIR/trainaudit_driver.py" 2>&1
    set -e
    echo
}

run_one "BUGGY" "$BUGGY_COMMIT"
run_one "FIXED" "$FIXED_COMMIT"
