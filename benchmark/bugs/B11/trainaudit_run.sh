#!/bin/bash
# Run B11 with TrainAudit T0 against both BUGGY and FIXED commits.
set -euo pipefail
DS_DIR="${DS_DIR:?Set DS_DIR to DeepSpeed repo root}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUGGY_COMMIT="005afe12~1"
FIXED_COMMIT="005afe12"

cd "$DS_DIR"
ORIG=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)
trap 'cd "$DS_DIR" && git checkout -q "$ORIG" 2>/dev/null || true' EXIT

run_one() {
    local label="$1" rev="$2"
    echo "===== TrainAudit on $label ($rev) ====="
    cd "$DS_DIR"
    git checkout -q "$rev"
    set +e
    torchrun --nproc_per_node=1 --master_port=$((29560 + RANDOM % 1000)) \
        "$SCRIPT_DIR/trainaudit_driver.py" 2>&1
    set -e
    echo
}

run_one "BUGGY" "$BUGGY_COMMIT"
run_one "FIXED" "$FIXED_COMMIT"
