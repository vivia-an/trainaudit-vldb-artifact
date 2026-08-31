#!/bin/bash
# B11 / DeepSpeed PR 5150 reproduction.
# Checks both buggy (005afe12~1) and fixed (005afe12) revisions.

set -euo pipefail
DS_DIR="${DS_DIR:?Set DS_DIR to DeepSpeed repo root}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUGGY_COMMIT="005afe12~1"
FIXED_COMMIT="005afe12"

cd "$DS_DIR"
ORIG_HEAD=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD)

cleanup() {
    cd "$DS_DIR"
    git checkout -q "$ORIG_HEAD" 2>/dev/null || true
}
trap cleanup EXIT

run_one() {
    local label="$1" rev="$2"
    echo "===== $label ($rev) ====="
    cd "$DS_DIR"
    git checkout -q "$rev"
    set +e
    bash "$SCRIPT_DIR/run.sh"
    set -e
    echo
}

run_one "BUGGY" "$BUGGY_COMMIT"
run_one "FIXED" "$FIXED_COMMIT"
