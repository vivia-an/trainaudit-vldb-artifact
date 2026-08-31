#!/usr/bin/env bash
# Install the CPU-compatible environment used by the public release checks.
set -euo pipefail
cd "$(dirname "$0")/.."

trainaudit_python=${TRAINAUDIT_PYTHON:-python3}

"$trainaudit_python" -m pip install --upgrade pip
"$trainaudit_python" -m pip install -r core/requirements-tested.txt

if ! "$trainaudit_python" -c 'import torch' >/dev/null 2>&1; then
  "$trainaudit_python" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1
fi

"$trainaudit_python" -m pip install --no-deps -e core/trainaudit_pkg
