#!/bin/bash
# Run TrainCheck on the 3 new D1' surrogates (CF1/CM1/OF1).
# Designed to run on eval-gpu-0 via SSH (matches batch_t0/t1.sh).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUGS=(CF1 CM1 OF1)
OUT="${1:-$HERE/batch_d1prime_new_results.txt}"
> "$OUT"
for b in "${BUGS[@]}"; do
    echo ">>> $b" | tee -a "$OUT"
    bash "$HERE/run_one.sh" "$b" 2>&1 | tee -a "$OUT"
done
echo "DONE batch d1prime_new; results in $OUT" | tee -a "$OUT"
