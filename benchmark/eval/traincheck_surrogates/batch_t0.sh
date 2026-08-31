#!/bin/bash
# Run all 7 T0-tier surrogates through TrainCheck and emit one contract line per bug.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUGS=(B11 B12 M-014 O-NEW-1 OC-NEW-2 O-005 O-NEW-9)
OUT="${1:-$HERE/batch_t0_results.txt}"
> "$OUT"
for b in "${BUGS[@]}"; do
    echo ">>> $b" | tee -a "$OUT"
    bash "$HERE/run_one.sh" "$b" 2>&1 | tee -a "$OUT"
done
echo "DONE batch t0; results in $OUT" | tee -a "$OUT"
