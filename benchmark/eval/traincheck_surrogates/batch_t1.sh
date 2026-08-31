#!/bin/bash
# Run all 7 T1-tier surrogates through TrainCheck and emit one contract line per bug.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUGS=(B1 B13 M-012 M-NEW-5 M-024 M-020 OC-NEW-3)
OUT="${1:-$HERE/batch_t1_results.txt}"
> "$OUT"
for b in "${BUGS[@]}"; do
    echo ">>> $b" | tee -a "$OUT"
    bash "$HERE/run_one.sh" "$b" 2>&1 | tee -a "$OUT"
done
echo "DONE batch t1; results in $OUT" | tee -a "$OUT"
