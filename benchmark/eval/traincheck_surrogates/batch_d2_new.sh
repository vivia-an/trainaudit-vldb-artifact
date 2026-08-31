#!/bin/bash
# Run TrainCheck pipeline (run_one.sh) on the 8 D2-new surrogates.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUGS=(ID1 CC1 PE1 AV1 TA1 SC1 CW1 LN1)
OUT="${1:-$HERE/batch_d2_new_results.txt}"
> "$OUT"
for b in "${BUGS[@]}"; do
    echo ">>> $b" | tee -a "$OUT"
    bash "$HERE/run_one.sh" "$b" 2>&1 | tee -a "$OUT"
done
echo "DONE batch d2_new; results in $OUT" | tee -a "$OUT"
