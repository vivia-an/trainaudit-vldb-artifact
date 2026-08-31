#!/bin/bash
# Run the 3 extra T1 surrogates (B2 / B3 / B8) that synthetic_17 covers
# but synthetic_14 missed. Same emission protocol as batch_t1.sh.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUGS=(B2 B3 B8)
OUT="${1:-$HERE/batch_t1_extra_results.txt}"
> "$OUT"
for b in "${BUGS[@]}"; do
    echo ">>> $b" | tee -a "$OUT"
    bash "$HERE/run_one.sh" "$b" 2>&1 | tee -a "$OUT"
done
echo "DONE batch t1 extra; results in $OUT" | tee -a "$OUT"
