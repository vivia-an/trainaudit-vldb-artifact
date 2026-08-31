#!/usr/bin/env bash
# Quick liveness/progress dump for the ablation runs.
# Reads the latest stdout files in logs/ablation/ and the partial JSON
# reports written by the incremental-dump hook.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== running python3 -m sdccheck processes ==="
ps -ef | grep -E "python3? -m sdccheck" | grep -v grep || echo "  (none)"
echo

echo "=== reports JSON (size + n records) ==="
for f in logs/ablation/*.json; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    counts=$(python3 -c "
import json
def norm(s):
    return str(s).split('.')[-1].lower()
try:
    d=json.load(open('$f'))
    n=len(d)
    p=sum(1 for r in d if norm(r['status'])=='pass')
    fa=sum(1 for r in d if norm(r['status'])=='fail')
    e=sum(1 for r in d if norm(r['status'])=='error')
    print(f'{n} {p} {fa} {e}')
except Exception:
    print('? ? ? ?')")
    n=$(echo "$counts" | awk '{print $1}')
    p=$(echo "$counts" | awk '{print $2}')
    fa=$(echo "$counts" | awk '{print $3}')
    e=$(echo "$counts" | awk '{print $4}')
    sz=$(stat -c %s "$f" 2>/dev/null)
    printf "  %-60s sz=%6dB  total=%4s pass=%4s fail=%4s err=%4s\n" "$base" "$sz" "$n" "$p" "$fa" "$e"
done
echo

echo "=== latest stdout tails (last 'progress' line per file) ==="
for f in logs/ablation/*.stdout; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    line=$(grep -E "检查约束 [0-9]+/[0-9]+:" "$f" 2>/dev/null | tail -1)
    completion=$(grep -E "reports written|检查完成" "$f" 2>/dev/null | tail -1)
    if [ -n "$completion" ]; then
        printf "  [DONE] %-50s %s\n" "$base" "$completion"
    elif [ -n "$line" ]; then
        printf "  [run ] %-50s %s\n" "$base" "$line"
    else
        printf "  [    ] %-50s (no progress yet)\n" "$base"
    fi
done
