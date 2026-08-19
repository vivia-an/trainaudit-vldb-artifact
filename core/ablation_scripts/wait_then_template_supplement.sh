#!/usr/bin/env bash
# Wait until category holdout driver exits, then start Phase-1 template supplement.
# Does NOT touch lib while run_mining_holdout.sh is still writing.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="${ROOT}/logs/holdout_mining/_template_waiter.log"
LOCK="${ROOT}/logs/holdout_mining/_template_waiter.lock"
mkdir -p "${ROOT}/logs/holdout_mining"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[waiter] $(date '+%F %T') another waiter holds $LOCK; exit" | tee -a "$LOG"
  exit 0
fi

echo "[waiter] $(date '+%F %T') waiting for run_mining_holdout.sh to finish (lib untouched)" | tee -a "$LOG"
while pgrep -f 'scripts/ablation/run_mining_holdout.sh' >/dev/null 2>&1; do
  sleep 60
done
echo "[waiter] $(date '+%F %T') category driver gone; starting template supplement" | tee -a "$LOG"
nohup bash "${ROOT}/scripts/ablation/run_mining_template_supplement.sh" 30 \
  >> "${ROOT}/logs/holdout_mining/_template_supplement.log" 2>&1 &
echo "[waiter] tmpl_driver=$!" | tee -a "$LOG"
# hold flock until supplement starts; lock released on exit
