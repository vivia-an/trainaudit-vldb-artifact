#!/usr/bin/env bash
# Rewrite the workspace roots in the shipped scripts to a base you choose.
#
# Historical launch records retain the workspace roots used by their original
# runs. This utility rewrites those paths into a scratch copy without changing
# the archived records.
#
#   bash scripts/localize_paths.sh --base /my/workspace --out /tmp/localized
#   bash scripts/localize_paths.sh --base /my/workspace --in-place   # if you accept the edit
#
# Some historical launchers write experiment outputs below the base you give
# them. Point --base at a scratch workspace unless you explicitly want in-place
# paths.
#
# Under --base BASE, a path like /volume/qscai/lsk/Megatron-LM becomes BASE/Megatron-LM.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=""; OUT=""; INPLACE=0
while [ $# -gt 0 ]; do
  case $1 in
    --base) BASE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --in-place) INPLACE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$BASE" ] || { echo "need --base DIR" >&2; exit 2; }
[ -n "$OUT" ] || [ "$INPLACE" = 1 ] || { echo "need --out DIR or --in-place" >&2; exit 2; }

ROOTS=(/volume/qscai/lsk /volume/qscai/cqs/workspace/paper /volume/qscai/cqs/temp
       /volume/qscai/cqs /volume/posttrain/users/lsk/sdc/lsk /volume/pt-train/users)

targets=$(grep -rl "/volume/" --include='*.py' --include='*.sh' benchmark core 2>/dev/null || true)
[ -n "$targets" ] || { echo "nothing to localize"; exit 0; }

if [ "$INPLACE" = 1 ]; then
  dest=.
else
  mkdir -p "$OUT"
  for f in $targets; do mkdir -p "$OUT/$(dirname "$f")"; cp -f "$f" "$OUT/$f"; done
  dest=$OUT
fi

n=0
for f in $targets; do
  path="$dest/$f"
  for r in "${ROOTS[@]}"; do
    sed -i "s|${r}|${BASE}|g" "$path"
  done
  n=$((n+1))
done

left=$(grep -rho "/volume/[A-Za-z0-9_./-]*" $(for f in $targets; do echo "$dest/$f"; done) 2>/dev/null | sort -u | head -5 || true)
printf 'rewrote %d file(s) into %s\n' "$n" "$([ "$INPLACE" = 1 ] && echo 'the working tree' || echo "$OUT")"
if [ -n "$left" ]; then
  printf 'still referencing /volume (check these by hand):\n%s\n' "$left"
else
  printf 'no /volume references remain\n'
fi
