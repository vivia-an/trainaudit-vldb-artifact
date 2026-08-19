#!/usr/bin/env bash
# Fetch and verify the trace databases behind the guard ablation (Sec 5.3).
#
# They are published as a release asset rather than committed: 117 DuckDB files,
# 212 MiB raw. Extraction reconstructs <run>/Collector/*.db under --dest.
#
#   bash scripts/fetch_trace_dbs.sh [--dest DIR] [--verify-only]
set -euo pipefail

REPO=${REPO:-vivia-an/trainaudit-vldb-artifact}
TAG=${TAG:-trace-dbs-v1}
ASSET=trainaudit-trace-dbs.tar.gz
DEST=$(pwd)/trace_dbs
VERIFY_ONLY=0
HERE=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST="$HERE/benchmark/injection/trace_db_manifest.csv"

while [ $# -gt 0 ]; do
  case $1 in
    --dest) DEST=$2; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

verify() {
  python3 - "$MANIFEST" "$DEST" <<'PY'
import csv, hashlib, pathlib, sys
manifest, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = list(csv.DictReader(manifest.open()))
missing = bad = 0
for r in rows:
    f = dest / r["path"]
    if not f.exists():
        missing += 1
        continue
    if hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256"]:
        print(f"  checksum mismatch: {r['path']}")
        bad += 1
runs = len({r["run"] for r in rows})
print(f"{len(rows) - missing - bad}/{len(rows)} files verified across {runs} runs"
      + (f", {missing} missing" if missing else "")
      + (f", {bad} corrupt" if bad else ""))
sys.exit(1 if (missing or bad) else 0)
PY
}

if [ "$VERIFY_ONLY" = 1 ]; then
  verify; exit $?
fi

mkdir -p "$DEST"
if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --dir "$DEST" --clobber
else
  curl -fL -o "$DEST/$ASSET" \
    "https://github.com/$REPO/releases/download/$TAG/$ASSET"
fi

tar -xzf "$DEST/$ASSET" -C "$DEST"
rm -f "$DEST/$ASSET"
verify

cat <<EOF

Extracted to $DEST

To re-run the guard ablation against them:
  export MEGATRON="$DEST"
  export SDCCHECK_ROOT=<a checkout with core/ on the path>
  bash core/ablation_scripts/run_d1_phase3.sh
Then compare the aggregate false positives against experiments/guard_ablation/d1_results.csv
(342 full / 429 without pi_precond / 598 without pi_topo).
EOF
