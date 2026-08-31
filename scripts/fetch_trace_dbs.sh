#!/usr/bin/env bash
# Fetch and verify the trace databases behind the guard ablation (Sec 5.3).
#
# They are published as a release asset rather than committed. Extraction reconstructs
# <run>/Collector/ under --dest, including the .db.wal sidecars: DuckDB keeps uncommitted
# data there, and for 7 of the 43 runs the .db file alone is a 12 KB header with no rows.
# (v1 of the bundle omitted the sidecars; the merged databases are self-contained, so the
# guard ablation was unaffected, but per-rank data for those runs was not usable.)
#
#   bash scripts/fetch_trace_dbs.sh [--dest DIR] [--verify-only]
set -euo pipefail

REPO=${REPO:-vivia-an/trainaudit-vldb-artifact}
TAG=${TAG:-trace-dbs-v2}
ASSET=${ASSET:-trainaudit-trace-dbs.tar.gz}
# Honoured from the environment like REPO/TAG/ASSET/MANIFEST above, or via --dest.
# It was previously positional-only, so `DEST=events_dbs bash scripts/fetch_trace_dbs.sh`
# silently extracted into ./trace_dbs -- mixing the two schemas in one directory.
DEST=${DEST:-$(pwd)/trace_dbs}
VERIFY_ONLY=0
HERE=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST=${MANIFEST:-$HERE/benchmark/injection/trace_db_manifest.csv}
# The events-schema bundle is a separate release with its own manifest:
#   TAG=trace-events-v2 ASSET=trainaudit-events-traces.tar.gz \
#   MANIFEST=$HERE/benchmark/injection/events_trace_manifest.csv \
#     bash scripts/fetch_trace_dbs.sh --dest DIR
case "$TAG" in trace-events-*)
  ASSET=${ASSET:-trainaudit-events-traces.tar.gz}
  [ "$ASSET" = "trainaudit-trace-dbs.tar.gz" ] && ASSET=trainaudit-events-traces.tar.gz
  case "$MANIFEST" in *trace_db_manifest.csv)
    MANIFEST="$HERE/benchmark/injection/events_trace_manifest.csv" ;; esac
  ;;
esac

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

# Refuse to mix the two schemas in one directory. The coredump bundle lays down
# <run>/Collector/*.db (+ .wal sidecars); the events bundle lays down *.duckdb. Extracting
# one over the other leaves a directory that satisfies neither manifest and produces
# confusing downstream failures rather than an obvious one.
if [ -d "$DEST" ]; then
  case "$TAG" in
    trace-events-*) other=$(find "$DEST" -name '*.db' -print -quit 2>/dev/null); want="events"; has="coredump" ;;
    *)              other=$(find "$DEST" -name '*.duckdb' -print -quit 2>/dev/null); want="coredump"; has="events" ;;
  esac
  if [ -n "$other" ]; then
    echo "refusing to extract the $want bundle into $DEST: it already holds $has-schema traces" >&2
    echo "  found: $other" >&2
    echo "  use a separate --dest (or DEST=) for each bundle" >&2
    exit 3
  fi
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

if [ "$TAG" = "trace-events-v2" ] || [ "$TAG" = "trace-events-v1" ]; then
  cat <<EOF

Extracted to $DEST

These are events-schema traces: events(event_id, step, rank, hookpoint, ts_ns,
schema_version, payload). The guard ablation does NOT read these — it reads the
coredump-schema bundle (TAG=trace-dbs-v2).

The build.snapshot payloads carry the per-parameter cross-rank records used by
the topology-aware verifier.
EOF
else
  cat <<EOF

Extracted and checksum-verified at $DEST.

The coredump-schema bundle can be used with the compiled SQL examples in
core/config/generated_sql.json. Generating SQL for new constraints requires a
configured LLM provider; recorded SQL can be executed offline.
EOF
fi
