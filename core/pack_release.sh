#!/usr/bin/env bash
# Pack self-contained TrainAudit core (agents + config + smoke + miner).
set -euo pipefail
CORE="$(cd "$(dirname "$0")" && pwd)"
SDC="$(cd "$CORE/.." && pwd)"
OUT="${1:-/tmp/trainaudit-core}"
rm -rf "$OUT"
mkdir -p "$OUT"

# top-level docs + entrypoints
for f in README.md LICENSE CITATION.cff SECURITY.md CONTRIBUTING.md \
         CODE_OF_CONDUCT.md requirements.txt .gitignore \
         run_miner.py run_smoke.py make_toy_trace.py example_verifier.py \
         topology_prune.py collector_stub.py pack_release.sh; do
  [[ -f "$CORE/$f" ]] && cp -a "$CORE/$f" "$OUT/"
done

# agents + config (full mining stack)
cp -a "$CORE/agents" "$OUT/agents"
cp -a "$CORE/config" "$OUT/config"

# data (csv + template_induction; no *.db)
mkdir -p "$OUT/data"
cp -a "$CORE/data/funnel_counts.csv" "$OUT/data/"
cp -a "$CORE/data/funnel_skip_l3_results.csv" "$OUT/data/"
[[ -f "$CORE/data/.gitignore" ]] && cp -a "$CORE/data/.gitignore" "$OUT/data/"
[[ -f "$CORE/data/.gitkeep" ]] && cp -a "$CORE/data/.gitkeep" "$OUT/data/"
if [[ -d "$CORE/data/template_induction" ]]; then
  cp -a "$CORE/data/template_induction" "$OUT/data/template_induction"
fi

# funnel script (prefer package-local copy)
mkdir -p "$OUT/scripts"
if [[ -f "$CORE/scripts/reproduce_funnel_counts.py" ]]; then
  cp -a "$CORE/scripts/reproduce_funnel_counts.py" "$OUT/scripts/"
else
  cp -a "$SDC/scripts/reproduce_funnel_counts.py" "$OUT/scripts/"
fi
# Already rooted on package parent; no rewrite required for core_algo/scripts copy.

# run_smoke / run_miner expect agents/ beside them — already flat
# patch run_smoke AGENTS path if it still says HERE/"agents" — OK

test -f "$OUT/agents/workflow_task_generate_with_writeAgent.py"
test -f "$OUT/run_miner.py"
test -f "$OUT/config/pattern_catalog_snapshot.json"

if grep -RInE 'api_key:[[:space:]]*sk-' "$OUT" --include='*.yaml' --include='*.yml' 2>/dev/null; then
  echo "REFUSE: secret api_key" >&2
  exit 2
fi
if grep -RInE 'C:/Users/|/home/[^/]+/' "$OUT" --include='*.py' 2>/dev/null | grep -v 'Megatron-LM/normal_db' ; then
  echo "REFUSE: machine-specific absolute path" >&2
  exit 2
fi
# drop caches from pack
find "$OUT" -type d -name '__pycache__' -prune -exec rm -rf {} +
echo "OK packed → $OUT"
echo "Smoke:  (cd $OUT && python3 run_smoke.py)"
echo "Miner:  (cd $OUT && SDC_PAPER_ALIGN=1 python3 run_miner.py)"
