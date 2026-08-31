#!/usr/bin/env bash
# Stable, offline release checks for the publicly supported artifact paths.
set -uo pipefail
cd "$(dirname "$0")/.."

passed=0
failed=0

run() {
  local label=$1
  shift
  local log
  log=$(mktemp)
  if "$@" >"$log" 2>&1; then
    passed=$((passed + 1))
    printf '[OK]   %s\n' "$label"
  else
    failed=$((failed + 1))
    printf '[FAIL] %s\n' "$label"
    sed -n '1,240p' "$log"
  fi
  rm -f "$log"
}

run "pipeline smoke" python3 core/run_smoke.py
run "Real-SE detection records" python3 benchmark/eval/real_sdc/extract_detection_csv.py
run "Real-SE paired replay records" python3 benchmark/eval/real_sdc/extract_replay_outcomes.py
run "matched Catalog ablation" python3 benchmark/eval/paper_v2/verify_catalog_direct_ablation.py
run "catalog generalization" python3 benchmark/eval/verify_catalog_generalization.py --check
run "corpus construction" python3 benchmark/eval/verify_corpus_construction.py --check
run "annotation agreement" python3 benchmark/eval/verify_irr.py --check
run "taxonomy table" python3 benchmark/eval/verify_taxonomy_table.py --check
run "appendix per-case table" python3 benchmark/eval/verify_appendix_detection_table.py --check
run "collector microbenchmark" python3 benchmark/injection/parse_overhead_logs.py --check
run "schema coverage" python3 benchmark/eval/verify_tier_coverage_axis.py --check
run "frozen catalog checksum" bash -c 'cd core/config && sha256sum -c frozen_template_catalog.sha256'
run "release manifests" python3 scripts/check_release_manifests.py
run "paper citations" python3 scripts/check_citations.py
run "public release hygiene" python3 scripts/check_public_release.py
run "release dependencies" python3 -c \
  'import duckdb, numpy, pydantic, pytest, torch, yaml'
run "TrainAudit package tests" bash -c \
  'cd core/trainaudit_pkg && python3 -m pytest -q --no-header'
run "generated records remain clean" git diff --exit-code

printf '\nRelease checks: %d passed, %d failed\n' "$passed" "$failed"
test "$failed" -eq 0
