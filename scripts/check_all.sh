#!/usr/bin/env bash
# Run every offline check in this artifact and summarise.
#
#   bash scripts/check_all.sh                 # checks that need no data download
#   bash scripts/check_all.sh --traces DIR    # also the checks that need the coredump bundle
#   bash scripts/check_all.sh --events DIR    # also the checks that need the events bundle
#
# Needs python3 with duckdb for the trace-dependent checks; everything else is stdlib
# plus pdftotext (poppler-utils) for the figure check.
set -uo pipefail
cd "$(dirname "$0")/.."

TRACES=""
EVENTS=""
while [ $# -gt 0 ]; do
  case $1 in
    --traces) TRACES=$2; shift 2 ;;
    --events) EVENTS=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

pass=0; fail=0
run() {                       # run <label> <command...>
  local label=$1; shift
  printf '\n=== %s\n' "$label"
  if "$@"; then pass=$((pass+1)); else fail=$((fail+1)); printf '    ^ FAILED\n'; fi
}

run "published numbers, recomputed from the shipped data" \
    python3 scripts/verify_paper_numbers.py -q
run "numbers printed inside the figures, against their data" \
    python3 scripts/verify_figures.py
run "Table tab:overhead, from the raw H20 logs" \
    python3 benchmark/injection/parse_overhead_logs.py --check
run "Real-SE per-case verdicts, from SMOKE_REPORT.md" \
    python3 benchmark/eval/real_sdc/extract_detection_csv.py
run "Real-SE replay outcomes, from the per-case smoke logs" \
    python3 benchmark/eval/real_sdc/extract_replay_outcomes.py
run "mining funnel counts" \
    python3 core/scripts/reproduce_funnel_counts.py
# the .sha256 names a bare filename, so it has to be checked from its own directory
# The scripts a reviewer runs first. Their outputs (toy .db files) are gitignored, so this
# leaves the tree clean.
run "entry-point scripts a reviewer runs first" \
    bash -c 'set -e
      python3 core/run_smoke.py            >/dev/null
      python3 core/make_toy_trace.py       >/dev/null
      python3 core/example_verifier.py     >/dev/null
      python3 core/topology_prune.py       >/dev/null
      python3 core/collector_stub.py       >/dev/null
      echo "  run_smoke, make_toy_trace, example_verifier, topology_prune, collector_stub: all OK"'
run "guard ablation, recomputed without the duplicated-rank databases" \
    python3 benchmark/injection/recompute_ablation_clean.py
run "temporal pattern-coverage holdout" \
    python3 benchmark/eval/temporal_pattern_holdout.py
run "predicate necessity across the 392-record corpus" \
    python3 benchmark/eval/predicate_necessity.py
run "hookpoint vocabularies: paper vs corpus" \
    python3 benchmark/eval/hookpoint_coverage.py
run "the corpus's own declared coding uncertainty" \
    python3 benchmark/eval/coding_uncertainty.py
run "is the SQL barrier the property shape or the guard?" \
    python3 benchmark/eval/expressibility_vs_guard.py
run "what the diagnosis files do and do not contain" \
    python3 benchmark/eval/diagnosis_data_audit.py
# The shipped run scripts carry the absolute paths of the machines they ran on. They are left
# that way on purpose; localize_paths.sh rewrites them into a copy. Check it still covers them.
run "the trainaudit package imports and its tests pass" \
    bash -c 'cd core/trainaudit_pkg && PYTHONPATH=. python3 -m pytest tests -q 2>&1 | tail -1'
run "shipped scripts can be localized to another workspace" \
    bash -c 'd=$(mktemp -d); bash scripts/localize_paths.sh --base /tmp/ws --out "$d" \
             | tail -2; rm -rf "$d"'

run "main.tex's appendix references all resolve" \
    python3 paper/check_appendix_refs.py
# the .sha256 names a bare filename, so it has to be checked from its own directory
run "frozen Pattern Catalog integrity" \
    bash -c 'cd core/config && sha256sum -c frozen_template_catalog.sha256' 

if [ -n "$TRACES" ]; then
  run "per-rank captures genuinely distinct" \
      python3 benchmark/injection/audit_rank_captures.py --root "$TRACES"
  run "trace bundle integrity" \
      bash scripts/fetch_trace_dbs.sh --dest "$TRACES" --verify-only
  for d in normal_db/dp_normal tp_router_test_db; do
    [ -f "$TRACES/$d/Collector/merged_coredump.db" ] || continue
    run "compiled SQL executes on $d" \
        python3 core/validate_generated_sql.py --db "$TRACES/$d/Collector/merged_coredump.db"
  done
else
  printf '\n(skipped the coredump-trace checks; pass --traces DIR after '
  printf 'scripts/fetch_trace_dbs.sh)\n'
fi

if [ -n "$EVENTS" ]; then
  run "events-bundle integrity" \
      env TAG=trace-events-v2 bash scripts/fetch_trace_dbs.sh --dest "$EVENTS" --verify-only
  run "the guard's skip-vs-check split is exercised" \
      python3 benchmark/injection/audit_guard_groups.py --root "$EVENTS"
else
  printf '\n(skipped the events-trace checks; pass --events DIR after '
  printf 'TAG=trace-events-v2 scripts/fetch_trace_dbs.sh)\n'
fi

printf '\n%s\n' "----------------------------------------"
printf '%d check group(s) passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
