#!/usr/bin/env bash
# Assemble the PVLDB submission artifact from the local research workspace.
# Idempotent: safe to re-run; overwrites copied files, never deletes hand-written docs.
set -euo pipefail

WS=${WS:-/volume/posttrain/users/lsk/sdc/lsk}
PAPER=$WS/overleaf/sdc_llm_icml_2025          # VLDB submission source (main @ "Format paper for VLDB 2027 submission")
WORK=$WS/sdc_llm_icml_2025                    # older SIGMOD checkout, holds real_sdc/ + provenance audit
OUT=${OUT:-$WS/trainaudit-vldb-artifact}

mkdir -p "$OUT"/{paper,core,collector,benchmark,baselines,figures,docs,scripts,experiments}

say() { printf '  %-58s %s\n' "$1" "$2"; }

# copytree SRC DST [extra tar --exclude patterns...]
# rsync is not available on this box; tar preserves structure and honours excludes.
copytree() {
  local src=$1 dst=$2; shift 2
  [ -d "$src" ] || return 0
  mkdir -p "$dst"
  ( cd "$src" && tar cf - \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=.venv \
      --exclude='*.egg-info' "$@" . ) | ( cd "$dst" && tar xf - )
}

# ---------------------------------------------------------------- 1. paper
echo "[1/9] paper sources + built PDFs"
for f in main.tex appendix.tex numbers.tex appendix-refs.tex main.bib pvldb.sty acmart.cls ACM-Reference-Format.bst build.sh; do
  [ -f "$PAPER/$f" ] && cp -f "$PAPER/$f" "$OUT/paper/" && say "$f" copied || say "$f" MISSING
done
[ -f "$PAPER/main.pdf" ] && cp -f "$PAPER/main.pdf" "$OUT/paper/main.pdf"
[ -f "$WORK/appendix.pdf" ] && cp -f "$WORK/appendix.pdf" "$OUT/paper/appendix_supplement.pdf"

# ---------------------------------------------------------------- 2. core mining pipeline
echo "[2/9] core mining pipeline (Pattern Catalog + FSM + Accept gate)"
if [ -d "$WS/sdccheck/core_algo" ]; then
  copytree "$WS/sdccheck/core_algo" "$OUT/core"
  say "core_algo -> core/" copied
fi
# verifier + constraint library from the full research tree (not in the public skeleton)
for f in main.py dp_consistency_check.sql; do
  [ -f "$WS/sdccheck/$f" ] && cp -f "$WS/sdccheck/$f" "$OUT/core/" && say "sdccheck/$f" copied
done
mkdir -p "$OUT/core/ablation_scripts"
[ -d "$WS/sdccheck/scripts/ablation" ] && cp -rf "$WS/sdccheck/scripts/ablation/." "$OUT/core/ablation_scripts/" && say "scripts/ablation" copied
for f in ERROR_INJECTION_GUIDE.md INJECTION_ARCHITECTURE.md INJECTION_METHODS.md HIERARCHICAL_CONSTRAINT_CHECK_README.md AGENTS_CONFIG_INTEGRATION.md; do
  [ -f "$WS/sdccheck/$f" ] && cp -f "$WS/sdccheck/$f" "$OUT/docs/" && say "docs/$f" copied
done

# ---------------------------------------------------------------- 3. collector (VTimeline)
echo "[3/9] training instrumentation (live collector = VTimeline)"
if [ -d "$WS/VTimeline" ]; then
  copytree "$WS/VTimeline" "$OUT/collector/vtimeline" --exclude='*.bak*' --exclude=build
  say "VTimeline (live collector)" copied
fi
[ -f "$WS/Megatron-LM/new_megatron_collector.py" ] && cp -f "$WS/Megatron-LM/new_megatron_collector.py" "$OUT/collector/megatron_collector_legacy_sha256.py"

# ---------------------------------------------------------------- 4. evaluation tree
# The paper cites artifact paths as benchmark/eval/... (e.g. real_sdc/, template_induction/),
# so the artifact mirrors those paths exactly. The full research tree is 9.8 GB; two
# directories carry almost all of it and are indexed instead of copied:
#   rebuttal_v1/ (9.2 GB replay outputs) and hunt_log/ (550 MB mining logs).
echo "[4/9] evaluation tree (benchmark/eval, paths as cited in the paper)"
copytree "$WORK/benchmark/eval" "$OUT/benchmark/eval" \
  --exclude=rebuttal_v1 --exclude=hunt_log --exclude=.ipynb_checkpoints
# overlay anything newer that only exists in the VLDB paper checkout
copytree "$PAPER/benchmark/eval" "$OUT/benchmark/eval"
for f in REPRODUCTION_GUIDE.md SCHEMA.md README.md phase2_real_bug_archaeology.md status.json run_on_gpu.sh; do
  src="$PAPER/benchmark/$f"; [ -f "$src" ] || src="$WORK/benchmark/$f"
  [ -f "$src" ] && cp -f "$src" "$OUT/benchmark/$f"
done
say "benchmark/eval" "$(find "$OUT/benchmark/eval" -type f | wc -l) files, $(du -sb --apparent-size "$OUT/benchmark/eval" | cut -f1 | numfmt --to=iec)"
# index for the two excluded heavy directories
{
  echo "# Heavy evaluation outputs kept outside git (see docs/DATA_AVAILABILITY.md)"
  echo "# rel_path,size_bytes,n_files"
  for d in "$WORK/benchmark/eval/rebuttal_v1" "$WORK/benchmark/eval/hunt_log"; do
    [ -d "$d" ] || continue
    printf '%s,%s,%s\n' "${d#$WS/}" "$(du -sb "$d" | cut -f1)" "$(find "$d" -type f | wc -l)"
    find "$d" -maxdepth 1 -mindepth 1 | while read -r sub; do
      printf '%s,%s,%s\n' "${sub#$WS/}" "$(du -sb "$sub" | cut -f1)" "$(find "$sub" -type f 2>/dev/null | wc -l)"
    done
  done
} > "$OUT/benchmark/heavy_outputs_index.csv"
say "heavy_outputs_index.csv" "$(( $(wc -l < "$OUT/benchmark/heavy_outputs_index.csv") - 2 )) entries"

# ---------------------------------------------------------------- 5. corpus / taxonomy extras
echo "[5/9] taxonomy methodology + provenance audit"
[ -f "$WORK/docs/v2_semantic_guided/taxonomy_methodology.md" ] && cp -f "$WORK/docs/v2_semantic_guided/taxonomy_methodology.md" "$OUT/benchmark/eval/" && say "taxonomy_methodology.md" copied
[ -f "$WORK/benchmark/eval/corpus_provenance_audit.md" ] && say "corpus_provenance_audit.md" "in eval tree"

# ---------------------------------------------------------------- 6. fault injection (vtime-instrumented replays)
echo "[6/9] fault-injection drivers (Megatron-LM inject_* + trace DB index)"
mkdir -p "$OUT/benchmark/injection/launch_scripts" "$OUT/benchmark/injection/overhead_raw"
n=0
for f in "$WS"/Megatron-LM/pretrain_inject_*.sh "$WS"/Megatron-LM/pretrain_dp_normal*.sh \
         "$WS"/Megatron-LM/generate_multi_direction_normal.sh "$WS"/Megatron-LM/measure_overhead.sh \
         "$WS"/Megatron-LM/run_overhead_*.sh "$WS"/Megatron-LM/batch_sdccheck*.sh \
         "$WS"/Megatron-LM/batch_constraint_check.sh "$WS"/Megatron-LM/long_run_invariant_mining.sh; do
  [ -f "$f" ] && cp -f "$f" "$OUT/benchmark/injection/launch_scripts/" && n=$((n+1))
done
say "injection/overhead launch scripts" "$n copied"
# raw H20 overhead measurement logs backing Table tab:overhead (192s -> 27.5s -> 25s @ 732ms baseline)
m=0
for f in "$WS"/Megatron-LM/overhead_*.log "$WS"/Megatron-LM/logs_gpu_lsk32/*.log \
         "$WS"/Megatron-LM/logs_gpu_lsk32/nohup_overhead.out "$WS"/Megatron-LM/logs_gpu_lsk32/*.sh; do
  [ -f "$f" ] && cp -f "$f" "$OUT/benchmark/injection/overhead_raw/" && m=$((m+1))
done
say "overhead_raw (H20 logs)" "$m copied"
# index of the on-disk trace DBs (paths + sizes), not the DBs themselves
{
  echo "# Trace-DB index (fault-injected + clean runs)"
  echo "# generated by scripts/assemble_from_workspace.sh; DBs stay out of git (see docs/DATA_AVAILABILITY.md)"
  echo "# db_dir,size_bytes,n_files"
  for d in "$WS"/Megatron-LM/*_test_db "$WS"/Megatron-LM/normal_db "$WS"/Megatron-LM/dp_normal_db "$WS"/Megatron-LM/overhead_*_db; do
    [ -d "$d" ] || continue
    printf '%s,%s,%s\n' "${d#$WS/}" "$(du -sb "$d" | cut -f1)" "$(find "$d" -type f | wc -l)"
  done
} > "$OUT/benchmark/injection/trace_db_index.csv"
say "trace_db_index.csv" "$(( $(wc -l < "$OUT/benchmark/injection/trace_db_index.csv") - 3 )) DBs indexed"

# ---------------------------------------------------------------- 7. guard-ablation logs (42db x 3lib)
echo "[7/9] leave-one-out guard ablation + holdout mining logs"
mkdir -p "$OUT/experiments/guard_ablation" "$OUT/experiments/holdout_mining"
for f in d1_results.csv _phase3_master.log; do
  [ -f "$WS/sdccheck/logs/ablation_d1/$f" ] && cp -f "$WS/sdccheck/logs/ablation_d1/$f" "$OUT/experiments/guard_ablation/${f#_}" && say "guard_ablation/${f#_}" copied
done
find "$WS/sdccheck/logs/ablation_d1" -maxdepth 1 -name '*.json' -exec cp -f {} "$OUT/experiments/guard_ablation/" \; 2>/dev/null || true
mkdir -p "$OUT/experiments/guard_ablation_no_adversarial"
find "$WS/sdccheck/logs/ablation_no_adv" -maxdepth 1 \( -name '*.csv' -o -name '*.json' \) -exec cp -f {} "$OUT/experiments/guard_ablation_no_adversarial/" \; 2>/dev/null || true
say "guard ablation cells" "$(find "$OUT/experiments" -name '*.json' | wc -l) json + $(find "$OUT/experiments" -name '*.csv' | wc -l) csv"
copytree "$WS/sdccheck/logs/holdout_mining" "$OUT/experiments/holdout_mining"
# 12 MB LLM mining transcripts compress ~20x; keep them, gzipped
find "$OUT/experiments/holdout_mining" -type f -size +2M -name '*.log' -exec gzip -f {} \; 2>/dev/null || true

# ---------------------------------------------------------------- 8. baselines
echo "[8/9] baseline detectors (scripts live in benchmark/eval; logs collected here)"
copytree "$WS/sdccheck/logs/baselines" "$OUT/baselines/logs"
[ -d "$WS/traincheck/TrainCheck" ] && {
  printf 'TrainCheck baseline uses the official implementation, checked out locally at\n  %s\nsize: %s\nWe do not vendor it here; see docs/DATA_AVAILABILITY.md.\n' \
    "traincheck/TrainCheck" "$(du -sh "$WS/traincheck/TrainCheck" | cut -f1)" > "$OUT/baselines/TRAINCHECK_UPSTREAM.txt"
  say "TRAINCHECK_UPSTREAM.txt" written
}

# ---------------------------------------------------------------- 9. figures
echo "[9/9] figures: PDFs used by the paper + their generator scripts"
mkdir -p "$OUT/figures/generators"
used=$(grep -ohE 'includegraphics(\[[^]]*\])?\{[^}]*\}' "$PAPER/main.tex" "$PAPER/appendix.tex" \
       | sed -E 's/.*\{(.*)\}/\1/' | sort -u)
miss=0; got=0
while read -r rel; do
  [ -z "$rel" ] && continue
  if [ -f "$PAPER/$rel" ]; then
    mkdir -p "$OUT/$(dirname "$rel")"; cp -f "$PAPER/$rel" "$OUT/$rel"; got=$((got+1))
  else
    say "figure $rel" MISSING; miss=$((miss+1))
  fi
done <<< "$used"
say "figures used by paper" "$got copied, $miss missing"
cp -f "$PAPER"/figures/gen_*.py "$OUT/figures/generators/" 2>/dev/null || true
for f in FIGURE_BRIEF_FOR_DESIGNER.md FIGURE9_DESIGN_BRIEF.md; do
  [ -f "$PAPER/$f" ] && cp -f "$PAPER/$f" "$OUT/docs/"
done

# planning / provenance docs that belong with the artifact
for f in experiment_registry.md artifact_checklist.md REAL_BUG_DETECTION_RUNBOOK.md REPO_INDEX.md; do
  [ -f "$PAPER/$f" ] && cp -f "$PAPER/$f" "$OUT/docs/"
done
copytree "$PAPER/docs" "$OUT/docs/paper_drafts"

echo
echo "assembled at $OUT  ($(du -sh "$OUT" | cut -f1))"
