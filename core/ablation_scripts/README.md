# Ablation Experiment — Reviewer Q3 Response

## Goal
Produce a leave-one-out FP-decomposition table to address reviewer Q3:
> "57 → 0 是单点轶事，不是统计证据 ... 必须补 Table — no adversarial /
>  no precondition / no topology, full mechanism — FP 分解"

## Files

- `make_ablation_libs.py` — generates 3 patched constraint libraries:
  - `config/lib_full.json`         (unchanged baseline)
  - `config/lib_no_topo.json`      (strips topology keys from `applicable_conditions`)
  - `config/lib_no_precond.json`   (strips non-topology preconditions)
- `run_matrix.sh`                  — runs 3 libs × N dbs sequentially
- `status.sh`                      — live progress dump for in-flight runs
- `aggregate_fp.py`                — produces `ablation_results.csv` from per-cell JSONs

## Code touchpoints (added for ablation, all minimal)

- `sdccheck/predefined_constraints.py` — `_load_predefined_constraints` now reads
  `SDC_CONSTRAINTS_FILE` env var if `config_path` is None.
- `sdccheck/__main__.py` — added `--constraints-file`, `--provider`, `--reports-out`
  flags; sets `SDC_INCREMENTAL_REPORTS` env var so partial reports survive a kill.
- `sdccheck/llm_orchestrator.py::run` — incremental dump after every constraint to
  `SDC_INCREMENTAL_REPORTS`; status normalization (`ResultStatus.PASS` -> `pass`).
- `config/llm_config.yaml`            — replaced the OpenAI-templated config with a
  DeepSeek-only one (original preserved at `llm_config.yaml.bak.YYYYMMDD`).

## Run

```bash
cd /volume/posttrain/users/lsk/sdc/lsk/sdccheck

# 1. Build libraries
python3 scripts/ablation/make_ablation_libs.py

# 2. Run the matrix (3 libs × 4 local dbs). Set PER_RUN_TIMEOUT high (3600s).
./scripts/ablation/run_matrix.sh 3600

# 3. Watch progress while it runs
./scripts/ablation/status.sh

# 4. Aggregate after completion
python3 scripts/ablation/aggregate_fp.py \
    --reports-dir logs/ablation \
    --library-json config/lib_full.json \
    --csv-out logs/ablation/ablation_results.csv
```

## Caveats

1. **No D1 fixed-commit dbs locally.** `/volume/qscai/lsk/Megatron-LM/*_test_db`
   referenced by `batch_sdccheck_d.sh` is not mounted on this host. We only have
   4 local dbs in `sdccheck/data/`, used as a methodological smoke test.
   The full reviewer-quality numbers require running this same matrix on the
   17 D1 fixed-commit dbs once they are reachable.

2. **`--provider deepseek`** uses the hardcoded API key in
   `agents/llm_config.yaml`. Mock provider is unusable for the verifier because
   it returns canned SQL that doesn't match real schemas.

3. **Status normalization.** Earlier runs may write `ResultStatus.PASS` style
   strings; the aggregator splits on `.` and lowercases.

4. **Fourth row (no-adversarial)** — two phases:
   - **Phase-1 (verifier proxy, fast):** `make_ablation_libs.py` builds
     `config/lib_no_adversarial.json` (strips *all* `applicable_conditions`).
     Run `./scripts/ablation/run_no_adversarial_d1.sh smoke` →
     `logs/ablation_no_adv/no_adversarial_results.csv`.
   - **Phase-2 (true θ=0 mining, ~8h):** `./scripts/ablation/run_mining_no_adversarial.sh`
     → `config/lib_no_adversarial_mined.json` + `logs/ablation_no_adv/_mining.log`.
