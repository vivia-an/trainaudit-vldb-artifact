# TrainAudit Core

Self-contained reference for **verified constraint mining**:
Pattern Catalog → multi-agent FSM → Accept (CE ∧ Conf) → healthy-run → SQL verifier.

## Layout

```
core_algo/
  run_miner.py          # LLM mining entry (uses agents/)
  run_smoke.py          # offline smoke (no API key)
  agents/               # FSM + AutoGen workflow + Accept gate
  config/               # frozen Pattern Catalog (T01–T35)
  data/                 # toy traces + funnel CSVs
  *.py                  # toy trace, verifier, topology prune, …
```

## Install

```bash
pip install -r requirements.txt
```

## Offline smoke (no API key)

```bash
python3 run_smoke.py
```

## Mining with agents + FSM

```bash
export SDC_PAPER_ALIGN=1              # paper path (templates + Accept + FSM)
export SDC_TARGET_TEMPLATE=T01        # optional
export DEEPSEEK_API_KEY=...           # required for LLM rounds
python3 run_miner.py
```

| Env | Behavior |
|-----|----------|
| `SDC_PAPER_ALIGN` unset / `0` | Control arm: open-ended mining, **no** catalog templates |
| `SDC_PAPER_ALIGN=1` | Catalog S1 → FSM S1–S5 → Accept → healthy-run |

Optional: `SDC_HEALTHY_DBS` (colon-separated DuckDB paths). Without it, smoke/toy DBs under `data/` are used; monorepo Megatron clean dumps are discovered only if present beside the package.

Call chain:

```
run_miner.py
  → agents/workflow_task_generate_with_writeAgent.py
       ├─ catalog_picker.pick_pattern / prompt_block
       ├─ fsm_stages.MiningFSM  (S1→S2→S3→S4→S5)
       └─ paper_accept_gate.check_accept
```

LLM config SSOT: `agents/llm_config.yaml` (mirrored in `config/llm_config.yaml`).
Agent system prompts are English (`agents/mining_prompts.py`).

## Funnel sizes (structural reproduce)

```bash
python3 scripts/reproduce_funnel_counts.py
```

| Stage | L1 | L2 | L3 | L4 | Deploy |
|-------|----|----|----|----|--------|
| Count | 420 | 5334 | 3436 | 357 | 45 |

This recomputes from checked-in breakdown/CSV artifacts; it does **not** re-run LLM mining of 5334 candidates.

## Package for release

```bash
bash pack_release.sh /tmp/trainaudit-core
```

## Citation / License / Security

See `CITATION.cff`, `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`.
