# Agents (verified mining FSM)

Multi-agent miner used by `../run_miner.py`.

| Module | Role |
|--------|------|
| `workflow_task_generate_with_writeAgent.py` | AutoGen group chat + write-back |
| `mining_prompts.py` | English system prompts (routing + 4-stage funnel) |
| `fsm_stages.py` | S1–S5 transitions |
| `catalog_picker.py` | S1 PickPattern + prompt block |
| `paper_accept_gate.py` | Accept(CE ∧ Conf) + healthy-run |
| `healthy_sql.py` | SQL resolve for healthy-run |
| `evidence_retrieval.py` | S2 grep/read tools |
| `llm_config.py` / `llm_config.yaml` | LLM via env keys only (SSOT) |
| `context_variables.py` | Shared workflow context |
| `chinese_generator_no_unicode.py` | English constraint JSON helpers (filename legacy) |
| `ag2_deepseek_thinking_patch.py` | DeepSeek reasoning_content patch |

Default constraint library path: `predefined_constraints.json` beside this folder (no machine-specific absolutes).

```bash
cd ..   # core_algo/
export SDC_PAPER_ALIGN=1
export SDC_TARGET_TEMPLATE=T01
export DEEPSEEK_API_KEY=...
python3 run_miner.py
```
