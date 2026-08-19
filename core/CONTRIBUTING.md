# Contributing

This tree is an **anonymous review artifact**. External contributions are not expected
during the review period; de-anonymize maintainership at camera-ready.

1. Keep API keys out of the tree — use `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`.
2. Run offline smoke before PRs: `python3 run_smoke.py`.
3. Prefer changes under `agents/` + `config/`; do not add machine-specific absolute paths.
4. Paper path is `SDC_PAPER_ALIGN=1`; default `0` is the open-ended control arm — do not break it.
5. Funnel numbers are reproduced structurally via `scripts/reproduce_funnel_counts.py` (not full LLM re-mining).
