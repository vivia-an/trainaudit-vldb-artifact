# Which overhead file backs which claim

Two unrelated overhead measurements live in this artifact. Read this before citing either.

| File | Workload | What it is | Used by the paper? |
|---|---|---|---|
| `../injection/overhead_h20.csv` (+ `overhead_raw/`, `parse_overhead_logs.py`) | **1.2B GPT, H20, DP=2, bf16** | the measurement `tab:overhead` (§5.5) reports: 732 ms baseline step, full dump 192 s → 27.5 s → 25 s | **yes** |
| `overhead.csv`, `paper_table_overhead.md`, `paper_table_overhead_optimized.md` | `gpt-tiny` (12 layers, h=128, seq=64), **CPU, single rank** | an early 2026-05 hook-sampling study; reports +829.7% / +876.6% and an async-vs-sampling comparison | **no** — superseded |

The `gpt-tiny` CPU numbers are kept for provenance only. They over-state overhead
because a 19 ms toy step makes any fixed per-hook cost dominate, which is exactly why
the reported measurement was redone on a representative 1.2B model on GPU.

Regenerate the reported table and check it against the paper:

```bash
python3 ../injection/parse_overhead_logs.py --check
```
