#!/usr/bin/env python3
"""TrainAudit verified-mining entrypoint (multi-agent FSM).

Usage:
  export SDC_PAPER_ALIGN=1          # catalog + Accept + FSM
  export SDC_TARGET_TEMPLATE=T01    # optional
  export DEEPSEEK_API_KEY=...
  python3 run_miner.py

Default SDC_PAPER_ALIGN=0 keeps the open-ended control arm (no templates).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE / "agents"
sys.path.insert(0, str(AGENTS))
os.chdir(AGENTS)  # chinese_generator writes relative JSON beside agents/


def main() -> int:
    from ag2_deepseek_thinking_patch import apply_ag2_reasoning_content_patch

    apply_ag2_reasoning_content_patch()

    constraints_file = os.environ.get(
        "SDC_CONSTRAINTS_OUTPUT",
        str(AGENTS / "predefined_constraints.json"),
    )
    max_iterations = int(os.environ.get("SDC_MAX_ITERATIONS", "100"))
    align = os.environ.get("SDC_PAPER_ALIGN", "0")

    print("=" * 60)
    print("TrainAudit miner")
    print("SDC_PAPER_ALIGN=%s  template=%s" % (
        align,
        os.environ.get("SDC_TARGET_TEMPLATE", "(auto)"),
    ))
    print("constraints → %s" % constraints_file)
    print("=" * 60)

    from workflow_task_generate_with_writeAgent import WorkflowTaskGenerateWithWriteAgent

    generator = WorkflowTaskGenerateWithWriteAgent(
        constraints_file_path=constraints_file,
        max_iterations=max_iterations,
    )

    task = """
Collaborative constraint reasoning for distributed LLM training.
Propose guarded constraints, validate with counterexamples, and persist
accepted rules via WRITE_CONSTRAINT.

Routing: megatron_expert → coordinate → write_agent → JSON library.
Prefer NEXT_ACTION: WRITE_CONSTRAINT after validation.
"""
    target_category = os.environ.get("SDC_TARGET_CATEGORY", "").strip()
    if target_category:
        task += (
            f'\nFocus category "{target_category}". '
            f'Call analyze_existing_constraints(category_filter="{target_category}") first.\n'
        )

    ctx = generator.run(task)
    written = ctx.get("constraints_written", 0)
    print("constraints_written=%s write_completed=%s" % (
        written,
        ctx.get("write_completed"),
    ))
    return 0 if ctx.get("write_completed") or written else 1


if __name__ == "__main__":
    raise SystemExit(main())
