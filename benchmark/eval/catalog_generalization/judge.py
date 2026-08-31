"""Coverage judge — shared by all arms so the metric is arm-neutral.

For one held-out bug and one frozen invariant taxonomy, decide whether checking
any invariant type in the taxonomy on a runtime trace would be VIOLATED by the
bug (i.e. would detect it). Strictness rules, both essential for fairness:

  1. A match requires the invariant to actually flip satisfied->violated on
     runtime trace data because of this bug — not mere semantic similarity.
  2. If the bug is only detectable by reading source (no runtime trace field
     distinguishes buggy from clean), the answer is NONE regardless of any
     semantic match. This handles the paper's runtime-unobservable gap
     identically across arms, so it cannot bias the comparison.

The judge runs at temperature 0 — it is measurement, not proposal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "catalog_ablation"))

from deepseek_client import DeepSeekClient  # noqa: E402

JUDGE_SYSTEM = (
    "You audit whether a runtime invariant checker could detect a known silent "
    "training bug. You are given (1) a bug's root-cause description with its "
    "required trace fields, and (2) a taxonomy of invariant types, each with an "
    "id and name. Decide whether checking ANY ONE of these invariant types on a "
    "runtime execution trace would be VIOLATED by this bug — i.e. would flip "
    "from satisfied to violated because of this bug, and thereby detect it.\n\n"
    "Strict rules:\n"
    "  - A match requires that the invariant, evaluated on runtime trace data, "
    "would ACTUALLY be violated by this specific bug. Semantic resemblance is "
    "not enough.\n"
    "  - If the bug is only detectable by reading source code (no runtime trace "
    "field reliably distinguishes buggy from clean), answer NONE even if a "
    "taxonomy entry sounds related.\n"
    "  - Pick the single best-matching id, or NONE.\n\n"
    "Reply with strict JSON only:\n"
    '  {"covered": true|false, "matched_id": "<id or NONE>", '
    '"why": "<one sentence>"}')


def _bug_block(bug: dict) -> str:
    return (
        f"Bug id: {bug['bug_id']}\n"
        f"Framework: {bug['framework']}\n"
        f"Category: {bug['category']}\n"
        f"Invariant type (annotator): {bug['invariant_type']}\n"
        f"Required trace fields: {bug['required_trace_fields']}\n"
        f"Runtime-observability tier: {bug['tier_field']}\n"
        f"Root cause / detection: {bug['rationale']}")


def _taxonomy_block(taxonomy: list) -> str:
    lines = []
    for t in taxonomy:
        name = t.get("name") or t.get("description", "")
        lines.append(f"  - {t['id']}: {name}")
    return "\n".join(lines)


def judge_coverage(bug: dict, taxonomy: list, client: DeepSeekClient) -> dict:
    user = ("Bug:\n" + _bug_block(bug)
            + "\n\nInvariant taxonomy:\n" + _taxonomy_block(taxonomy)
            + "\n\nWould a runtime check of any taxonomy entry detect this bug?")
    resp = client(JUDGE_SYSTEM, user, max_tokens=8192)
    return _parse(resp), resp


def _parse(text: str) -> dict:
    for probe in (text.strip(),
                  text[text.find("{"):text.rfind("}") + 1]
                  if "{" in text and "}" in text else ""):
        if not probe:
            continue
        try:
            b = json.loads(probe)
        except Exception:
            continue
        if isinstance(b, dict) and "covered" in b:
            return {"covered": bool(b["covered"]),
                    "matched_id": str(b.get("matched_id", "NONE")),
                    "why": str(b.get("why", ""))}
    return {"covered": False, "matched_id": "PARSE_ERROR", "why": text[:120]}
