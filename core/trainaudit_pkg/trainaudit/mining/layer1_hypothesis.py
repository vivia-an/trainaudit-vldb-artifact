"""Layer 1: LLM hypothesis generation.

Take a chunk of framework source code (e.g. Megatron's router.py) and
ask the LLM to propose Hypotheses that, if violated, would indicate a
silent error in this code. The Hypotheses then feed Layer 2 enumeration
+ Layer 3 healthy validation.

Same pluggable LLMClient interface as C2's RCA agent: in CI we use
StubLLMClient (deterministic), in production swap in claude-proxy-v3
or the Anthropic SDK.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from ..catalog import catalog_templates
from ..diagnosis.rca_agent import LLMClient
from .hypothesis_schema import Hypothesis, RelationType


_CATALOG_CHOICES = "\n".join(
    f"  - {template.template_id}: {template.name} "
    f"({template.relation_operator})"
    for template in catalog_templates()
)

_SYSTEM_PROMPT = (
    "You are an LLM training framework auditor. Given a snippet of "
    "framework source code, select 2–4 applicable entries from the frozen "
    "Pattern Catalog and propose how to ground them. Do not invent a new "
    "template id or relation family. Each Hypothesis "
    "must specify:\n"
    "  - catalog_template_id: one id from the catalog below\n"
    "  - relation_type ∈ {cross_rank_equal, tensor_stat_bound, "
    "payload_field_compare, cross_step_monotonic, structural_presence, "
    "conditional_check}\n"
    "  - entities: list of trace entities the invariant references "
    "(e.g. ['param', 'grad_norm'])\n"
    "  - dimensions: ['rank', 'step'] etc.\n"
    "  - rationale: one sentence on why this invariant matters\n\n"
    "Return JSON: {\"hypotheses\": [...]}\n\n"
    "Frozen Pattern Catalog:\n"
    + _CATALOG_CHOICES
    + "\n")


_SYSTEM_PROMPT_FREEFORM = (
    "You are an LLM training framework auditor. Given a snippet of "
    "framework source code, propose 2–4 invariants that, if violated at "
    "runtime, would indicate a silent error in this code. Each Hypothesis "
    "must specify:\n"
    "  - relation_type ∈ {cross_rank_equal, tensor_stat_bound, "
    "payload_field_compare, cross_step_monotonic, structural_presence, "
    "conditional_check}\n"
    "  - entities: list of trace entities the invariant references "
    "(e.g. ['param', 'grad_norm'])\n"
    "  - dimensions: ['rank', 'step'] etc.\n"
    "  - rationale: one sentence on why this invariant matters\n\n"
    "Return JSON: {\"hypotheses\": [...]}\n")


def _build_user_prompt(source: str, framework: str = "") -> str:
    fw = f"Framework: {framework}\n" if framework else ""
    return (f"{fw}Source code:\n```\n{source[:8000]}\n```\n\n"
            f"Propose Hypotheses as JSON.")


def propose_hypotheses(source: str, *,
                        framework: str = "",
                        llm_client: Optional[LLMClient] = None,
                        use_catalog: bool = True
                        ) -> List[Hypothesis]:
    """Ask the LLM for Hypothesis candidates from a source code chunk.

    When `use_catalog` is False the LLM is given no Pattern Catalog and any
    template id it happens to emit is discarded, so the returned Hypotheses
    carry `catalog_template_id=None`. This is the ablation arm.
    """
    if llm_client is None:
        llm_client = _DefaultStubHypothesisLLM()
    system = _SYSTEM_PROMPT if use_catalog else _SYSTEM_PROMPT_FREEFORM
    response = llm_client(system, _build_user_prompt(source, framework))
    return _parse_hypothesis_response(response, use_catalog=use_catalog)


def _parse_hypothesis_response(text: str,
                                *, use_catalog: bool = True
                                ) -> List[Hypothesis]:
    """Tolerantly extract a JSON {"hypotheses": [...]} block from LLM text."""
    # Try fenced ```json ... ``` first
    candidates = []
    fence = text.rfind("```json")
    if fence != -1:
        end = text.find("```", fence + 7)
        if end != -1:
            try:
                candidates.append(json.loads(text[fence + 7:end].strip()))
            except Exception:
                pass
    # Try the whole payload, then the outermost { ... } span. A real LLM
    # replies with bare nested JSON, for which the rfind("{") probe below
    # lands on the last *inner* object and always fails.
    for blob_text in (text.strip(),
                      text[text.find("{"):text.rfind("}") + 1]
                      if "{" in text and "}" in text else ""):
        if not blob_text:
            continue
        try:
            candidates.append(json.loads(blob_text))
        except Exception:
            pass
    # Try last { ... } block
    last_open = text.rfind("{")
    if last_open != -1:
        try:
            candidates.append(json.loads(text[last_open:]))
        except Exception:
            pass
    for blob in candidates:
        if not isinstance(blob, dict):
            continue
        items = blob.get("hypotheses") or []
        out: List[Hypothesis] = []
        for h in items:
            try:
                out.append(Hypothesis(
                    relation_type=RelationType(h["relation_type"]),
                    catalog_template_id=(h.get("catalog_template_id")
                                          if use_catalog else None),
                    entities=list(h.get("entities", [])),
                    dimensions=list(h.get("dimensions", [])),
                    scope_hint=h.get("scope_hint", {}),
                    rationale=h.get("rationale", ""),
                ))
            except Exception:
                continue
        if out:
            return out
    return []


# ---- stub that returns plausible hypotheses without LLM access ----------


class _DefaultStubHypothesisLLM:
    """Deterministic stub: pattern-match on source for known fingerprints
    (clip_grad / RMSNorm / softmax / scheduler) and emit a corresponding
    Hypothesis. Sufficient for harness self-test without real LLM."""

    def __call__(self, system: str, user: str, *,
                 max_tokens: int = 1024) -> str:
        src_lower = user.lower()
        hypotheses = []
        if "clip_grad" in src_lower or "max_norm" in src_lower:
            hypotheses.append({
                "catalog_template_id": "T09",
                "relation_type": "payload_field_compare",
                "entities": ["norm", "threshold"],
                "dimensions": [],
                "rationale": ("clip_grad must enforce post_norm <= max_norm; "
                              "violation means clipping was bypassed"),
            })
        if "rmsnorm" in src_lower or "layernorm" in src_lower:
            hypotheses.append({
                "catalog_template_id": "T04",
                "relation_type": "tensor_stat_bound",
                "entities": ["norm_output"],
                "dimensions": [],
                "rationale": ("normalizer output rms must stay near 1.0; "
                              "drift indicates eps placement / L2-vs-RMS bug"),
            })
        if "router" in src_lower or "softmax" in src_lower or "topk" in src_lower:
            hypotheses.append({
                "catalog_template_id": "T17",
                "relation_type": "tensor_stat_bound",
                "entities": ["router_output"],
                "dimensions": [],
                "rationale": ("router output must NOT be a one-hot "
                              "distribution; topk-then-softmax over a size-1 "
                              "dim is a known degenerate case"),
            })
        if "scheduler" in src_lower or "last_epoch" in src_lower:
            hypotheses.append({
                "catalog_template_id": "T11",
                "relation_type": "conditional_check",
                "entities": ["param_group", "scheduler"],
                "dimensions": [],
                "rationale": ("scheduler resume requires initial_lr present "
                              "on every param_group"),
            })
        if not hypotheses:
            hypotheses.append({
                "catalog_template_id": "T18",
                "relation_type": "structural_presence",
                "entities": ["module"],
                "dimensions": [],
                "rationale": "fallback: build snapshot must report parameters",
            })
        return "```json\n" + json.dumps({"hypotheses": hypotheses}) + "\n```"
