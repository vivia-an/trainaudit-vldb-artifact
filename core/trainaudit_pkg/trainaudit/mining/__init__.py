"""TrainAudit catalog-guided invariant mining (paper §3.2 layers 1–4).

Layer 1: catalog selection and grounding hypotheses (LLM)
Layer 2: predicate-shape enumeration (deterministic)
Layer 3: healthy-trace validation + tolerance auto-learning (layer3_validate.py)
Layer 4: spurious filter (LLM)

The deterministic core (Layers 2–3) closes the mining loop without any
LLM dependency: a canonical catalog template plus a Hypothesis grounding
strategy and trace schema → up to ~50 candidate Predicates → run each
on healthy traces → keep the ones that hold → re-run on buggy traces →
violations identify silent corruption. The LLM layers (1, 4) extend the
hypothesis pool and prune spurious predicates respectively.
"""
from .hypothesis_schema import Hypothesis, RelationType
from .layer1_hypothesis import propose_hypotheses
from .layer2_enumerate import enumerate_predicates, schema_introspect
from .layer3_validate import (HealthyValidationResult, infer_tolerance,
                               validate_against_healthy)
from .layer4_filter import FilterDecision, filter_predicates

__all__ = [
    "Hypothesis", "RelationType",
    "propose_hypotheses",
    "enumerate_predicates", "schema_introspect",
    "HealthyValidationResult", "infer_tolerance",
    "validate_against_healthy",
    "FilterDecision", "filter_predicates",
]
