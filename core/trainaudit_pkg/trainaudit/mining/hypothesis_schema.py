"""Hypothesis schema — a catalog-guided input to the Layer 2 enumerator.

A Hypothesis selects one canonical Pattern Catalog entry and an internal
enumeration strategy, without binding either to a specific trace field. The
Layer 2 enumerator combines it with trace schema introspection to produce
concrete framework-grounded `Predicate` candidates.

The six internal enumeration strategies map to four SQL predicate shapes plus
two cross-event derivations:

  cross_rank_equal     → STRUCTURAL_PRESENCE on build.snapshot.cross_rank_cksums
  tensor_stat_bound    → TENSOR_STAT_BOUND   (numeric stats on tensor outputs)
  payload_field_compare→ PAYLOAD_FIELD_COMPARE (one or two scalar fields)
  cross_step_monotonic → PAYLOAD_FIELD_COMPARE with bound.kind=monotonic
  structural_presence  → STRUCTURAL_PRESENCE (keys / counts must be non-zero)
  conditional_check    → CONDITIONAL_CHECK (precondition gates the bound)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..catalog import get_catalog_template


class RelationType(str, Enum):
    CROSS_RANK_EQUAL = "cross_rank_equal"
    TENSOR_STAT_BOUND = "tensor_stat_bound"
    PAYLOAD_FIELD_COMPARE = "payload_field_compare"
    CROSS_STEP_MONOTONIC = "cross_step_monotonic"
    STRUCTURAL_PRESENCE = "structural_presence"
    CONDITIONAL_CHECK = "conditional_check"


@dataclass
class Hypothesis:
    """One catalog-template grounding request.

    Example — "for every replica param, cross-rank cksums must agree":
        Hypothesis(
            relation_type=RelationType.CROSS_RANK_EQUAL,
            entities=["param"],
            dimensions=["rank"],
            scope_hint={"hookpoint": "build.snapshot",
                         "payload_path": "$.cross_rank_cksums[*]"},
        )

    Example — "in any module forward output, l2_norm must be finite":
        Hypothesis(
            relation_type=RelationType.TENSOR_STAT_BOUND,
            entities=["tensor"],
            dimensions=[],
            scope_hint={"hookpoints": ["module.fwd.post"]},
        )
    """

    relation_type: RelationType
    catalog_template_id: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    scope_hint: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.relation_type, RelationType):
            self.relation_type = RelationType(self.relation_type)
        if self.catalog_template_id is not None:
            get_catalog_template(self.catalog_template_id)
