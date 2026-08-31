"""Compatibility rules for internal SQL predicate shapes."""
from __future__ import annotations

from typing import Dict, FrozenSet

from .predicate import BoundKind, PredicateShape


SHAPE_BOUND_KINDS: Dict[PredicateShape, FrozenSet[BoundKind]] = {
    PredicateShape.TENSOR_STAT_BOUND: frozenset({
        BoundKind.BOUND, BoundKind.EQUALITY, BoundKind.PRESENT,
        BoundKind.MONOTONIC,
    }),
    PredicateShape.PAYLOAD_FIELD_COMPARE: frozenset({
        BoundKind.BOUND, BoundKind.EQUALITY, BoundKind.MONOTONIC,
    }),
    PredicateShape.CONDITIONAL_CHECK: frozenset({
        BoundKind.BOUND, BoundKind.EQUALITY, BoundKind.PRESENT,
    }),
    PredicateShape.STRUCTURAL_PRESENCE: frozenset({
        BoundKind.PRESENT, BoundKind.EQUALITY, BoundKind.BOUND,
    }),
}


def is_compatible(shape: PredicateShape, kind: BoundKind) -> bool:
    return kind in SHAPE_BOUND_KINDS[shape]
