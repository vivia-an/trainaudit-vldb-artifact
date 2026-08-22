"""TrainAudit invariant DSL.

A predicate is the structured form of an invariant rule. Each predicate
uses one internal compilation shape and compiles to a DuckDB SQL query
(Workstream A2). Semantic templates come only from the Pattern Catalog.
The DSL co-exists with hand-written Python rules — see
`registry/MAPPING.md` for the classification of every existing rule.
"""
from .predicate import (Bound, BoundCondition, BoundKind, Precondition,
                        Predicate, PredicateShape, Scope, Tolerance)
from .loader import LoaderError, load_predicate, load_predicates_dir
from .compiler import CompiledRule, Violation, compile_predicate
from .runner import run_compiled_one, run_dsl_rules, violation_event_ids

__all__ = [
    "Bound", "BoundCondition", "BoundKind", "Precondition", "Predicate",
    "PredicateShape", "Scope", "Tolerance",
    "LoaderError", "load_predicate", "load_predicates_dir",
    "CompiledRule", "Violation", "compile_predicate",
    "run_compiled_one", "run_dsl_rules", "violation_event_ids",
]
