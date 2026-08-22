"""Canonical Pattern Catalog snapshot used by mining and deployment.

This module is the implementation-side index of the frozen catalog evaluated
in ``benchmark/eval/template_induction``.  A catalog template is a semantic
obligation (T01--T35); executable predicates remain framework-specific
groundings of one of these entries.  Internal DSL shapes and miner relation
types are compilation details, not additional template catalogs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


FROZEN_CATALOG_SHA256 = (
    "cfa30e182a4aa5c6637423dcbd95f15cb20c2b34a1da80000f5df6c6b8176734"
)


@dataclass(frozen=True)
class CatalogTemplate:
    template_id: str
    name: str
    relation_operator: str


_TEMPLATES = (
    CatalogTemplate("T01", "cross-rank-replica-equality", "equality_across_ranks"),
    CatalogTemplate("T02", "logged-metric-reduction-consistency", "equality_across_ranks"),
    CatalogTemplate("T03", "config-invariant-execution-equivalence", "reference_equivalence"),
    CatalogTemplate("T04", "spec-reference-conformance", "reference_equivalence"),
    CatalogTemplate("T05", "designated-dtype-fidelity", "dtype_preservation"),
    CatalogTemplate("T06", "storage-dtype-interpretation-match", "dtype_preservation"),
    CatalogTemplate("T07", "gradient-accumulation-conservation", "conservation"),
    CatalogTemplate("T08", "scale-factor-consistency", "value_scaling_consistency"),
    CatalogTemplate("T09", "gradient-norm-computation-fidelity", "value_scaling_consistency"),
    CatalogTemplate("T10", "producer-consumer-completion-ordering", "ordering"),
    CatalogTemplate("T11", "checkpoint-restore-state-equality", "restoration_after_reload"),
    CatalogTemplate("T12", "checkpoint-save-completeness", "count_frequency_match"),
    CatalogTemplate("T13", "dataloader-bookkeeping-reset", "restoration_after_reload"),
    CatalogTemplate("T14", "scaling-scalar-finiteness", "boundedness"),
    CatalogTemplate("T15", "counter-monotonic-progress", "monotonicity"),
    CatalogTemplate("T16", "positional-encoding-position-fidelity", "index_consistency"),
    CatalogTemplate("T17", "loss-parameter-gradient-connectivity", "gradient_flow"),
    CatalogTemplate("T18", "configured-value-effectiveness", "config_effectiveness"),
    CatalogTemplate("T19", "packed-sequence-attention-isolation", "exclusivity"),
    CatalogTemplate("T20", "padding-loss-exclusion", "exclusivity"),
    CatalogTemplate("T21", "overflow-skip-step-validity", "count_frequency_match"),
    CatalogTemplate("T22", "parameter-update-effectiveness", "update_effectiveness"),
    CatalogTemplate("T23", "unrelated-operation-state-preservation", "state_preservation"),
    CatalogTemplate("T24", "intra-run-recomputation-determinism", "determinism"),
    CatalogTemplate("T25", "sharded-layout-conformance", "sharding_layout_consistency"),
    CatalogTemplate("T26", "exactly-once-side-effect", "count_frequency_match"),
    CatalogTemplate("T27", "init-distribution-conformance", "value_scaling_consistency"),
    CatalogTemplate("T28", "master-working-weight-copy-fidelity", "copy_consistency"),
    CatalogTemplate("T29", "router-gate-probability-range-validity", "boundedness"),
    CatalogTemplate("T30", "static-attention-pattern-mass-exclusion", "exclusivity"),
    CatalogTemplate("T31", "sort-key-order-conformance", "ordering"),
    CatalogTemplate("T32", "pipeline-layer-partition-conformance", "structural_integrity"),
    CatalogTemplate("T33", "supervision-label-index-range-validity", "boundedness"),
    CatalogTemplate("T34", "training-step-budget-data-sufficiency", "count_frequency_match"),
    CatalogTemplate("T35", "frozen-parameter-update-exclusion", "exclusivity"),
)

CATALOG_BY_ID: Dict[str, CatalogTemplate] = {
    template.template_id: template for template in _TEMPLATES
}


def catalog_templates() -> Iterable[CatalogTemplate]:
    """Return the canonical frozen entries in template-id order."""
    return _TEMPLATES


def get_catalog_template(template_id: str) -> CatalogTemplate:
    """Resolve a canonical template id, raising on legacy/unknown ids."""
    try:
        return CATALOG_BY_ID[template_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown catalog template {template_id!r}; expected T01--T35 "
            f"from frozen snapshot {FROZEN_CATALOG_SHA256[:12]}"
        ) from exc


def maybe_catalog_template(
    template_id: Optional[str],
) -> Optional[CatalogTemplate]:
    """Resolve an optional catalog grounding."""
    return None if template_id is None else get_catalog_template(template_id)
