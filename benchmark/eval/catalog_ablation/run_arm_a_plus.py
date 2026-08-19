"""S5 robustness check — arm A+ : give the catalog every benefit of the doubt.

The A-arm prompt renders each template as `T12: name (count_frequency_match)`
and then asks for `relation_type` from a disjoint 6-value enum. 80/80 of the
A-arm items we dropped carried a *valid* template_id and set `relation_type` to
exactly that template's `relation_operator` — the LLM used the catalog
correctly and the harness threw the result away, because no
relation_operator -> RelationType mapping exists anywhere in the codebase.

Arm A+ supplies that missing mapping and replays the SAME cached A-arm LLM
responses. If the catalog's advantage was being masked by this harness gap,
A+ is where it shows up. The mapping is authored here (it does not exist in
the system) and is deliberately generous to the catalog.

Usage:  python run_arm_a_plus.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "trainaudit"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "benchmark/eval/rebuttal_v1/A1_mining_funnel"))

from pattern_guided_llm import PatternGuidedFilterLLM
from trainaudit.catalog import catalog_templates
from trainaudit.mining import (enumerate_predicates, filter_predicates,
                               schema_introspect, validate_against_healthy)
from trainaudit.mining.hypothesis_schema import Hypothesis, RelationType
from trainaudit.store import TraceStore

HEALTHY_PER_FW = {
    "megatron":  REPO / "benchmark/eval/hunt_log/novel_hunt/megatron_clean/trace_rank0.duckdb",
    "deepspeed": REPO / "benchmark/eval/hunt_log/novel_hunt/deepspeed_bf16_only/trace_rank0.duckdb",
    "olmo":      REPO / "benchmark/eval/hunt_log/novel_hunt/olmo_core_baseline/trace_rank0.duckdb",
    "olmo_core": REPO / "benchmark/eval/hunt_log/novel_hunt/olmo_core_baseline/trace_rank0.duckdb",
}

# The missing bridge: catalog's 20 semantic operators -> L2's 6 dispatch keys.
# Authored for this robustness check; no such mapping exists in the codebase.
OPERATOR_TO_RELATION = {
    "equality_across_ranks":        RelationType.CROSS_RANK_EQUAL,
    "determinism":                  RelationType.CROSS_RANK_EQUAL,
    "state_preservation":           RelationType.CROSS_RANK_EQUAL,
    "boundedness":                  RelationType.TENSOR_STAT_BOUND,
    "index_consistency":            RelationType.TENSOR_STAT_BOUND,
    "gradient_flow":                RelationType.TENSOR_STAT_BOUND,
    "reference_equivalence":        RelationType.PAYLOAD_FIELD_COMPARE,
    "dtype_preservation":           RelationType.PAYLOAD_FIELD_COMPARE,
    "conservation":                 RelationType.PAYLOAD_FIELD_COMPARE,
    "value_scaling_consistency":    RelationType.PAYLOAD_FIELD_COMPARE,
    "count_frequency_match":        RelationType.PAYLOAD_FIELD_COMPARE,
    "update_effectiveness":         RelationType.PAYLOAD_FIELD_COMPARE,
    "copy_consistency":             RelationType.PAYLOAD_FIELD_COMPARE,
    "ordering":                     RelationType.CROSS_STEP_MONOTONIC,
    "monotonicity":                 RelationType.CROSS_STEP_MONOTONIC,
    "restoration_after_reload":     RelationType.CONDITIONAL_CHECK,
    "config_effectiveness":         RelationType.CONDITIONAL_CHECK,
    "exclusivity":                  RelationType.STRUCTURAL_PRESENCE,
    "sharding_layout_consistency":  RelationType.STRUCTURAL_PRESENCE,
    "structural_integrity":         RelationType.STRUCTURAL_PRESENCE,
}

BY_ID = {t.template_id: t for t in catalog_templates()}
LEGAL = {r.value for r in RelationType}


def parse_with_rescue(text: str):
    """A-arm parse, but recover items whose relation_type is a catalog
    relation_operator by mapping it through the template's operator."""
    blob = None
    for probe in (text.strip(),
                  text[text.find("{"):text.rfind("}") + 1]
                  if "{" in text and "}" in text else ""):
        if not probe:
            continue
        try:
            b = json.loads(probe)
        except Exception:
            continue
        if isinstance(b, dict) and isinstance(b.get("hypotheses"), list):
            blob = b
            break
    if blob is None:
        return [], 0
    out, rescued = [], 0
    for h in blob["hypotheses"]:
        if not isinstance(h, dict):
            continue
        rt_raw = str(h.get("relation_type"))
        tid = h.get("catalog_template_id")
        rt = None
        if rt_raw in LEGAL:
            rt = RelationType(rt_raw)
        elif tid in BY_ID:
            # the LLM echoed the template's operator — map it back
            rt = OPERATOR_TO_RELATION.get(BY_ID[tid].relation_operator)
            if rt is None:
                rt = OPERATOR_TO_RELATION.get(rt_raw)
            if rt is not None:
                rescued += 1
        if rt is None:
            continue
        try:
            out.append(Hypothesis(
                relation_type=rt,
                catalog_template_id=tid if tid in BY_ID else None,
                entities=list(h.get("entities", [])),
                dimensions=list(h.get("dimensions", [])),
                scope_hint=h.get("scope_hint", {}),
                rationale=h.get("rationale", ""),
            ))
        except Exception:
            continue
    return out, rescued


def main():
    recs = [json.loads(l) for l in (HERE / "l1_raw.jsonl").read_text().splitlines()
            if l.strip()]
    recs = [r for r in recs if not r.get("error") and r["arm"] == "A"]
    print(f"replaying {len(recs)} cached A-arm responses with operator rescue")

    stores, schemas = {}, {}
    for fw, p in HEALTHY_PER_FW.items():
        stores[fw] = TraceStore(str(p))
        schemas[fw] = schema_introspect(stores[fw])
    l4 = PatternGuidedFilterLLM(deployed_rule_count=32)

    agg = defaultdict(lambda: {"L1": 0, "L2": 0, "L3": 0, "L4": 0,
                               "rescued": 0, "survivors": set()})
    for i, r in enumerate(recs, 1):
        key = r["rep"]
        a = agg[key]
        fw = r["framework"]
        hyps, resc = parse_with_rescue(r["response"])
        a["L1"] += len(hyps)
        a["rescued"] += resc
        preds = []
        for h in hyps:
            try:
                preds.extend(enumerate_predicates(h, schemas[fw]))
            except Exception:
                pass
        a["L2"] += len(preds)
        l3 = []
        for p in preds:
            try:
                if validate_against_healthy(p, [stores[fw]]).accepted:
                    l3.append(p)
            except Exception:
                pass
        a["L3"] += len(l3)
        if l3:
            for d in filter_predicates(l3, llm_client=l4):
                if d.keep:
                    a["L4"] += 1
                    a["survivors"].add(d.predicate_id)
        if i % 100 == 0:
            print(f"  {i}/{len(recs)}", flush=True)
    for s in stores.values():
        s.close()

    rows = []
    for rep in sorted(agg):
        a = agg[rep]
        rows.append({"rep": rep, "L1": a["L1"], "L2": a["L2"], "L3": a["L3"],
                     "L4": a["L4"], "rescued": a["rescued"],
                     "yield": a["L4"] / a["L2"] if a["L2"] else 0,
                     "distinct_survivors": len(a["survivors"]),
                     "survivors": sorted(a["survivors"])})
    (HERE / "arm_a_plus_results.json").write_text(json.dumps(rows, indent=2))

    y = [r["yield"] for r in rows]
    print("\n=== ARM A+ (catalog, with operator->relation rescue) ===")
    for r in rows:
        print(f"  rep{r['rep']}: L1={r['L1']} L2={r['L2']} L3={r['L3']} "
              f"L4={r['L4']} rescued={r['rescued']} yield={r['yield']:.4f} "
              f"distinct={r['distinct_survivors']}")
    print(f"\nA+ yield median {st.median(y):.4f} range [{min(y):.4f},{max(y):.4f}]")
    print(f"A+ distinct survivors (union): "
          f"{len(set().union(*[set(r['survivors']) for r in rows]))}")


if __name__ == "__main__":
    main()
