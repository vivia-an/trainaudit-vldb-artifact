"""S5 Step 3, Phase 2 — replay cached L1 responses through L2/L3/L4.

Reads l1_raw.jsonl (produced by run_l1.py) and runs the deterministic
downstream stages per (arm, rep). L2/L3/L4 code is byte-identical across
arms — the only thing that differed is the L1 system prompt — so any
funnel difference is attributable to the Pattern Catalog alone.

L4 uses the same catalog-blind `PatternGuidedFilterLLM` the published
funnel used: it decides on hookpoint/field/value and never reads
`catalog_template_id`, so it cannot favour either arm.

Also counts SPEC §9's anomaly classes: hypotheses that parse but cannot
enter L2 (empty entities), and raw items rejected before becoming a
Hypothesis (bad relation_type / bad template id).

Usage:
    python run_ablation.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "trainaudit"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "benchmark/eval/rebuttal_v1/A1_mining_funnel"))

from pattern_guided_llm import PatternGuidedFilterLLM
from trainaudit.mining import (enumerate_predicates, filter_predicates,
                               schema_introspect, validate_against_healthy)
from trainaudit.mining.hypothesis_schema import Hypothesis, RelationType
from trainaudit.mining.layer1_hypothesis import _parse_hypothesis_response
from trainaudit.store import TraceStore

# Same clean traces the published funnel used. SPEC §4.3 names
# benchmark/sweep/_runs/dense_190M, which does not exist in this repo
# (never committed); these are the traces run_funnel.py actually reads.
# Both arms use the identical trace per framework.
HEALTHY_PER_FW = {
    "megatron":  REPO / "benchmark/eval/hunt_log/novel_hunt/megatron_clean/trace_rank0.duckdb",
    "deepspeed": REPO / "benchmark/eval/hunt_log/novel_hunt/deepspeed_bf16_only/trace_rank0.duckdb",
    "olmo":      REPO / "benchmark/eval/hunt_log/novel_hunt/olmo_core_baseline/trace_rank0.duckdb",
    "olmo_core": REPO / "benchmark/eval/hunt_log/novel_hunt/olmo_core_baseline/trace_rank0.duckdb",
}

L1_RAW = HERE / "l1_raw.jsonl"


def raw_item_count(text: str) -> int:
    """How many hypothesis objects the LLM actually emitted, regardless of
    whether they survive Hypothesis construction. Lets us separate 'LLM
    proposed nothing' from 'proposal was unusable'."""
    for probe in (text.strip(),
                  text[text.find("{"):text.rfind("}") + 1]
                  if "{" in text and "}" in text else ""):
        if not probe:
            continue
        try:
            blob = json.loads(probe)
        except Exception:
            continue
        if isinstance(blob, dict) and isinstance(blob.get("hypotheses"), list):
            return len(blob["hypotheses"])
    fence = text.rfind("```json")
    if fence != -1:
        end = text.find("```", fence + 7)
        if end != -1:
            try:
                blob = json.loads(text[fence + 7:end].strip())
                return len(blob.get("hypotheses", []))
            except Exception:
                pass
    return 0


def classify_drops(text: str, use_catalog: bool) -> Counter:
    """Why raw hypothesis items failed to become Hypothesis objects."""
    reasons: Counter = Counter()
    n_raw = 0
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
        return reasons
    for h in blob["hypotheses"]:
        n_raw += 1
        if not isinstance(h, dict):
            reasons["not-an-object"] += 1
            continue
        try:
            RelationType(h.get("relation_type"))
        except Exception:
            reasons["bad-or-missing-relation_type"] += 1
            continue
        try:
            Hypothesis(
                relation_type=RelationType(h["relation_type"]),
                catalog_template_id=(h.get("catalog_template_id")
                                     if use_catalog else None),
                entities=list(h.get("entities", [])),
                dimensions=list(h.get("dimensions", [])),
                scope_hint=h.get("scope_hint", {}),
                rationale=h.get("rationale", ""),
            )
        except Exception as e:  # noqa: BLE001
            reasons[f"rejected-{type(e).__name__}"] += 1
    return reasons


def main():
    records = [json.loads(l) for l in L1_RAW.read_text().splitlines() if l.strip()]
    records = [r for r in records if not r.get("error")]
    print(f"loaded {len(records)} L1 responses")

    schemas = {}
    stores = {}
    for fw, path in HEALTHY_PER_FW.items():
        if not path.exists():
            print(f"FATAL: healthy trace missing for {fw}: {path}")
            return
        stores[fw] = TraceStore(str(path))
        schemas[fw] = schema_introspect(stores[fw])

    l4_llm = PatternGuidedFilterLLM(deployed_rule_count=32)

    # (arm, rep, framework) -> counters
    agg = defaultdict(lambda: {
        "L1_raw_items": 0, "L1_hypothesis": 0, "L2_enumerated": 0,
        "L3_pass": 0, "L4_keep": 0,
        "hyp_empty_entities": 0, "hyp_l2_zero": 0,
        "drops": Counter(), "survivors": [],
    })

    for i, rec in enumerate(records, 1):
        arm, rep, fw = rec["arm"], rec["rep"], rec["framework"]
        use_catalog = rec["use_catalog"]
        key = (arm, rep, fw)
        a = agg[key]
        text = rec["response"]
        schema = schemas[fw]

        a["L1_raw_items"] += raw_item_count(text)
        a["drops"] += classify_drops(text, use_catalog)

        hyps = _parse_hypothesis_response(text, use_catalog=use_catalog)
        a["L1_hypothesis"] += len(hyps)

        predicates = []
        for h in hyps:
            if not h.entities:
                a["hyp_empty_entities"] += 1
            try:
                ps = enumerate_predicates(h, schema)
            except Exception:
                ps = []
            if not ps:
                a["hyp_l2_zero"] += 1
            predicates.extend(ps)
        a["L2_enumerated"] += len(predicates)

        l3_passed = []
        for p in predicates:
            try:
                r = validate_against_healthy(p, [stores[fw]])
            except Exception:
                continue
            if r.accepted:
                l3_passed.append(p)
        a["L3_pass"] += len(l3_passed)

        if l3_passed:
            for d in filter_predicates(l3_passed, llm_client=l4_llm):
                if d.keep:
                    a["L4_keep"] += 1
                    a["survivors"].append(d.predicate_id)

        if i % 100 == 0:
            print(f"  {i}/{len(records)}", flush=True)

    for s in stores.values():
        s.close()

    out = []
    for (arm, rep, fw), a in sorted(agg.items()):
        out.append({
            "arm": arm, "rep": rep, "framework": fw,
            "L1_raw_items": a["L1_raw_items"],
            "L1_hypothesis": a["L1_hypothesis"],
            "L2_enumerated": a["L2_enumerated"],
            "L3_pass": a["L3_pass"],
            "L4_keep": a["L4_keep"],
            "hyp_empty_entities": a["hyp_empty_entities"],
            "hyp_l2_zero": a["hyp_l2_zero"],
            "drops": dict(a["drops"]),
            "survivors": sorted(set(a["survivors"])),
        })
    (HERE / "per_cell_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved per_cell_results.json ({len(out)} cells)")


if __name__ == "__main__":
    main()
