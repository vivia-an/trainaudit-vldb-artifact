"""A1 Mining Funnel — runs 4-layer pipeline per (framework × pattern)
cell and aggregates L1/L2/L3/L4/Deployed counts.

Pipeline:
  L1: PatternGuidedLLM proposes Hypothesis blobs from seed source
  L2: enumerate_predicates deterministically expands to Predicate list
  L3: validate_against_healthy on the framework's clean DuckDB trace
  L4: PatternGuidedFilterLLM keeps/rejects each L3-passed predicate
  Deployed: count entries in trainaudit/trainaudit/rules/T{0,1}*.py

Output: a1_funnel.csv + a1_reject_taxonomy.csv + a1_summary.json
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
sys.path.insert(0, str(REPO / "trainaudit"))
sys.path.insert(0, str(REPO / "benchmark/eval/rebuttal_v1/A1_mining_funnel"))

from trainaudit.mining import (propose_hypotheses, enumerate_predicates,
                                schema_introspect, validate_against_healthy,
                                filter_predicates)
from trainaudit.store import TraceStore
from pattern_guided_llm import PatternGuidedLLM, PatternGuidedFilterLLM


HEALTHY_PER_FW = {
    "megatron":  REPO / "benchmark/eval/hunt_log/novel_hunt/megatron_clean/trace_rank0.duckdb",
    "deepspeed": REPO / "benchmark/eval/hunt_log/novel_hunt/deepspeed_bf16_only/trace_rank0.duckdb",
    "olmo":      REPO / "benchmark/eval/hunt_log/novel_hunt/olmo_core_baseline/trace_rank0.duckdb",
    "olmo_core": REPO / "benchmark/eval/hunt_log/novel_hunt/olmo_core_baseline/trace_rank0.duckdb",
}

FRAMEWORK_ROOTS = {
    "megatron":  REPO / "exp/frameworks/Megatron-LM",
    "deepspeed": REPO / "exp/frameworks/DeepSpeed",
    "olmo":      REPO / "exp/frameworks/OLMo",
    "olmo_core": REPO / "exp/frameworks/OLMo-core",
}


def deployed_rule_count(framework: str) -> int:
    """Count deployed rules whose adapter touches this framework. Since
    rules are framework-agnostic, this is just the total count (32)."""
    py = list((REPO / "trainaudit/trainaudit/rules").glob("T*.py"))
    yaml = list((REPO / "trainaudit/trainaudit/dsl/registry").rglob("*.yaml"))
    return len(py) + len(yaml)


def main():
    seed_files = json.loads((Path(__file__).parent / "seed_files.json").read_text())

    out_per_fw = {}
    reject_reasons = []
    l1_llm = PatternGuidedLLM()
    l4_llm = PatternGuidedFilterLLM(deployed_rule_count=32)

    for fw, cells in seed_files.items():
        healthy_path = HEALTHY_PER_FW[fw]
        if not healthy_path.exists():
            print(f"WARN: {fw} healthy trace missing: {healthy_path}")
            continue
        healthy_store = TraceStore(str(healthy_path))
        try:
            schema = schema_introspect(healthy_store)
        except Exception as e:
            print(f"WARN: {fw} schema_introspect failed: {e}")
            continue

        l1_total = 0
        l2_total = 0
        l3_passed = 0
        l3_rejected = 0
        l4_kept = 0
        l4_rejected = 0
        per_pattern = []

        for cell in cells:
            pid = cell["pattern"]
            for src_rel in cell["files"]:
                src_path = FRAMEWORK_ROOTS[fw] / src_rel
                if not src_path.exists():
                    continue
                try:
                    source = src_path.read_text(errors="replace")[:8000]
                except Exception:
                    continue
                hyps = propose_hypotheses(source, framework=fw, llm_client=l1_llm)
                l1_total += len(hyps)

                predicates = []
                for h in hyps:
                    try:
                        ps = enumerate_predicates(h, schema)
                    except Exception as e:
                        ps = []
                    predicates.extend(ps)
                l2_total += len(predicates)

                l3_pass_local = 0
                for p in predicates:
                    try:
                        r = validate_against_healthy(p, [healthy_store])
                    except Exception as e:
                        l3_rejected += 1
                        reject_reasons.append({
                            "framework": fw, "pattern": pid,
                            "stage": "L3",
                            "predicate_id": getattr(p, "id", "?"),
                            "reason": f"L3 error: {type(e).__name__}",
                        })
                        continue
                    if r.accepted:
                        l3_pass_local += 1
                    else:
                        l3_rejected += 1
                        reject_reasons.append({
                            "framework": fw, "pattern": pid,
                            "stage": "L3",
                            "predicate_id": p.id,
                            "reason": r.rejection_reason or "violates on healthy",
                        })
                l3_passed += l3_pass_local

                # L4 filter on the L3-passed predicates only
                l3_passed_preds = [p for p in predicates
                                    if (validate_against_healthy(p, [healthy_store])).accepted]
                if l3_passed_preds:
                    decisions = filter_predicates(l3_passed_preds, llm_client=l4_llm)
                    for d in decisions:
                        if d.keep:
                            l4_kept += 1
                        else:
                            l4_rejected += 1
                            reject_reasons.append({
                                "framework": fw, "pattern": pid,
                                "stage": "L4",
                                "predicate_id": d.predicate_id,
                                "reason": d.reason,
                            })

                per_pattern.append({
                    "pattern": pid, "seed_file": src_rel,
                    "L1": len(hyps), "L2": len(predicates),
                    "L3_pass": l3_pass_local,
                })

        out_per_fw[fw] = {
            "L1_hypothesis": l1_total,
            "L2_enumerated": l2_total,
            "L3_healthy_passed": l3_passed,
            "L3_rejected": l3_rejected,
            "L4_kept": l4_kept,
            "L4_rejected": l4_rejected,
            "Deployed": deployed_rule_count(fw),
            "per_pattern": per_pattern,
        }
        healthy_store.close()
        print(f"\n=== {fw} ===")
        print(f"  L1 hypotheses       : {l1_total}")
        print(f"  L2 enumerated       : {l2_total}")
        print(f"  L3 healthy-passed   : {l3_passed}  (rejected {l3_rejected})")
        print(f"  L4 adversarial-pass : {l4_kept}    (rejected {l4_rejected})")
        print(f"  Deployed total      : {out_per_fw[fw]['Deployed']}  (cross-framework shared)")

    # Funnel CSV
    out_dir = Path(__file__).parent
    funnel_csv = out_dir / "a1_funnel.csv"
    with funnel_csv.open("w") as f:
        f.write("stage,megatron,deepspeed,olmo,olmo_core\n")
        for key in ["L1_hypothesis","L2_enumerated","L3_healthy_passed","L4_kept","Deployed"]:
            vals = [str(out_per_fw.get(fw, {}).get(key, 0)) for fw in
                    ["megatron","deepspeed","olmo","olmo_core"]]
            label = {"L1_hypothesis": "L1 Hypothesis",
                      "L2_enumerated": "L2 Enumerated",
                      "L3_healthy_passed": "L3 Healthy-pass",
                      "L4_kept": "L4 Adversarial-pass",
                      "Deployed": "Deployed"}[key]
            f.write(label + "," + ",".join(vals) + "\n")
    print(f"\nSaved {funnel_csv}")

    # Reject taxonomy
    reject_csv = out_dir / "a1_reject_taxonomy.csv"
    with reject_csv.open("w") as f:
        f.write("framework,pattern,stage,predicate_id,reason\n")
        for r in reject_reasons:
            f.write(f"{r['framework']},{r['pattern']},{r['stage']},"
                    f"{r['predicate_id']},\"{r['reason'][:200]}\"\n")
    # Reject reason categories
    l3_cats = Counter()
    l4_cats = Counter()
    for r in reject_reasons:
        bucket = (l3_cats if r["stage"] == "L3" else l4_cats)
        rs_lo = r["reason"][:200].lower()
        # Order matters — earlier categories take precedence
        if "compile" in rs_lo or "catalog" in rs_lo or "json_each" in rs_lo:
            bucket["compile-error"] += 1
        elif "fired on" in rs_lo or "healthy" in rs_lo or "violates" in rs_lo:
            bucket["healthy-violation"] += 1
        elif "schema" in rs_lo or "field-missing" in rs_lo:
            bucket["schema-field-missing"] += 1
        elif "π_topo" in r["reason"] or "topo-scope" in rs_lo:
            bucket["no-π_topo-scope"] += 1
        elif "workload" in rs_lo or "absolute" in rs_lo or "constant" in rs_lo or "auto-enum" in rs_lo:
            bucket["workload-specific-constant"] += 1
        elif "hookpoint" in rs_lo:
            bucket["no-applicable-hookpoint"] += 1
        elif "redundant" in rs_lo or "deployed" in rs_lo:
            bucket["redundant-with-existing"] += 1
        else:
            bucket["other"] += 1

    print("\n=== Reject reason taxonomy ===")
    print(f"L3 ({sum(l3_cats.values())} rejects):")
    for cat, n in l3_cats.most_common():
        pct = n / max(1, sum(l3_cats.values())) * 100
        print(f"  {cat}: {n} ({pct:.0f}%)")
    print(f"L4 ({sum(l4_cats.values())} rejects):")
    for cat, n in l4_cats.most_common():
        pct = n / max(1, sum(l4_cats.values())) * 100
        print(f"  {cat}: {n} ({pct:.0f}%)")

    (out_dir / "a1_summary.json").write_text(json.dumps({
        "per_framework": out_per_fw,
        "reject_reason_taxonomy": {"L3": dict(l3_cats), "L4": dict(l4_cats)},
    }, indent=2))
    print(f"\nSaved a1_funnel.csv, a1_reject_taxonomy.csv, a1_summary.json")


if __name__ == "__main__":
    main()
