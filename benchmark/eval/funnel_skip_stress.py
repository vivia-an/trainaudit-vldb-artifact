"""S4 Step 2/3: stress test L2 (skip-L3) and L3-passed (skip-L4) candidates on
cross-validation clean traces.

We rebuild the A1 mining funnel up to L3 (deterministically) for each of the 4
frameworks and produce three predicate cohorts per framework:

  * L2_full       — every L2 enumeration (~5334 total across 4 fws)
  * L3_passed     — L2 predicates that pass `validate_against_healthy` on the
                    *training* healthy trace (~3436 total)
  * L4_kept       — L3-passed predicates that the PatternGuidedFilterLLM keeps
                    (~357 total)

Each cohort is then **replayed** on *cross-validation* clean traces drawn from
`benchmark/sweep/_runs/` and `benchmark/fault_injection/_runs/FAULT-000-*` —
disjoint from the L3 training trace per SPEC §3.3.

Output:
  benchmark/eval/funnel_skip_l3_results.csv  (Step 2 — sub-sample 50/L2 cohort)
  benchmark/eval/funnel_skip_l4_results.csv  (Step 3 — full L3-passed cohort)

The script is a one-shot reproduction.
"""
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
sys.path.insert(0, str(REPO / "trainaudit"))
sys.path.insert(0, str(REPO / "benchmark/eval/rebuttal_v1/A1_mining_funnel"))

from trainaudit.mining import (
    propose_hypotheses,
    enumerate_predicates,
    schema_introspect,
    validate_against_healthy,
    filter_predicates,
)
from trainaudit.store import TraceStore
from pattern_guided_llm import PatternGuidedLLM, PatternGuidedFilterLLM


# --- paths --------------------------------------------------------------------

HEALTHY_TRAIN_PER_FW = {
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

# Cross-validation clean traces (must be *disjoint* from L3 training traces).
# We pick two independent clean traces:
#   - sweep `dense_190M` (Megatron 190M, different seed, same framework)
#   - fault_injection `FAULT-000-control` (Megatron control, same framework,
#     different config tree)
# Per-framework cross trace assignment (best-effort: pick something distinct
# from HEALTHY_TRAIN_PER_FW[fw]).
CROSS_TRACES = {
    "megatron": [
        REPO / "benchmark/sweep/_runs/dense_190M/trace_rank0.duckdb",
        REPO / "benchmark/fault_injection/_runs/FAULT-000-control/trace_rank0.duckdb",
    ],
    "deepspeed": [
        REPO / "benchmark/sweep/_runs/dense_190M/trace_rank0.duckdb",
        REPO / "benchmark/fault_injection/_runs/FAULT-000-control/trace_rank0.duckdb",
    ],
    "olmo": [
        REPO / "benchmark/sweep/_runs/dense_190M/trace_rank0.duckdb",
        REPO / "benchmark/fault_injection/_runs/FAULT-000-control/trace_rank0.duckdb",
    ],
    "olmo_core": [
        REPO / "benchmark/sweep/_runs/dense_190M/trace_rank0.duckdb",
        REPO / "benchmark/fault_injection/_runs/FAULT-000-control/trace_rank0.duckdb",
    ],
}

SEED = 4242


# --- build cohorts ------------------------------------------------------------


def build_cohorts(fw: str):
    """Return (l2_list, l3_list, l4_list) of Predicate objects for framework fw.
    Each list element is (pattern_id, src_path, Predicate).
    """
    seed_files = json.loads((REPO / "benchmark/eval/rebuttal_v1/A1_mining_funnel/seed_files.json").read_text())
    if fw not in seed_files:
        return [], [], []

    cells = seed_files[fw]
    train_path = HEALTHY_TRAIN_PER_FW[fw]
    if not train_path.exists():
        print(f"WARN: {fw} train trace missing {train_path}")
        return [], [], []
    train_store = TraceStore(str(train_path))
    schema = schema_introspect(train_store)

    l1_llm = PatternGuidedLLM()
    l4_llm = PatternGuidedFilterLLM(deployed_rule_count=32)

    l2_all = []
    l3_all = []

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
            for h in hyps:
                try:
                    preds = enumerate_predicates(h, schema)
                except Exception:
                    preds = []
                for p in preds:
                    l2_all.append((pid, src_rel, p))
                    try:
                        r = validate_against_healthy(p, [train_store])
                    except Exception:
                        continue
                    if r.accepted:
                        l3_all.append((pid, src_rel, p))

    # L4: apply PatternGuidedFilterLLM on l3_all
    l3_preds_only = [t[2] for t in l3_all]
    decisions = filter_predicates(l3_preds_only, llm_client=l4_llm) if l3_preds_only else []
    keep_ids = {d.predicate_id for d in decisions if d.keep}
    l4_all = [t for t in l3_all if t[2].id in keep_ids]

    train_store.close()
    return l2_all, l3_all, l4_all


# --- replay -------------------------------------------------------------------


def replay_cohort(cohort, trace_path: Path, label: str):
    """Run every predicate in cohort against trace_path; return list of dicts:
    {predicate_id, pattern, n_violations, fired}.
    """
    if not trace_path.exists():
        print(f"WARN: trace missing {trace_path}")
        return []
    store = TraceStore(str(trace_path))
    out = []
    for (pid_pattern, src_rel, p) in cohort:
        try:
            r = validate_against_healthy(p, [store])
            n_viol = r.n_violations_per_trace[0] if r.n_violations_per_trace else 0
            fired = (not r.accepted)
            error = r.rejection_reason if (fired and "compile/run error" in r.rejection_reason) else ""
        except Exception as e:
            n_viol = 0
            fired = False
            error = f"{type(e).__name__}: {e}"
        out.append({
            "label": label,
            "pattern": pid_pattern,
            "predicate_id": p.id,
            "n_violations": n_viol,
            "fired": fired,
            "error": error,
        })
    store.close()
    return out


# --- step 2 (skip L3) ---------------------------------------------------------


def step2_skip_l3(out_csv: Path):
    """Sub-sample 50 L2 candidates per framework; replay on cross traces."""
    random.seed(SEED)
    rows = []
    n_total_per_fw = {}
    sampled_per_fw = {}
    fp_counts_per_fw = defaultdict(lambda: defaultdict(int))  # fw -> trace -> fp

    for fw in ["megatron", "deepspeed", "olmo", "olmo_core"]:
        print(f"[Step 2] building cohorts for {fw}...")
        l2, l3, l4 = build_cohorts(fw)
        n_total_per_fw[fw] = len(l2)
        if not l2:
            sampled_per_fw[fw] = 0
            continue
        # Sub-sample 50 (or full if smaller) per SPEC §3.2 fallback.
        k = min(50, len(l2))
        sampled = random.sample(l2, k)
        sampled_per_fw[fw] = k
        for tr in CROSS_TRACES[fw]:
            print(f"[Step 2] replay {fw} L2 sample={k} on {tr.name}...")
            res = replay_cohort(sampled, tr, label=f"L2_sample/{fw}/{tr.parent.name}")
            for r in res:
                r["framework"] = fw
                r["trace"] = str(tr.relative_to(REPO))
                rows.append(r)
                if r["fired"]:
                    fp_counts_per_fw[fw][tr.parent.name] += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("framework,trace,pattern,predicate_id,n_violations,fired,error\n")
        for r in rows:
            err = (r.get("error") or "").replace(",", ";").replace("\n", " ")
            f.write(f"{r['framework']},{r['trace']},{r['pattern']},{r['predicate_id']},"
                    f"{r['n_violations']},{int(r['fired'])},\"{err[:120]}\"\n")

    print("\n=== Step 2 summary (FP = fired on cross-validation clean trace) ===")
    total_fp = 0
    total_eval = 0
    for fw, traces in fp_counts_per_fw.items():
        for tr, n_fp in traces.items():
            print(f"  {fw} on {tr}: {n_fp}/{sampled_per_fw[fw]} fired")
            total_fp += n_fp
            total_eval += sampled_per_fw[fw]
    if total_eval:
        rate = total_fp / total_eval
        # 95% Wilson CI for proportion
        z = 1.96
        denom = 1 + z * z / total_eval
        ctr = (rate + z * z / (2 * total_eval)) / denom
        half = z * ((rate * (1 - rate) / total_eval + z * z / (4 * total_eval ** 2)) ** 0.5) / denom
        lo, hi = max(0, ctr - half), min(1, ctr + half)
        print(f"\n  Pooled FP rate = {total_fp}/{total_eval} = {rate:.3f}  (95% CI: {lo:.3f}-{hi:.3f})")
    return n_total_per_fw, sampled_per_fw, fp_counts_per_fw


# --- step 3 (skip L4) ---------------------------------------------------------


def step3_skip_l4(out_csv: Path):
    """Run every L3-passed predicate on cross-validation traces (full cohort, no
    sub-sampling). Compare against L4_kept cohort to show how much
    adversarial-pass filtering removes."""
    rows = []
    fp_counts = defaultdict(lambda: defaultdict(int))  # (cohort, fw) -> trace -> count
    cohort_sizes = defaultdict(dict)

    for fw in ["megatron", "deepspeed", "olmo", "olmo_core"]:
        print(f"[Step 3] building cohorts for {fw}...")
        l2, l3, l4 = build_cohorts(fw)
        cohort_sizes["L3_passed"][fw] = len(l3)
        cohort_sizes["L4_kept"][fw]   = len(l4)
        for tr in CROSS_TRACES[fw]:
            for label, cohort in (("L3_passed", l3), ("L4_kept", l4)):
                if not cohort:
                    continue
                print(f"[Step 3] replay {fw} {label}={len(cohort)} on {tr.name}...")
                res = replay_cohort(cohort, tr, label=f"{label}/{fw}/{tr.parent.name}")
                for r in res:
                    r["framework"] = fw
                    r["cohort"] = label
                    r["trace"] = str(tr.relative_to(REPO))
                    rows.append(r)
                    if r["fired"]:
                        fp_counts[(label, fw)][tr.parent.name] += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("cohort,framework,trace,pattern,predicate_id,n_violations,fired,error\n")
        for r in rows:
            err = (r.get("error") or "").replace(",", ";").replace("\n", " ")
            f.write(f"{r['cohort']},{r['framework']},{r['trace']},{r['pattern']},"
                    f"{r['predicate_id']},{r['n_violations']},{int(r['fired'])},\"{err[:120]}\"\n")

    print("\n=== Step 3 summary (FP = fired on cross-validation clean trace) ===")
    for (label, fw), traces in fp_counts.items():
        n = cohort_sizes[label][fw]
        for tr, n_fp in traces.items():
            print(f"  {label} {fw} on {tr}: {n_fp}/{n} fired ({100*n_fp/max(1,n):.1f}%)")
    return cohort_sizes, fp_counts


# --- main --------------------------------------------------------------------


if __name__ == "__main__":
    out_dir = REPO / "benchmark/eval"
    step2_skip_l3(out_dir / "funnel_skip_l3_results.csv")
    print()
    step3_skip_l4(out_dir / "funnel_skip_l4_results.csv")
