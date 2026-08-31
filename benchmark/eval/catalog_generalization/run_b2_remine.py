"""Arm B2 — free-form re-mine per held-out bug (the null hypothesis).

Tests: "you don't need a frozen catalog; just re-run a free-form LLM on each
new component." For every held-out bug a free-form proposer sees the framework,
component category and observable trace surface — but NOT the root cause — and
proposes up to 5 runtime invariants. The same coverage judge then scores whether
any of that bug's own fresh proposals would catch it.

This gives free-form a per-bug custom proposal the frozen catalog never gets, so
B2 is an upper bound on free-form coverage. It is also handed the annotated
required_trace_fields, which leak information toward the bug — deliberately
generous to B2, so that a catalog win is conservative.

Cost, unlike arms A/B1, scales with the number of new bugs: one propose call per
held-out bug. That cost asymmetry is itself a dependent variable.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "catalog_ablation"))

from deepseek_client import DeepSeekClient
from judge import judge_coverage

_lock = threading.Lock()

PROPOSER_SYSTEM = (
    "You are auditing an LLM training framework component for silent errors. "
    "Given the framework, the component category, and the runtime trace fields "
    "available, propose up to 5 runtime invariants that, if violated on an "
    "execution trace, would reveal a silent bug in this component. You do NOT "
    "know the specific bug — propose the invariants a careful auditor would "
    "check for this kind of component. Each needs a short name and a one-line "
    "statement of what must hold.\n"
    'Reply with strict JSON: {"invariants": [{"id": "P1", "name": "...", '
    '"statement": "..."}, ...]}')


def propose(bug: dict, client: DeepSeekClient):
    user = (f"Framework: {bug['framework']}\n"
            f"Component category: {bug['category']}\n"
            f"Annotator invariant_type hint: {bug['invariant_type']}\n"
            f"Available runtime trace fields: {bug['required_trace_fields']}\n\n"
            f"Propose up to 5 runtime invariants for this component.")
    resp = client(PROPOSER_SYSTEM, user, max_tokens=8192)
    for probe in (resp.strip(),
                  resp[resp.find("{"):resp.rfind("}") + 1]
                  if "{" in resp and "}" in resp else ""):
        if not probe:
            continue
        try:
            b = json.loads(probe)
        except Exception:
            continue
        if isinstance(b, dict) and isinstance(b.get("invariants"), list):
            tax = []
            for i, inv in enumerate(b["invariants"], 1):
                if isinstance(inv, dict):
                    tax.append({"id": inv.get("id") or f"P{i}",
                                "name": (inv.get("name", "") + ": "
                                         + inv.get("statement", "")).strip(": ")})
            return tax
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cov_B2.jsonl")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    heldout = json.loads((HERE / "heldout_bugs.json").read_text())
    out_path = HERE / args.out
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["bug_id"])
            except Exception:
                pass
    todo = [b for b in heldout if b["bug_id"] not in done]
    print(f"held-out {len(heldout)} | done {len(done)} | to run {len(todo)}")

    prop_client = DeepSeekClient(temperature=1.0)
    judge_client = DeepSeekClient(temperature=0.0)

    def work(bug):
        tax = propose(bug, prop_client)
        if not tax:
            rec = {"bug_id": bug["bug_id"], "framework": bug["framework"],
                   "tier_field": bug["tier_field"], "covered": False,
                   "matched_id": "NO_PROPOSAL", "why": "proposer returned nothing",
                   "n_proposed": 0}
        else:
            verdict, _ = judge_coverage(bug, tax, judge_client)
            rec = {"bug_id": bug["bug_id"], "framework": bug["framework"],
                   "tier_field": bug["tier_field"], **verdict,
                   "n_proposed": len(tax)}
        with _lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(work, todo):
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)

    recs = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    cov = sum(1 for r in recs if r["covered"])
    print(f"\nCOVERAGE B2 (re-mine): {cov}/{len(recs)} = {cov/len(recs)*100:.1f}%")
    print(f"propose calls={prop_client.n_calls} judge calls={judge_client.n_calls}")
    print(f"tokens: propose in/out {prop_client.prompt_tokens}/"
          f"{prop_client.completion_tokens}, judge in/out "
          f"{judge_client.prompt_tokens}/{judge_client.completion_tokens}")


if __name__ == "__main__":
    main()
