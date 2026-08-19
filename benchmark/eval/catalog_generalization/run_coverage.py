"""Run the coverage judge for one taxonomy over all held-out bugs.

Usage:
    DEEPSEEK_API_KEY=... python run_coverage.py --taxonomy taxonomy_catalog.json --out cov_A.jsonl [--workers 32]

Resumable: skips bugs already judged in --out.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="heldout_bugs.json")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    taxonomy = json.loads((HERE / args.taxonomy).read_text())
    heldout = json.loads((HERE / args.heldout).read_text())
    out_path = HERE / args.out

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["bug_id"])
            except Exception:
                pass
    todo = [b for b in heldout if b["bug_id"] not in done]
    print(f"taxonomy {args.taxonomy} ({len(taxonomy)} entries) | "
          f"held-out {len(heldout)} | done {len(done)} | to run {len(todo)}")

    client = DeepSeekClient(temperature=0.0)

    def work(bug):
        verdict, raw = judge_coverage(bug, taxonomy, client)
        rec = {"bug_id": bug["bug_id"], "framework": bug["framework"],
               "tier_field": bug["tier_field"], **verdict}
        with _lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(work, todo):
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)

    # summary
    recs = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    cov = sum(1 for r in recs if r["covered"])
    print(f"\nCOVERAGE {args.taxonomy}: {cov}/{len(recs)} = {cov/len(recs)*100:.1f}%")
    print(f"judge calls={client.n_calls} tokens in/out "
          f"{client.prompt_tokens}/{client.completion_tokens}")


if __name__ == "__main__":
    main()
