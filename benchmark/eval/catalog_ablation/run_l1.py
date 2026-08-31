"""S5 Step 3, Phase 1 — run Layer 1 for both arms with a real LLM.

Both arms iterate the *identical* (framework x pattern x seed_file) loop
that `A1_mining_funnel/run_funnel.py` uses to produce the paper's 420, so
the arms stay comparable to the published funnel. The only difference
between arms is `use_catalog`, which selects the system prompt inside
`propose_hypotheses`.

L1 is the expensive, stochastic stage, so it is run once here and its raw
responses cached to l1_raw.jsonl. Phase 2 (run_ablation.py) replays that
cache through the deterministic L2/L3/L4 stages.

Usage:
    DEEPSEEK_API_KEY=... python run_l1.py [--reps 5] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "trainaudit"))
sys.path.insert(0, str(HERE))

from deepseek_client import MODEL, TEMPERATURE, MAX_TOKENS, DeepSeekClient
from trainaudit.mining.layer1_hypothesis import (_SYSTEM_PROMPT,
                                                 _SYSTEM_PROMPT_FREEFORM,
                                                 _build_user_prompt)

SEED_FILES = REPO / "benchmark/eval/rebuttal_v1/A1_mining_funnel/seed_files.json"
FRAMEWORK_ROOTS = {
    "megatron":  REPO / "exp/frameworks/Megatron-LM",
    "deepspeed": REPO / "exp/frameworks/DeepSpeed",
    "olmo":      REPO / "exp/frameworks/OLMo",
    "olmo_core": REPO / "exp/frameworks/OLMo-core",
}
OUT = HERE / "l1_raw.jsonl"

_write_lock = threading.Lock()


def build_tasks(reps: int):
    """One task per (arm, rep, framework, pattern, seed_file) — mirroring
    run_funnel.py's loop exactly."""
    seed_files = json.loads(SEED_FILES.read_text())
    tasks = []
    for arm, use_catalog in (("A", True), ("B", False)):
        for rep in range(reps):
            for fw, cells in seed_files.items():
                for cell in cells:
                    for src_rel in cell["files"]:
                        src_path = FRAMEWORK_ROOTS[fw] / src_rel
                        if not src_path.exists():
                            continue
                        tasks.append({
                            "arm": arm, "use_catalog": use_catalog,
                            "rep": rep, "framework": fw,
                            "pattern": cell["pattern"], "src_rel": src_rel,
                        })
    return tasks


def task_key(t) -> str:
    return f'{t["arm"]}|{t["rep"]}|{t["framework"]}|{t["pattern"]}|{t["src_rel"]}'


def load_done() -> set:
    if not OUT.exists():
        return set()
    done = set()
    for line in OUT.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error") is None:
            done.add(task_key(r))
    return done


def run_one(t, client: DeepSeekClient):
    src_path = FRAMEWORK_ROOTS[t["framework"]] / t["src_rel"]
    source = src_path.read_text(errors="replace")[:8000]
    system = _SYSTEM_PROMPT if t["use_catalog"] else _SYSTEM_PROMPT_FREEFORM
    user = _build_user_prompt(source, t["framework"])
    rec = dict(t)
    try:
        rec["response"] = client(system, user, max_tokens=MAX_TOKENS)
        rec["error"] = None
    except Exception as e:  # noqa: BLE001
        rec["response"] = ""
        rec["error"] = f"{type(e).__name__}: {e}"
    with _write_lock:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = build_tasks(args.reps)
    done = load_done()
    todo = [t for t in tasks if task_key(t) not in done]
    print(f"total tasks {len(tasks)} | already done {len(done)} | "
          f"to run {len(todo)}")
    print(f"model={MODEL} temperature={TEMPERATURE} max_tokens={MAX_TOKENS}")

    client = DeepSeekClient()
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(lambda t: run_one(t, client), todo), 1):
            if rec["error"]:
                n_err += 1
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {n_err} errors, "
                      f"tokens in/out {client.prompt_tokens}/"
                      f"{client.completion_tokens}", flush=True)

    print(f"\nSaved {OUT}")
    print(f"calls={client.n_calls} errors={n_err} "
          f"prompt_tokens={client.prompt_tokens} "
          f"completion_tokens={client.completion_tokens}")


if __name__ == "__main__":
    main()
