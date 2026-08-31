"""Build arm B1's frozen taxonomy: a free-form invariant taxonomy induced by
open-coding, WITHOUT the catalog, then size-matched to the catalog's 35 entries.

Source material = the free-form (no-catalog) L1 proposals already generated for
the ablation (catalog_ablation/l1_raw.jsonl, arm B). These are framework-source
-derived and bug-agnostic — the same footing as the catalog, and frozen before
the held-out '-NEW' bugs. We cluster their rationales into canonical invariant
types via an LLM open-coding pass and keep the 35 most frequent, so B1 and the
catalog are compared at equal taxonomy size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "catalog_ablation"))

from deepseek_client import DeepSeekClient

L1_RAW = HERE.parent / "catalog_ablation/l1_raw.jsonl"
TARGET_SIZE = 35

INDUCE_SYSTEM = (
    "You are inducing a canonical taxonomy of runtime invariant types for LLM "
    "training frameworks, by open-coding a list of proposed invariants. Merge "
    "duplicates and near-duplicates into canonical types. Each canonical type "
    "gets a short kebab-case name and a one-line definition. Return the types "
    "sorted by how many input proposals they subsume, most frequent first.\n"
    'Reply with strict JSON: {"types": [{"name": "...", "definition": "...", '
    '"count": <int>}, ...]}')


def load_freeform_rationales():
    import importlib
    sys.path.insert(0, str(Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025/trainaudit")))
    l1 = importlib.import_module("trainaudit.mining.layer1_hypothesis")
    rats = []
    for line in L1_RAW.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or r["arm"] != "B":
            continue
        hyps = l1._parse_hypothesis_response(r["response"], use_catalog=False)
        for h in hyps:
            rt = h.relation_type.value
            rat = (h.rationale or "").strip()
            if rat:
                rats.append(f"[{rt}] {rat}")
    return rats


def main():
    rats = load_freeform_rationales()
    # dedup identical strings to keep the induction prompt bounded
    uniq = sorted(set(rats))
    print(f"free-form rationales: {len(rats)} total, {len(uniq)} unique")

    client = DeepSeekClient(temperature=0.0)
    # induce in one pass over a bounded sample if very large
    sample = uniq[:600]
    listing = "\n".join(f"  - {s[:200]}" for s in sample)
    user = (f"Open-code these {len(sample)} proposed invariants into canonical "
            f"types:\n{listing}\n\nReturn the canonical taxonomy as JSON.")
    resp = client(INDUCE_SYSTEM, user, max_tokens=8192)

    types = []
    for probe in (resp.strip(),
                  resp[resp.find("{"):resp.rfind("}") + 1]
                  if "{" in resp and "}" in resp else ""):
        if not probe:
            continue
        try:
            b = json.loads(probe)
        except Exception:
            continue
        if isinstance(b, dict) and isinstance(b.get("types"), list):
            types = b["types"]
            break
    if not types:
        print("FAILED to induce taxonomy; raw response head:\n", resp[:500])
        return

    types = sorted(types, key=lambda t: -int(t.get("count", 0)))[:TARGET_SIZE]
    taxonomy = [{"id": f"F{i:02d}", "name": t.get("name", ""),
                 "definition": t.get("definition", ""),
                 "count": t.get("count", 0)}
                for i, t in enumerate(types, 1)]
    (HERE / "taxonomy_freeform.json").write_text(json.dumps(taxonomy, indent=2))
    print(f"induced free-form taxonomy: {len(taxonomy)} types "
          f"(target {TARGET_SIZE})")
    for t in taxonomy[:10]:
        print(f"  {t['id']} {t['name']} (subsumes {t['count']})")


if __name__ == "__main__":
    main()
