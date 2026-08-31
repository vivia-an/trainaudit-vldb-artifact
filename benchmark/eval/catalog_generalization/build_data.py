"""S5b Generalization — build the freeze/held-out split and taxonomy A.

Split (temporal, frozen-before-seeing by construction):
  freeze pool = 128 original bugs (bug_id without 'NEW', excluding the B* smoke set)
  held-out    = 249 '-NEW' bugs discovered in the later expansion round

Taxonomy A = the 35 frozen catalog templates. The catalog predates the '-NEW'
bugs, so testing whether it covers them is a genuine generalization test.

Emits:
  freeze_bugs.json, heldout_bugs.json  (the bug descriptions the judge reads)
  taxonomy_catalog.json                (arm A: 35 templates)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "trainaudit"))

from trainaudit.catalog import catalog_templates

ANNOT = REPO / "benchmark/eval/v2_full/annotations_392_v2.json"

# Fields the coverage judge is allowed to read about each bug.
BUG_FIELDS = ("bug_id", "framework", "category", "invariant_type",
              "required_trace_fields", "check_stage", "tier_field", "rationale")


def bug_record(x):
    return {k: x.get(k) for k in BUG_FIELDS}


def main():
    annots = json.loads(ANNOT.read_text())["annotations"]
    freeze, heldout = [], []
    for x in annots:
        bid = x["bug_id"]
        if "NEW" in bid:
            heldout.append(bug_record(x))
        elif bid.startswith("B") and bid[1:].isdigit():
            continue  # B* smoke set — neither freeze nor held-out
        else:
            freeze.append(bug_record(x))

    (HERE / "freeze_bugs.json").write_text(json.dumps(freeze, indent=2))
    (HERE / "heldout_bugs.json").write_text(json.dumps(heldout, indent=2))

    taxonomy = [{"id": t.template_id, "name": t.name,
                 "relation_operator": t.relation_operator}
                for t in catalog_templates()]
    (HERE / "taxonomy_catalog.json").write_text(json.dumps(taxonomy, indent=2))

    print(f"freeze pool : {len(freeze)} bugs")
    print(f"held-out    : {len(heldout)} bugs")
    print(f"taxonomy A  : {len(taxonomy)} catalog templates")


if __name__ == "__main__":
    main()
