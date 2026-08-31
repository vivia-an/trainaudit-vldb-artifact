"""Validate extraction JSONL outputs: completeness, id coverage, field presence."""
import json
from pathlib import Path

HERE = Path(__file__).parent
REQUIRED = ["case_id", "semantic_objects", "expected_relation", "relation_statement",
            "observable_signal", "topology_scope", "validity_precondition",
            "training_phase", "evidence_from_issue", "evidence_from_fix", "confidence"]


def load_jsonl(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def check(tag, files, input_json, expect_n):
    ids_expected = [c["bug_id"] for c in json.load(open(HERE / "inputs" / input_json))]
    rows = []
    for f in files:
        p = HERE / f
        if not p.exists():
            print(f"[{tag}] MISSING file {f}")
            continue
        rows += load_jsonl(p)
    got = [r.get("case_id") for r in rows]
    dup = {i for i in got if got.count(i) > 1}
    missing_fields = [(r.get("case_id"), k) for r in rows for k in REQUIRED if k not in r or r[k] in (None, "")]
    miss = set(ids_expected[:expect_n] if expect_n == len(ids_expected) else ids_expected) - set(got)
    extra = set(got) - set(ids_expected)
    print(f"[{tag}] rows={len(rows)} expected={expect_n} dup={sorted(dup)} "
          f"missing_ids={sorted(miss) if len(miss) < 20 else len(miss)} extra={sorted(extra)} "
          f"missing_fields={missing_fields[:10]}")
    return rows


if __name__ == "__main__":
    check("seed", [f"seed/extract_e1_p{i}.jsonl" for i in range(1, 5)],
          "seed_cases.json", 128)
    check("dev", [f"development/extract_e1_p{i}.jsonl" for i in range(1, 7)],
          "development_cases.json", 186)
    check("e2", [f"analysis/extract_e2_p{i}.jsonl" for i in range(1, 4)],
          "e2_cases.json", 79)
