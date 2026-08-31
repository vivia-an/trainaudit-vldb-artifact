#!/usr/bin/env python3
"""Recover the compiled SQL for each constraint from the recorded LLM interaction logs.

The constraint libraries ship rule *specifications* — name, type, guards — not SQL. The
translation to SQL is done once per constraint by `SQLAgent`, and every one of those calls
was logged during the runs that produced the paper's numbers. This lifts the SQL back out,
so the deterministic object Sec 4.5 describes ("pi_topo and pi_precond become WHERE
filters, pi_schema the HAVING condition") can be read and executed directly.

    python3 core/extract_generated_sql.py --logs <dir> [--out core/config/generated_sql.json]

Input logs are `llm_interactions_*.log`: pretty-printed JSON records interleaved with plain
log lines. Only `agent_name: "SQLAgent"` records matter; their `response` is a JSON string
carrying an `sql` key.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys

DEC = json.JSONDecoder()


def records(text):
    """Yield every JSON object in the file, skipping the plain log lines between them."""
    i = 0
    n = len(text)
    while True:
        i = text.find('{\n  "', i)
        if i < 0:
            return
        try:
            obj, end = DEC.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            yield obj
        i = end


def sql_of(rec):
    """The generated SQL, from a response that is itself JSON (or fenced JSON)."""
    resp = (rec.get("output") or {}).get("response") if isinstance(rec.get("output"), dict) else None
    resp = resp or rec.get("response")
    if not isinstance(resp, str):
        return None
    body = resp.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if m:
        body = m.group(1).strip()
    try:
        return (json.loads(body) or {}).get("sql")
    except json.JSONDecodeError:
        # Some responses are not valid JSON (unescaped backslashes in the SQL, truncation).
        # Pull the sql value textually and unescape only the sequences we expect.
        m = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', body, re.S)
        if not m:
            return None
        raw = m.group(1)
        for a, b in (("\\n", "\n"), ("\\t", "\t"), ("\\r", "\r"),
                     ('\\"', '"'), ("\\\\", "\\")):
            raw = raw.replace(a, b)
        return raw


def constraint_of(rec):
    """The constraint the SQL was generated for. The prompt embeds its full JSON."""
    inp = rec.get("input") or {}
    prompt = inp.get("user_prompt") if isinstance(inp, dict) else None
    prompt = prompt or rec.get("user_prompt") or ""
    i = prompt.find("{")
    while i >= 0:
        try:
            obj, _ = DEC.raw_decode(prompt, i)
        except ValueError:
            i = prompt.find("{", i + 1)
            continue
        if isinstance(obj, dict) and "name" in obj and "type" in obj:
            return {
                "name": obj.get("name") or "(unnamed)",
                "type": obj.get("type") or "",
                "description": (obj.get("description") or "")[:400],
                "applicable_conditions": obj.get("applicable_conditions") or {},
                "tables": obj.get("tables") or [],
                "spec_logic": obj.get("logic") or "",
            }
        i = prompt.find("{", i + 1)
    return {"name": "(unnamed)", "type": "", "description": "",
            "applicable_conditions": {}, "tables": [], "spec_logic": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="directory holding llm_interactions_*.log")
    ap.add_argument("--out", default="core/config/generated_sql.json")
    ap.add_argument("--limit-logs", type=int, default=0)
    ap.add_argument("--max-variants", type=int, default=3,
                    help="keep at most N SQL variants per rule, most frequent first (0 = all)")
    args = ap.parse_args()

    logs = sorted(pathlib.Path(args.logs).glob("llm_interactions_*.log"))
    if args.limit_logs:
        logs = logs[-args.limit_logs:]
    if not logs:
        sys.exit(f"no llm_interactions_*.log under {args.logs}")

    by_rule = collections.defaultdict(dict)     # constraint -> sql_hash -> entry
    specs = {}                                  # constraint -> its specification
    stats = collections.Counter()
    for lg in logs:
        try:
            text = lg.read_text(errors="replace")
        except OSError:
            stats["unreadable"] += 1
            continue
        for rec in records(text):
            if rec.get("agent_name") != "SQLAgent":
                continue
            stats["sqlagent_records"] += 1
            sql = sql_of(rec)
            if not sql or "select" not in sql.lower():
                stats["no_sql"] += 1
                continue
            spec = constraint_of(rec)
            name = spec["name"]
            specs.setdefault(name, spec)
            h = hashlib.sha256(" ".join(sql.split()).encode()).hexdigest()[:12]
            entry = by_rule[name].setdefault(h, {"sql": sql, "n_occurrences": 0, "logs": []})
            entry["n_occurrences"] += 1
            if lg.name not in entry["logs"]:
                entry["logs"].append(lg.name)
            stats["sql_recovered"] += 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "SQL recovered from recorded SQLAgent interactions; see core/extract_generated_sql.py. "
                "A rule may have several variants because the SQL is generated per run.",
        "n_rules": len(by_rule),
        "rules": {
            name: {
                **specs.get(name, {}),
                "n_variants": len(v),
                "n_generations": sum(e["n_occurrences"] for e in v.values()),
                "variants": sorted(v.values(), key=lambda e: -e["n_occurrences"])[
                    : args.max_variants or None],
            } for name, v in sorted(by_rule.items())
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")

    print(f"scanned {len(logs)} log(s)")
    for k in ("sqlagent_records", "sql_recovered", "no_sql", "unreadable"):
        if stats[k]:
            print(f"  {k:<20}{stats[k]}")
    print(f"  distinct constraints {len(by_rule)}")
    multi = sum(1 for v in by_rule.values() if len(v) > 1)
    print(f"  with >1 SQL variant  {multi}")
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
