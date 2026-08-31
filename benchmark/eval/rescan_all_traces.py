"""Re-scan all novel_hunt traces with the updated rule corpus (now 26 rules
including T0-module-grad-output-alive). Report any NEW fire pattern beyond
what each trace's original rule_results.json shows.
"""
import json
import sys
from pathlib import Path

_REPO = Path("/volume/qscai/cqs/workspace/paper/sdc_llm_icml_2025")
sys.path.insert(0, str(_REPO / "trainaudit"))

import trainaudit
from trainaudit.store import TraceStore
from trainaudit.rules import all_rules


HUNT_DIR = _REPO / "benchmark" / "eval" / "hunt_log" / "novel_hunt"


def main():
    traces = sorted(HUNT_DIR.glob("*/trace_rank0.duckdb"))
    print(f"Found {len(traces)} traces to rescan")
    print(f"Rule corpus: {len(all_rules())} rules")

    new_signals = []
    for tp in traces:
        config_name = tp.parent.name
        store = TraceStore(str(tp))
        store.flush()
        # Run all rules
        results = []
        for r in all_rules():
            try:
                res = r.check(store.conn)
                results.append((r.rule_id, res))
            except Exception as e:
                results.append((r.rule_id, None))
        fired = [(rid, res) for rid, res in results
                 if res is not None and res.violated]
        # Compare with old rule_results.json
        old_path = tp.parent / "rule_results.json"
        old_fired = set()
        if old_path.exists():
            old = json.loads(old_path.read_text())
            old_fired = set(s["rule_id"] for s in old if s.get("violated"))

        new_fires = [(rid, res) for rid, res in fired if rid not in old_fired]
        if new_fires:
            print(f"\n=== {config_name} ===")
            for rid, res in new_fires:
                n_v = (len(res.evidence.get("violation_event_ids", []))
                       if res.evidence else 0)
                print(f"  NEW FIRE: {rid}  ({n_v} violations)")
                print(f"    msg: {res.message[:200]}")
                if res.evidence and res.evidence.get("sample"):
                    sample = res.evidence["sample"][:2]
                    print(f"    sample: {str(sample)[:300]}")
                new_signals.append({
                    "config": config_name,
                    "rule_id": rid,
                    "n_violations": n_v,
                    "msg": res.message[:200],
                    "sample": (res.evidence.get("sample") or [])[:2]
                              if res.evidence else [],
                })

    print(f"\n{'='*80}\nSummary: {len(new_signals)} new fire(s) across "
          f"{len(traces)} traces\n{'='*80}")
    if new_signals:
        for ns in new_signals:
            print(f"  [{ns['config']}] {ns['rule_id']}: {ns['n_violations']} ev")
    out = _REPO / "benchmark" / "eval" / "rescan_all_traces_result.json"
    out.write_text(json.dumps(new_signals, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
