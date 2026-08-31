#!/usr/bin/env python3
"""Offline smoke test: toy trace → SQL verifier → catalog/FSM/Accept/healthy → prune."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE / "agents"
sys.path.insert(0, str(AGENTS))
sys.path.insert(0, str(HERE))


def main() -> int:
    from make_toy_trace import build
    from example_verifier import T01_SQL, verify
    from topology_prune import prune_library
    from paper_accept_gate import AcceptContext, check_accept, discover_healthy_dbs
    from catalog_picker import get_template
    from fsm_stages import MiningFSM

    toy = HERE / "data" / "toy_merged.db"
    buggy = HERE / "data" / "toy_buggy.db"
    build(toy, buggy=False)
    build(buggy, buggy=True)

    clean_v = verify(toy, T01_SQL)
    bug_v = verify(buggy, T01_SQL)
    assert not clean_v, clean_v
    assert bug_v, "buggy toy must fire T01"
    print("[smoke] verifier clean=0 buggy=%d OK" % len(bug_v))

    t01 = get_template("T01")
    assert t01 and t01.get("id") == "T01"
    print("[smoke] catalog T01=%s OK" % t01.get("name"))

    fsm = MiningFSM("S1")
    for st in ("S2", "S3", "S4", "S5"):
        fsm.advance(st)
    print("[smoke] FSM → %s OK" % fsm.stage)

    os.environ["SDC_PAPER_ALIGN"] = "1"
    os.environ["SDC_HEALTHY_DBS"] = str(toy)
    os.environ.pop("SDC_ALLOW_HEALTHY_DEFER", None)
    sample = {
        "name": "t01_layernorm_replica",
        "logic": "layernorm cksum equal across ranks",
        "description": "schema witness cksum equality",
        "confidence": 0.95,
        "template_id": "T01",
        "applicable_conditions": {"tp": ">1", "stage": "model-after-optimizer-step"},
    }
    ctx = AcceptContext(ce_verify_count=2, template_id="T01")
    r = check_accept(sample, ctx)
    assert r.ok and r.healthy == "pass", r
    print("[smoke] Accept+healthy pass OK")

    os.environ["SDC_HEALTHY_DBS"] = str(buggy)
    r2 = check_accept(sample, AcceptContext(ce_verify_count=2, template_id="T01"))
    assert not r2.ok and r2.healthy == "fail", r2
    print("[smoke] Accept rejects buggy healthy FP OK")

    lib = {
        "t01": {"pi_topo": {"tp": ">1"}, "logic": T01_SQL},
        "need_dp": {"pi_topo": {"dp": ">1"}, "logic": "SELECT 1 WHERE 1=0"},
    }
    active = prune_library(lib, {"tp": 2, "dp": 1})
    assert "t01" in active and "need_dp" not in active
    print("[smoke] topology_prune OK")

    print("PASS: core_algo smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
