"""Job-start topology prune: keep constraints whose π_topo matches live τ."""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional


def _parse_cmp(expr: str) -> Optional[tuple]:
    s = str(expr).strip()
    m = re.match(r"^(>=|<=|==|=|>|<)\s*(.+)$", s)
    if not m:
        return None
    op, raw = m.group(1), m.group(2).strip()
    if op == "=":
        op = "=="
    if raw.lower() in ("true", "false"):
        return op, raw.lower() == "true"
    try:
        if "." in raw:
            return op, float(raw)
        return op, int(raw)
    except ValueError:
        return op, raw.strip("'\"")


def _hold(actual: Any, op: str, expect: Any) -> bool:
    if op == "==":
        return actual == expect
    if op == ">":
        return actual > expect
    if op == ">=":
        return actual >= expect
    if op == "<":
        return actual < expect
    if op == "<=":
        return actual <= expect
    return False


def prune_library(
    constraints: Mapping[str, Dict[str, Any]],
    topology: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return constraints whose topo guards match ``topology``.

    Example topology: ``{"tp": 2, "dp": 1, "pp": 1, "tpl": False}``.
    """
    active: Dict[str, Dict[str, Any]] = {}
    for name, c in constraints.items():
        guards = c.get("pi_topo") or c.get("applicable_conditions") or {}
        if not isinstance(guards, dict):
            active[name] = c
            continue
        ok = True
        for key, expr in guards.items():
            if key in ("stage", "phase", "dtype", "_skip") or str(key).startswith("pi_"):
                continue
            if key not in topology:
                continue
            parsed = _parse_cmp(str(expr))
            if parsed is None:
                continue
            op, expect = parsed
            if not _hold(topology[key], op, expect):
                ok = False
                break
        if ok:
            active[name] = c
    return active


if __name__ == "__main__":
    lib = {
        "t01": {"pi_topo": {"tp": ">1", "tpl": "=False"}, "logic": "SELECT 1"},
        "dp_only": {"applicable_conditions": {"dp": ">1"}, "logic": "SELECT 1"},
    }
    assert "t01" in prune_library(lib, {"tp": 2, "dp": 1, "tpl": False})
    assert "t01" not in prune_library(lib, {"tp": 1, "dp": 1, "tpl": False})
    assert "dp_only" in prune_library(lib, {"tp": 1, "dp": 2})
    print("topology_prune self-check OK")
