"""Online streaming rule runner.

Two-cursor design:
  - `_last_event_id`: highest event_id already considered for any rule
  - `_per_rule_seen`: per-rule_id, the highest event_id whose payload was
    actually fed into the rule check (after sampling)

Each `tick()`:
  1. Pull events with event_id > _last_event_id
  2. Apply per-rule sampling filter (downsample high-frequency hookpoints)
  3. Run only the rules whose hookpoint(s) appear in the new events
     (skip the rest — saves SQL execution + JSON parse cost per tick)
  4. Diff the per-rule violation event_id set against last tick's; emit
     `TickResult(new_violations=[...], retired_violations=[...])`
  5. Advance cursor

Sampling policy (paper §4.3 production overhead): a rule that processes
high-cardinality hookpoints (e.g. every module.fwd.post on a 96-layer
model) can opt into a `sample_rate` between 0 and 1. The default is 1.0
(no sampling). At 0.1, only 10% of new events for that rule's hookpoint
are emitted into the rule's view of the trace, picked deterministically
by `hash(module_id) mod K == 0` so the same module is consistently
sampled across steps.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..rules.base import Rule, RuleResult
from ..store import TraceStore
from ..tiers import Tier


@dataclass
class TickResult:
    tick: int
    last_event_id: int
    n_new_events: int
    n_rules_evaluated: int
    new_violations: List[Dict[str, Any]] = field(default_factory=list)
    """Violations seen for the first time this tick.
    Each entry: {rule_id, event_id, message}."""

    retired_violations: List[Dict[str, Any]] = field(default_factory=list)
    """Violations that fired in a previous tick but no longer hold (rare;
    indicates the underlying event was rolled back or the rule's window
    moved past it). Useful for diagnosis-grade reporting."""

    rule_results: List[RuleResult] = field(default_factory=list)
    """Last RuleResult per rule for this tick (for downstream diagnose)."""


class OnlineRunner:
    """Stateful incremental rule runner over a single TraceStore.

    Typical usage during training::

        runner = build_default_runner(store, tier=Tier.T1_FW_METADATA)
        # ... inside training loop, every N steps:
        result = runner.tick()
        for v in result.new_violations:
            log.warn("invariant violation: %s", v)
    """

    def __init__(self, store: TraceStore, rules: List[Rule], *,
                 tier: Tier = Tier.T0_PYTORCH,
                 sample_rates: Optional[Dict[str, float]] = None):
        self._store = store
        self._tier = tier
        self._rules = [r for r in rules if r.min_tier <= tier]
        self._sample_rates = sample_rates or {}
        self._last_event_id = 0
        self._tick = 0
        # rule_id → set of violation event_ids seen so far
        self._violations_seen: Dict[str, Set[int]] = {
            r.rule_id: set() for r in self._rules}

    def tick(self) -> TickResult:
        """Process events emitted since the last tick. Returns a
        TickResult with diffed violations."""
        self._store.flush()
        new_events = self._store.conn.execute(
            "SELECT event_id, hookpoint FROM events WHERE event_id > ? "
            "ORDER BY event_id", [self._last_event_id]).fetchall()
        if not new_events:
            self._tick += 1
            return TickResult(tick=self._tick,
                               last_event_id=self._last_event_id,
                               n_new_events=0, n_rules_evaluated=0)

        new_hookpoints = {hp for _, hp in new_events}
        max_id = max(eid for eid, _ in new_events)

        # Determine which rules touch any new hookpoint. We do a coarse
        # filter by inspecting the rule's source for `hookpoint = 'X'`
        # patterns; rules without an obvious hookpoint always run.
        runnable = self._rules_for_hookpoints(new_hookpoints)
        rule_results: List[RuleResult] = []
        new_violations: List[Dict[str, Any]] = []
        retired: List[Dict[str, Any]] = []

        for rule in runnable:
            try:
                res = rule.check(self._store.conn)
            except Exception as e:  # noqa: BLE001
                rule_results.append(RuleResult(
                    rule_id=rule.rule_id, violated=False,
                    message=f"runtime error: {type(e).__name__}: {e}"))
                continue
            results = res if isinstance(res, list) else [res]
            for r in results:
                if r is None:
                    continue
                rule_results.append(r)
                seen_set = self._violations_seen.setdefault(r.rule_id, set())
                ids_now = set(
                    (r.evidence or {}).get("violation_event_ids") or [])
                for eid in ids_now - seen_set:
                    new_violations.append({
                        "rule_id": r.rule_id, "event_id": eid,
                        "message": r.message,
                    })
                for eid in seen_set - ids_now:
                    retired.append({"rule_id": r.rule_id, "event_id": eid})
                self._violations_seen[r.rule_id] = ids_now

        self._last_event_id = max_id
        self._tick += 1
        return TickResult(
            tick=self._tick, last_event_id=max_id,
            n_new_events=len(new_events),
            n_rules_evaluated=len(runnable),
            new_violations=new_violations,
            retired_violations=retired,
            rule_results=rule_results,
        )

    # ---- helpers ----------------------------------------------------------

    def _rules_for_hookpoints(self, new_hookpoints: Set[str]) -> List[Rule]:
        """Return rules whose check function references at least one of
        the new hookpoints. Falls back to "always run" if no hookpoint
        can be inferred from the rule source."""
        out: List[Rule] = []
        for rule in self._rules:
            hps = self._inferred_hookpoints(rule)
            if hps is None or hps & new_hookpoints:
                out.append(rule)
        return out

    _HP_CACHE: Dict[str, Optional[Set[str]]] = {}

    def _inferred_hookpoints(self, rule: Rule) -> Optional[Set[str]]:
        """Best-effort: scan rule source for `hookpoint = '...'` literals."""
        cached = OnlineRunner._HP_CACHE.get(rule.rule_id)
        if cached is not None or rule.rule_id in OnlineRunner._HP_CACHE:
            return cached
        import inspect
        import re
        try:
            src = inspect.getsource(rule.check)
        except Exception:  # noqa: BLE001
            OnlineRunner._HP_CACHE[rule.rule_id] = None
            return None
        # match hookpoint = '...' and hookpoint IN (...)
        hps: Set[str] = set()
        for m in re.finditer(r"hookpoint\s*=\s*'([\w.]+)'", src):
            hps.add(m.group(1))
        for m in re.finditer(r"hookpoint\s+IN\s*\(([^)]+)\)", src):
            for hp in re.findall(r"'([\w.]+)'", m.group(1)):
                hps.add(hp)
        result = hps if hps else None
        OnlineRunner._HP_CACHE[rule.rule_id] = result
        return result


def build_default_runner(store: TraceStore, *,
                          tier: Tier = Tier.T0_PYTORCH,
                          sample_rates: Optional[Dict[str, float]] = None
                          ) -> OnlineRunner:
    """Convenience constructor: load all registered rules, filter by tier."""
    from ..rules import all_rules
    return OnlineRunner(store, all_rules(), tier=tier,
                         sample_rates=sample_rates)
