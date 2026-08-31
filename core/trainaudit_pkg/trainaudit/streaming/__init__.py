"""Online streaming verification (paper §3.5 / §4.3 production-readiness).

The default `trainaudit.run_rules()` is a batch-mode scan: it loads every
event in the table and re-evaluates every rule each call. That's fine for
post-mortem diagnosis but too expensive for production training where
checks should fire incrementally as new events land.

OnlineRunner maintains a `last_processed_event_id` cursor and a per-rule
violation cache, so each `tick()` only re-runs rules that touch new
events. Sampling (by module_class hash mod K) is layered on top so a
high-frequency hookpoint can be downsampled without losing tail-event
coverage.
"""
from .online_runner import OnlineRunner, TickResult, build_default_runner

__all__ = ["OnlineRunner", "TickResult", "build_default_runner"]
