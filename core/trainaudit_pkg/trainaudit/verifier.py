"""Run all enabled rules against a trace store, return RuleResult list."""
import importlib
from pathlib import Path
from typing import List, Optional

from .rules import all_rules
from .rules.base import RuleResult
from .store import TraceStore
from .tiers import Tier


_DEFAULT_DSL_REGISTRY = Path(__file__).resolve().parent / "dsl" / "registry"


def run_rules(store: TraceStore,
              tier: Tier = Tier.T0_PYTORCH,
              *,
              use_dsl: bool = False,
              dsl_registry: Optional[Path] = None,
              rules_dir: Optional[Path] = None) -> List[RuleResult]:
    """Evaluate every rule whose min_tier <= `tier`.

    When `use_dsl=True`, evaluate DSL predicates from `dsl_registry/`
    (default: `trainaudit/dsl/registry/`) instead of the hand-written
    Python rule registry. Tier filter still applies (predicates carry
    `min_tier: T0_PYTORCH | T1_FW_METADATA | ...`).

    `rules_dir` (Optional, SPEC_S2 ablation): a sibling of `rules/` like
    `rules_no_topo`; loaded instead of `.rules`. Ignored with `use_dsl`.
    """
    store.flush()
    if use_dsl:
        return _run_dsl(store, tier, dsl_registry or _DEFAULT_DSL_REGISTRY)
    rules_to_run = _load_rules_from_dir(rules_dir) if rules_dir else all_rules()
    results: List[RuleResult] = []
    for r in rules_to_run:
        if r.min_tier > tier:
            continue
        try:
            res = r.check(store.conn)
        except Exception as e:  # noqa: BLE001
            results.append(RuleResult(
                rule_id=r.rule_id,
                violated=False,
                message=f"rule error: {type(e).__name__}: {e}",
            ))
            continue
        if isinstance(res, list):
            results.extend(res)
        elif res is not None:
            results.append(res)
    return results


_ABLATION_CACHE: dict = {}


def _load_rules_from_dir(rules_dir: Path):
    """Load rules from a `trainaudit.rules_<mode>` sibling package (SPEC_S2)."""
    key = str(Path(rules_dir).resolve())
    if key not in _ABLATION_CACHE:
        pkg = importlib.import_module(f"trainaudit.{Path(key).name}")
        _ABLATION_CACHE[key] = list(pkg.all_rules())
    return _ABLATION_CACHE[key]


def _run_dsl(store: TraceStore, tier: Tier,
             registry: Path) -> List[RuleResult]:
    from .dsl import load_predicates_dir, run_dsl_rules
    preds = load_predicates_dir(registry)
    # min_tier filter
    name_to_tier = {t.name: t for t in Tier}
    keep = [p for p in preds
            if name_to_tier.get(p.min_tier, Tier.T0_PYTORCH) <= tier]
    return run_dsl_rules(store, keep)


def summarize(results: List[RuleResult]) -> str:
    n_total = len(results)
    n_violated = sum(1 for r in results if r.violated)
    lines = [f"=== {n_violated}/{n_total} violations ==="]
    for r in results:
        tag = "VIOLATION" if r.violated else "ok"
        lines.append(f"  [{tag:9s}] {r.rule_id}: {r.message}")
    return "\n".join(lines)
