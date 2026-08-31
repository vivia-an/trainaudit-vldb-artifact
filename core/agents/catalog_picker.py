"""Pattern Catalog loader and S1 PickPattern for verified mining."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# core_algo/agents → core_algo/config/pattern_catalog_snapshot.json
_CORE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT = _CORE_ROOT / "config" / "pattern_catalog_snapshot.json"


def catalog_path() -> Path:
    env = os.environ.get("SDC_PATTERN_CATALOG", "")
    if env:
        return Path(env)
    if _DEFAULT.is_file():
        return _DEFAULT
    # fallback: sibling sdccheck/config when developing in monorepo
    alt = _CORE_ROOT.parent / "config" / "pattern_catalog_snapshot.json"
    return alt if alt.is_file() else _DEFAULT


def load_catalog() -> Dict[str, Any]:
    path = catalog_path()
    if not path.exists():
        return {"version": "missing", "templates": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_templates() -> List[Dict[str, Any]]:
    return list(load_catalog().get("templates") or [])


def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    tid = (template_id or "").strip()
    if not tid:
        return None
    for t in list_templates():
        if t.get("id") == tid or tid in (t.get("aliases") or []):
            return t
        if t.get("name") == tid:
            return t
    return None


def pick_pattern(library_categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """S1.PickPattern: prefer SDC_TARGET_TEMPLATE, else first uncovered-ish template."""
    target = os.environ.get("SDC_TARGET_TEMPLATE", "").strip()
    if target:
        t = get_template(target)
        if t:
            print(f"[trainaudit] S1 PickPattern → {t.get('id')} ({t.get('name')}) [env]")
            return t
        print(f"[trainaudit] S1 warn: SDC_TARGET_TEMPLATE={target} not in catalog, falling through")

    templates = list_templates()
    if not templates:
        return {"id": "", "name": "unspecified", "relation": "", "aliases": []}

    # Prefer templates whose alias/id not mentioned in env exclude list
    exclude = set(
        x.strip()
        for x in os.environ.get("SDC_CATALOG_EXCLUDE", "").split(",")
        if x.strip()
    )
    for t in templates:
        if t.get("id") in exclude:
            continue
        print(f"[trainaudit] S1 PickPattern → {t.get('id')} ({t.get('name')})")
        return t
    return templates[0]


def _join_list(vals: Any, *, limit: int = 8) -> str:
    if not vals:
        return "—"
    if isinstance(vals, str):
        return vals
    items = [str(x) for x in vals if x is not None]
    if not items:
        return "—"
    head = items[:limit]
    more = f" …(+{len(items) - limit})" if len(items) > limit else ""
    return "; ".join(head) + more


def prompt_block(template: Dict[str, Any]) -> str:
    """Build the S1 catalog block injected into the miner system prompt."""
    if not template or not template.get("id"):
        return ""
    aliases = _join_list(template.get("aliases") or [], limit=6)
    topo = template.get("default_pi_topo") or {}
    pre = template.get("default_pi_precond") or {}
    schema = (template.get("schema_predicate") or "").strip()
    witness = (template.get("witness") or "").strip()
    lines = [
        "",
        "[Pattern Catalog — S1 PickPattern]",
        f"Selected template_id={template.get('id')} name={template.get('name')}",
        f"relation={template.get('relation')} aliases={aliases}",
        f"objects={template.get('objects', '')}",
        f"obligation/witness={witness}",
    ]
    if schema:
        lines.append(f"schema_predicate={schema}")
    lines.append(
        "permitted_topology_guards="
        + _join_list(template.get("permitted_topology_guards"), limit=10)
    )
    lines.append(
        "permitted_preconditions="
        + _join_list(template.get("permitted_preconditions"), limit=10)
    )
    lines.append(
        "permitted_phases=" + _join_list(template.get("permitted_phases"), limit=8)
    )
    if topo:
        lines.append(f"default_pi_topo={json.dumps(topo, ensure_ascii=False)}")
    if pre:
        lines.append(f"default_pi_precond={json.dumps(pre, ensure_ascii=False)}")
    lines.append(
        "counterexamples(must NOT fire)="
        + _join_list(template.get("counterexamples"), limit=6)
    )
    lines.append(
        "positive_examples(corpus ids)="
        + _join_list(template.get("positive_examples"), limit=8)
    )
    lines.extend(
        [
            "Ground THIS template only into π_schema ∧ π_topo ∧ π_precond.",
            "Set constraint field template_id to the id above.",
            "Do not invent a different core relation or merge another Txx.",
            "Prefer guards from permitted_* / default_pi_*; stay inside merge_boundary.",
            "Emit confidence in [0,1].",
            "",
        ]
    )
    return "\n".join(lines)
