#!/usr/bin/env python3
"""Score silent-vs-loud evidence for each bug in manifest_v2.json.

Outputs:
- silent_evidence_392.json : per-bug evidence level + signals
- silent_evidence_warnings.csv : WARNING + L4 bugs needing manual review

Evidence levels:
  L1 : reproduced + clean expected_output (strong)
  L2 : root_cause/invariant matches silent keywords, no loud keywords (medium)
  L3 : title-level silent keyword match only (weak)
  L4 : no clean signal -> manual inspect
WARNING is orthogonal: any crash keyword in expected_output.buggy or
description triggers it and forces needs_manual_review=True.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = EVAL_DIR / "manifest_v2.json"
OUT_JSON = EVAL_DIR / "silent_evidence_392.json"
OUT_CSV = EVAL_DIR / "silent_evidence_warnings.csv"

# --- regex patterns ---------------------------------------------------------
SILENT_POSITIVE_EXPECTED = re.compile(
    r"BUG DETECTED|mismatch|diverged|wrong|differs|incorrect|inconsistent|CLEAN",
    re.IGNORECASE,
)
CRASH_KEYWORDS = re.compile(
    r"\bcrash\b|RuntimeError|AssertionError|\bOOM\b|\bNaN\b|\bInf\b|cuda error",
    re.IGNORECASE,
)
SILENT_ROOT_CAUSE = re.compile(
    r"silently|wrong value|corrupted|inconsistent|undercount|overcount|"
    r"duplicate|missing|not normalized|dtype|precision|race|reorder|flag|tag",
    re.IGNORECASE,
)
LOUD_ROOT_CAUSE = re.compile(
    r"throws|raises|crash|segfault|assertionerror|runtimeerror",
    re.IGNORECASE,
)
SILENT_TITLE = re.compile(
    r"skip|wrong|incorrect|mismatch|undercount|lost|duplicate|precision|"
    r"dtype|sync|reduce",
    re.IGNORECASE,
)


def safe(text):
    return text if isinstance(text, str) else ""


def classify(bug):
    signals = []
    warnings = []

    title = safe(bug.get("title"))
    description = safe(bug.get("description"))
    root_cause = safe(bug.get("root_cause"))
    invariant = safe(bug.get("invariant"))
    detection_method = safe(bug.get("detection_method"))
    repro = bug.get("reproduction_status")
    expected = bug.get("expected_output") or {}
    if isinstance(expected, dict):
        buggy_out = safe(expected.get("buggy"))
        fixed_out = safe(expected.get("fixed"))
    else:
        buggy_out = safe(expected)
        fixed_out = ""

    # ---- WARNING: crash keyword in expected_output.buggy ------------------
    if buggy_out and CRASH_KEYWORDS.search(buggy_out):
        warnings.append("crash_keyword_in_buggy_output")

    # ---- WARNING: loud keyword in description / root_cause ----------------
    desc_blob = " ".join([description, root_cause])
    if LOUD_ROOT_CAUSE.search(desc_blob):
        warnings.append("loud_keyword_in_description")

    # ---- L1: reproduced + clean expected_output ---------------------------
    has_repro = repro == "reproduced"
    has_clean_expected = bool(buggy_out) and bool(
        SILENT_POSITIVE_EXPECTED.search(buggy_out)
        or SILENT_POSITIVE_EXPECTED.search(fixed_out)
    )
    expected_has_crash = bool(buggy_out and CRASH_KEYWORDS.search(buggy_out))

    level = None
    if has_repro and has_clean_expected and not expected_has_crash:
        level = "L1"
        signals.append("reproduced")
        signals.append("expected_output_silent_keyword")
        if not LOUD_ROOT_CAUSE.search(desc_blob):
            signals.append("fix_diff_no_raise")
    # also accept L1 even if reproduction_status is not 'reproduced' but
    # we already counted that above; otherwise fall through.

    # ---- L2: silent keywords in root_cause / invariant --------------------
    if level is None:
        rc_blob = " ".join([root_cause, invariant, detection_method])
        if (
            rc_blob.strip()
            and SILENT_ROOT_CAUSE.search(rc_blob)
            and not LOUD_ROOT_CAUSE.search(rc_blob)
        ):
            level = "L2"
            signals.append("root_cause_silent_keyword")
            if invariant:
                signals.append("invariant_present")

    # ---- L3: title-only weak match ----------------------------------------
    if level is None:
        if SILENT_TITLE.search(title):
            level = "L3"
            signals.append("title_silent_keyword")

    # ---- L4 fallback ------------------------------------------------------
    if level is None:
        level = "L4"
        signals.append("no_clean_signal")

    needs_manual = bool(warnings) or level == "L4"
    return {
        "bug_id": bug.get("bug_id"),
        "source_pool": bug.get("source_pool"),
        "framework": bug.get("framework"),
        "evidence_level": "WARNING" if warnings else level,
        "underlying_level": level,
        "evidence_signals": signals,
        "warning_reasons": warnings,
        "needs_manual_review": needs_manual,
    }


def main():
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)
    bugs = manifest["bugs"]

    results = [classify(b) for b in bugs]

    # ---- meta counts ------------------------------------------------------
    by_level = Counter()
    for r in results:
        if r["warning_reasons"]:
            by_level["WARNING"] += 1
        else:
            by_level[r["evidence_level"]] += 1
    # Make sure all keys present
    for k in ("L1", "L2", "L3", "L4", "WARNING"):
        by_level.setdefault(k, 0)

    out = {
        "meta": {
            "total": len(results),
            "by_level": {k: by_level[k] for k in ("L1", "L2", "L3", "L4", "WARNING")},
        },
        "bugs": [
            {
                "bug_id": r["bug_id"],
                "evidence_level": r["evidence_level"],
                "evidence_signals": r["evidence_signals"],
                "warning_reasons": r["warning_reasons"],
                "needs_manual_review": r["needs_manual_review"],
            }
            for r in results
        ],
    }
    with OUT_JSON.open("w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- CSV: WARNING + L4 ------------------------------------------------
    bug_by_id = {b["bug_id"]: b for b in bugs}
    rows = []
    for r in results:
        if r["warning_reasons"] or r["underlying_level"] == "L4":
            b = bug_by_id[r["bug_id"]]
            expected = b.get("expected_output") or {}
            buggy_snip = ""
            if isinstance(expected, dict):
                buggy_snip = safe(expected.get("buggy"))[:200]
            rc_snip = safe(b.get("root_cause"))[:200]
            rows.append(
                {
                    "bug_id": r["bug_id"],
                    "framework": b.get("framework"),
                    "level": r["evidence_level"],
                    "reasons": ";".join(r["warning_reasons"] or r["evidence_signals"]),
                    "title": safe(b.get("title"))[:160],
                    "expected_output_buggy_snippet": buggy_snip,
                    "root_cause_snippet": rc_snip,
                }
            )
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bug_id",
                "framework",
                "level",
                "reasons",
                "title",
                "expected_output_buggy_snippet",
                "root_cause_snippet",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # ---- stdout report ----------------------------------------------------
    print("=" * 70)
    print(f"Total bugs: {len(results)}")
    print("\nDistribution by level:")
    for k in ("L1", "L2", "L3", "L4", "WARNING"):
        print(f"  {k:8s} : {by_level[k]:4d}")

    warnings_only = [r for r in results if r["warning_reasons"]]
    print(f"\nTop 10 WARNING bugs ({len(warnings_only)} total):")
    for r in warnings_only[:10]:
        b = bug_by_id[r["bug_id"]]
        title = safe(b.get("title"))[:80]
        print(
            f"  {r['bug_id']:20s} [{','.join(r['warning_reasons'])}] "
            f"underlying={r['underlying_level']} | {title}"
        )

    print("\nDistribution by source_pool x level:")
    grid = defaultdict(Counter)
    for r in results:
        lvl = "WARNING" if r["warning_reasons"] else r["evidence_level"]
        grid[r["source_pool"]][lvl] += 1
    pools = ["128_only", "both", "295_only", "295_orphan"]
    levels = ["L1", "L2", "L3", "L4", "WARNING"]
    header = f"{'pool':12s} " + " ".join(f"{l:>8s}" for l in levels) + f" {'total':>8s}"
    print(header)
    print("-" * len(header))
    for pool in pools:
        c = grid.get(pool, Counter())
        total = sum(c.values())
        line = f"{pool:12s} " + " ".join(f"{c.get(l,0):>8d}" for l in levels) + f" {total:>8d}"
        print(line)

    print("\nWritten:")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_CSV}")


if __name__ == "__main__":
    main()
