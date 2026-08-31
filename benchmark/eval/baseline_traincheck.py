"""E1: TrainCheck baseline.

Routes the synthetic D1 surrogates and (in subprocess mode) the 23 real bugs
through TrainCheck's collect → infer → check pipeline, producing the CSV
that paper §4.1's three-way comparison consumes.

Pipeline (per surrogate bug):
  1. traincheck-collect  -p <bug>_fixed.py  → trace_fixed/
  2. traincheck-infer    -f trace_fixed     → invariants.json
  3. traincheck-collect  -p <bug>_buggy.py  → trace_buggy/
  4. traincheck-checker  -f trace_buggy -i invariants.json
       → "Total failed invariants: N/M" — DETECTED if N>0

The driver lives at benchmark/eval/traincheck_surrogates/run_one.sh and is
invoked over SSH on eval-gpu-0 (TrainCheck instrumentation requires the
real H200 venv at /volume/qscai/cqs/temp/venv-cu126/).

Modes:
  --mode harness-check  → import / API smoke check (legacy)
  --mode synthetic      → parse batch_t{0,1}_results.txt → CSV
  --mode subprocess     → 23 real bugs (TODO; needs traincheck_run.sh per bug)
  --mode related-work   → fallback qualitative table (NOT used: doc 26 §5.5
                          rejected, must run on same集合)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TC_ROOT = REPO_ROOT / "exp" / "traincheck" / "TrainCheck"
HARNESS_ROOT = REPO_ROOT / "benchmark" / "eval"
SURROGATE_ROOT = HARNESS_ROOT / "traincheck_surrogates"

CSV_FIELDS = [
    "bug_id", "framework", "category", "phase", "set",
    "trainaudit_verdict", "traincheck_verdict",
    "traincheck_violations", "duration_s", "note",
]

# T1 surrogates that synthetic_runners.py emits as fake events (no real
# torch training). The traincheck_surrogates/<bug>_{fixed,buggy}.py rewrites
# these as real torch loops so TrainCheck has something to instrument.
T1_BUGS = {"B1", "B2", "B3", "B8",
           "B13", "M-012", "M-NEW-5", "M-024", "M-020", "OC-NEW-3"}


# ---- TrainCheck import smoke check (legacy) --------------------------------


def _import_traincheck() -> Tuple[bool, str]:
    try:
        if str(TC_ROOT) not in sys.path:
            sys.path.insert(0, str(TC_ROOT))
        import traincheck  # noqa: F401
        from traincheck import Instrumentor  # noqa: F401
        return True, f"TrainCheck imported from {TC_ROOT}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def harness_check() -> Dict[str, Any]:
    ok, msg = _import_traincheck()
    out: Dict[str, Any] = {"import_ok": ok, "import_msg": msg}
    if not ok:
        return out
    import traincheck
    out["traincheck_module_path"] = traincheck.__file__
    out["public_api"] = sorted(getattr(traincheck, "__all__", []))
    submods = []
    for sub in ("instrumentor", "checker", "infer_engine",
                 "collect_trace", "invariant"):
        try:
            __import__(f"traincheck.{sub}")
            submods.append(sub)
        except Exception as e:  # noqa: BLE001
            submods.append(f"{sub} (FAIL: {type(e).__name__})")
    out["submodules"] = submods
    return out


# ---- Parse batch results ---------------------------------------------------


_LINE_RE = re.compile(
    r"^\[(?P<id>[^\]]+)\]\s*"
    r"(?P<verdict>BUG DETECTED|CLEAN|FAIL)"
    r"\s*:\s*(?P<rest>.*)$"
)
_RATIO_RE = re.compile(r"traincheck:(\d+)/(\d+)")


def parse_batch_results(path: Path) -> Dict[str, Dict[str, Any]]:
    """Parse a batch_t{0,1}_results.txt file. Returns {bug_id: {...}}."""
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        bug_id = m.group("id")
        verdict_word = m.group("verdict")
        rest = m.group("rest")
        if verdict_word == "BUG DETECTED":
            verdict = "DETECTED"
            ratio = _RATIO_RE.search(rest)
            violations = (f"{ratio.group(1)}/{ratio.group(2)}"
                           if ratio else rest[:60])
        elif verdict_word == "CLEAN":
            verdict = "CLEAN"
            ratio = _RATIO_RE.search(rest)
            violations = (f"{ratio.group(1)}/{ratio.group(2)}"
                           if ratio else "0/?")
        else:
            verdict = "FAIL"
            violations = rest[:80]
        out[bug_id] = {"verdict": verdict, "violations": violations,
                       "raw": rest[:200]}
    return out


def _trainaudit_verdict_for(bug_id: str, phase: str) -> str:
    """Re-run synthetic_runners to get TrainAudit's verdict for this
    (bug_id, phase). Cached per-process so we don't re-run for both phases."""
    sys.path.insert(0, str(HARNESS_ROOT))
    from synthetic_runners import run_synthetic_bug  # type: ignore[import-not-found]
    if not hasattr(_trainaudit_verdict_for, "_cache"):
        _trainaudit_verdict_for._cache = {}  # type: ignore[attr-defined]
    cache = _trainaudit_verdict_for._cache  # type: ignore[attr-defined]
    if bug_id not in cache:
        try:
            outputs = run_synthetic_bug(bug_id)
            cache[bug_id] = {bug_phase: verdict
                              for (_, verdict, msg) in outputs
                              for bug_phase in [
                                  "buggy" if msg.startswith("[buggy]")
                                  else "fixed" if msg.startswith("[fixed]")
                                  else "?"]}
        except Exception as e:  # noqa: BLE001
            cache[bug_id] = {"_err": f"{type(e).__name__}: {e}"}
    entry = cache[bug_id]
    if phase in entry:
        v = entry[phase]
        if v == "BUG DETECTED":
            return "DETECTED"
        if v == "CLEAN":
            return "CLEAN"
        return "FAIL"
    return "?"


# ---- Synthetic mode --------------------------------------------------------


def run_synthetic(subset_path: Path, manifest_path: Path,
                   batch_t0: Path, batch_t1: Path,
                   batch_t1_extra: Path = None) -> List[Dict[str, Any]]:
    subset = json.loads(subset_path.read_text())
    bug_ids = subset["bugs"]
    manifest = {r["bug_id"]: r for r
                in json.loads(manifest_path.read_text())}
    tc_results: Dict[str, Dict[str, Any]] = {}
    tc_results.update(parse_batch_results(batch_t0))
    tc_results.update(parse_batch_results(batch_t1))
    if batch_t1_extra is not None:
        tc_results.update(parse_batch_results(batch_t1_extra))

    rows: List[Dict[str, Any]] = []
    for bug_id in bug_ids:
        meta = manifest.get(bug_id, {})
        framework = meta.get("framework", "?")
        category = meta.get("category", "?")
        tc = tc_results.get(bug_id)
        for phase in ("buggy", "fixed"):
            ta_verdict = _trainaudit_verdict_for(bug_id, phase)
            if phase == "buggy":
                if tc is None:
                    tc_verdict = "FAIL"
                    tc_violations = ""
                    note = ("no batch result row for this bug — surrogate "
                            "may not have been included in batch_t{0,1}.sh")
                else:
                    tc_verdict = tc["verdict"]
                    tc_violations = tc["violations"]
                    note = tc["raw"]
            else:
                # This legacy D1 parser only sees per-bug detection logs
                # generated from fixed-reference -> buggy-check runs. Paper FP
                # evidence must be populated from a separate held-out fixed
                # rerun checked against invariants learned from the reference.
                tc_verdict = "CLEAN"
                tc_violations = "0/N"
                note = ("legacy parser placeholder; use held-out fixed rerun "
                        "logs for paper FP evidence")
            rows.append({
                "bug_id": bug_id,
                "framework": framework,
                "category": category,
                "phase": phase,
                "set": "D1",
                "trainaudit_verdict": ta_verdict,
                "traincheck_verdict": tc_verdict,
                "traincheck_violations": tc_violations,
                "duration_s": "",  # batch did not record per-bug timing
                "note": note,
            })
    return rows


# ---- Output writers --------------------------------------------------------


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def write_paper_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    out = ["# Paper §4.1 — TrainAudit vs TrainCheck (D1 same-集合)\n\n"]
    out.append("## Per-bug verdict comparison (buggy phase)\n\n")
    out.append("| bug_id | framework | TrainAudit | TrainCheck | violations | note |\n")
    out.append("|---|---|---|---|---|---|\n")
    n_ta_det = n_tc_det = n_tc_fail = 0
    for r in rows:
        if r["phase"] != "buggy":
            continue
        if r["trainaudit_verdict"] == "DETECTED":
            n_ta_det += 1
        if r["traincheck_verdict"] == "DETECTED":
            n_tc_det += 1
        if r["traincheck_verdict"] == "FAIL":
            n_tc_fail += 1
        out.append(f"| {r['bug_id']} | {r['framework']} | "
                    f"{r['trainaudit_verdict']} | {r['traincheck_verdict']} | "
                    f"{r['traincheck_violations']} | {r['note'][:60]} |\n")
    n_buggy = sum(1 for r in rows if r["phase"] == "buggy")
    out.append("\n## Summary\n\n")
    out.append(f"- TrainAudit DETECTED: **{n_ta_det}/{n_buggy}** "
                f"({n_ta_det/max(n_buggy,1):.1%})\n")
    out.append(f"- TrainCheck DETECTED: **{n_tc_det}/{n_buggy}** "
                f"({n_tc_det/max(n_buggy,1):.1%})\n")
    out.append(f"- TrainCheck FAIL (driver/infer crash): {n_tc_fail}\n")
    path.write_text("".join(out))


# ---- CLI -------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["harness-check", "synthetic",
                                          "subprocess", "related-work"],
                    default="synthetic")
    ap.add_argument("--subset", default=str(HARNESS_ROOT / "synthetic_17.json"))
    ap.add_argument("--manifest", default=str(HARNESS_ROOT / "manifest.json"))
    ap.add_argument("--batch-t0",
                    default=str(SURROGATE_ROOT / "batch_t0_results.txt"))
    ap.add_argument("--batch-t1",
                    default=str(SURROGATE_ROOT / "batch_t1_results.txt"))
    ap.add_argument("--batch-t1-extra",
                    default=str(SURROGATE_ROOT / "batch_t1_extra_results.txt"))
    ap.add_argument("--out",
                    default=str(HARNESS_ROOT / "baseline_traincheck_results.csv"))
    ap.add_argument("--table-out",
                    default=str(HARNESS_ROOT / "paper_table_baseline_traincheck.md"))
    args = ap.parse_args()

    if args.mode == "harness-check":
        status = harness_check()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.mode == "synthetic":
        rows = run_synthetic(Path(args.subset), Path(args.manifest),
                              Path(args.batch_t0), Path(args.batch_t1),
                              Path(args.batch_t1_extra))
        write_csv(Path(args.out), rows)
        write_paper_table(Path(args.table_out), rows)
        n_buggy = sum(1 for r in rows if r["phase"] == "buggy")
        n_tc_det = sum(1 for r in rows
                        if r["phase"] == "buggy"
                        and r["traincheck_verdict"] == "DETECTED")
        n_tc_fail = sum(1 for r in rows
                         if r["phase"] == "buggy"
                         and r["traincheck_verdict"] == "FAIL")
        print(f"-> {args.out}")
        print(f"-> {args.table_out}")
        print(f"TrainCheck buggy detected: {n_tc_det}/{n_buggy}")
        print(f"TrainCheck buggy FAIL: {n_tc_fail}/{n_buggy}")
        return

    if args.mode == "subprocess":
        print("subprocess mode (23 real bugs) not yet implemented")
        sys.exit(2)

    if args.mode == "related-work":
        print("related-work mode rejected per doc 26 §5.5; user requires "
              "same-集合 quantitative comparison")
        sys.exit(2)


if __name__ == "__main__":
    main()
