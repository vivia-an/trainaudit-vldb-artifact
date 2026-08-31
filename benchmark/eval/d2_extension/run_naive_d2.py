"""Naïve baseline on 8 D2-new surrogates: NaN/Inf detection + RuntimeError check."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

BUGS = ["ID1", "CC1", "PE1", "AV1", "TA1", "SC1", "CW1", "LN1"]


def run_naive(bug, variant):
    f = ROOT / f"{bug}_{variant}.py"
    out = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=120)
    text = out.stdout + out.stderr
    triggered = (
        "nan" in text.lower() or "inf" in text.lower() or
        "RuntimeError" in text or "Exception" in text or
        out.returncode != 0
    )
    return {"bug_id": bug, "variant": variant, "triggered": triggered, "returncode": out.returncode,
            "tail": text.strip().splitlines()[-3:] if text else []}


def main():
    results = []
    for bug in BUGS:
        b = run_naive(bug, "buggy")
        f = run_naive(bug, "fixed")
        entry = {
            "bug_id": bug,
            "buggy_detected": b["triggered"],
            "fixed_fp": f["triggered"],
            "buggy_rc": b["returncode"], "fixed_rc": f["returncode"],
        }
        results.append(entry)
        print(f"  {bug}: buggy_detected={'✓' if entry['buggy_detected'] else '✗'}  "
              f"fixed_fp={'✓' if entry['fixed_fp'] else '✗'}  "
              f"buggy_rc={entry['buggy_rc']}, fixed_rc={entry['fixed_rc']}")

    detected = sum(1 for r in results if r["buggy_detected"])
    fp = sum(1 for r in results if r["fixed_fp"])
    out = {"tool": "naive", "n_bugs": 8, "buggy_detection": f"{detected}/8",
           "fixed_fp": f"{fp}/8", "results": results}

    (RESULTS / "naive_d2_new.json").write_text(json.dumps(out, indent=2))
    print(f"\nNaïve D2-new: {detected}/8 buggy detected, {fp}/8 fixed FP")


if __name__ == "__main__":
    main()
