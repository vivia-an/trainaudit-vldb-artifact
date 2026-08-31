#!/usr/bin/env python3
"""Extract the buggy- and fixed-side outcome of each Real-SE replay from its smoke log.

SMOKE_REPORT.md tabulates the buggy side only; the fixed-side verdicts behind the
0/17 false-positive claim live in prose. They are, however, recorded per case in
logs/smoke/<case>_smoke.log, which marks each side with a `===== ... BUGGY|FIXED ... =====`
banner and then a `[<case>] BUG DETECTED:` or `[<case>] CLEAN:` line.

Writes real_se_replay_outcomes.csv.
"""
import csv
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
LOGS = HERE / "logs" / "smoke"
OUT = HERE / "real_se_replay_outcomes.csv"
CURRENT = HERE / "real_se_detection.csv"      # the 17 cases the paper reports

# Banners vary: "===== BUGGY (sha) =====", "===== TrainAudit on FIXED (sha) =====",
# "===== TrainAudit T0 on BUGGY (sha~1) =====". Match the keyword anywhere in the banner.
SIDE = re.compile(r"=====.*?\b(BUGGY|FIXED)\b", re.I)
# ">>> Fixed: ASSERT fired (expected — bug prevented)" / "[B1] CLEAN: ..." / "[B1] BUG DETECTED: ..."
ARROW = re.compile(r">>>\s*(Buggy|Fixed):\s*(.+)", re.I)
TAGGED = re.compile(r"\[[^\]]+\]\s*(BUG DETECTED|CLEAN|ASSERT[^:]*)\s*:?\s*(.*)")
# Some drivers report a violation tally instead: "=== 0/17 violations ==="
TALLY = re.compile(r"===\s*(\d+)\s*/\s*(\d+)\s+violations\s*===")


def classify(text):
    t = text.upper()
    if "ASSERT" in t:
        return "ASSERT_FIRED"
    if "BUG DETECTED" in t or "DETECTED" in t:
        return "DETECTED"
    if "CLEAN" in t:
        return "CLEAN"
    return ""


def parse(path):
    side = None
    res = {"buggy": ("", ""), "fixed": ("", "")}
    for line in path.read_text(errors="replace").splitlines():
        m = SIDE.search(line)
        if m:
            side = m.group(1).lower()
            continue
        m = ARROW.search(line)
        if m:                                   # explicit ">>> Buggy:/Fixed:" summary wins
            res[m.group(1).lower()] = (classify(m.group(2)), m.group(2).strip()[:160])
            continue
        m = TAGGED.search(line)
        if m and side:
            verdict = classify(m.group(1))
            if verdict and not res[side][0]:
                res[side] = (verdict, (m.group(1) + ": " + m.group(2)).strip()[:160])
            continue
        m = TALLY.search(line)
        if m and side and not res[side][0]:
            n, total = int(m.group(1)), m.group(2)
            res[side] = ("DETECTED" if n else "CLEAN", f"{n}/{total} rules violated")
    return res


def main():
    if not LOGS.is_dir():
        sys.exit(f"missing {LOGS}")
    current = set()
    if CURRENT.exists():
        current = {r["case_id"] for r in csv.DictReader(CURRENT.open())}
    rows = []
    for f in sorted(LOGS.glob("*_smoke.log")):
        case = f.name[: -len("_smoke.log")]
        r = parse(f)
        rows.append({
            "case_id": case,
            "in_current_set": "yes" if case in current else "no",
            "buggy_verdict": r["buggy"][0], "fixed_verdict": r["fixed"][0],
            "buggy_detail": r["buggy"][1], "fixed_detail": r["fixed"][1],
            "log": f"logs/smoke/{f.name}",
        })
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scoped = [r for r in rows if r["in_current_set"] == "yes"] if current else rows
    det = [r for r in scoped if r["buggy_verdict"] == "DETECTED"]
    clean = [r for r in scoped if r["fixed_verdict"] == "CLEAN"]
    assert_ = [r for r in scoped if r["fixed_verdict"] == "ASSERT_FIRED"]
    fp = [r for r in scoped if r["fixed_verdict"] == "DETECTED"]
    print(f"wrote {OUT.name}: {len(rows)} logs, {len(scoped)} in the current case set")
    print(f"  buggy side detected        {len(det)}")
    print(f"  fixed side clean           {len(clean)}")
    print(f"  fixed side assertion-fired {len(assert_)}"
          + (f"  ({', '.join(r['case_id'] for r in assert_)})" if assert_ else ""))
    print(f"  fixed side false positive  {len(fp)}"
          + (f"  ({', '.join(r['case_id'] for r in fp)})" if fp else ""))
    print(f"  -> paired fixed-side evaluations {len(clean) + len(assert_)}, "
          f"false positives {len(fp)}")
    unparsed = [r["case_id"] for r in scoped if not r["buggy_verdict"] and not r["fixed_verdict"]]
    if unparsed:
        print(f"  no verdict parsed          {len(unparsed)} ({', '.join(unparsed)})")


if __name__ == "__main__":
    main()
