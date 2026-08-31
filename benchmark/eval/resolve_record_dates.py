#!/usr/bin/env python3
"""Resolve an upstream date for each of the 392 evidence-corpus records.

The temporal holdout in Sec 5.4 is recorded as blocked for want of per-record dates. The
dates are not missing, only unresolved: 98% of the records in manifest_v2.json carry an
issue URL or a commit hash, and both are dated upstream. This script asks GitHub for
those dates and writes record_dates.csv, from which a time split can be cut.

Resumable: existing rows in the output are kept and not re-fetched, so an interrupted
run can simply be restarted.

    python3 resolve_record_dates.py               # resolve everything still missing
    python3 resolve_record_dates.py --limit 50    # do a slice at a time
    python3 resolve_record_dates.py --summary     # just describe what is already resolved

Needs `gh` authenticated (it is only reading public repositories).
"""
import argparse
import collections
import csv
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "manifest_v2.json"
OUT = HERE / "record_dates.csv"
FIELDS = ["bug_id", "framework", "repo", "date", "date_kind", "source", "note"]

REPO_OF = {  # short framework name -> upstream repository, for records with only a hash
    "deepspeed": "microsoft/DeepSpeed",
    "megatron-lm": "NVIDIA/Megatron-LM",
    "olmo": "allenai/OLMo",
    "olmo-core": "allenai/OLMo-core",
}


def gh_json(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=45)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def blank(v):
    return not v or str(v).strip().lower() in ("", "none", "null", "n/a", "-")


def repo_for(rec):
    repo = rec.get("repo") or ""
    if "/" in repo:
        return repo
    return REPO_OF.get((repo or rec.get("framework") or "").lower())


def resolve(rec):
    """(date, kind, source, note) — issue creation is preferred; a commit date is a fallback."""
    url = rec.get("issue_url")
    if not blank(url):
        m = re.search(r"github\.com/([^/]+/[^/]+)/(issues|pull)/(\d+)", url)
        if m:
            owner, kind, num = m.group(1), m.group(2), m.group(3)
            # the issues endpoint serves pull requests too
            d = gh_json(f"repos/{owner}/issues/{num}")
            if d and d.get("created_at"):
                return d["created_at"][:10], f"{kind}_created", url, ""
    repo = repo_for(rec)
    for field in ("fixed_commit", "buggy_commit"):
        sha = rec.get(field)
        if blank(sha) or not repo:
            continue
        sha = str(sha).split("~")[0].strip()          # "abc123~1" -> "abc123"
        if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
            continue
        d = gh_json(f"repos/{repo}/commits/{sha}")
        if d:
            date = (d.get("commit", {}).get("committer") or {}).get("date")
            if date:
                return date[:10], field.replace("_commit", "_commit_date"), f"{repo}@{sha}", ""
    return "", "", "", "no resolvable identifier" if blank(url) and not repo else "lookup failed"


def load_existing():
    if not OUT.exists():
        return {}
    return {r["bug_id"]: r for r in csv.DictReader(OUT.open())}


def write(rows):
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([rows[k] for k in sorted(rows)])


def summarise(rows):
    got = [r for r in rows.values() if r["date"]]
    print(f"{len(got)}/{len(rows)} records dated")
    if not got:
        return
    dates = sorted(r["date"] for r in got)
    print(f"  span            {dates[0]} .. {dates[-1]}")
    print(f"  median          {dates[len(dates) // 2]}")
    print("  by source kind  " + ", ".join(
        f"{k}={v}" for k, v in collections.Counter(r["date_kind"] for r in got).most_common()))
    by_year = collections.Counter(r["date"][:4] for r in got)
    print("  by year         " + ", ".join(f"{y}={n}" for y, n in sorted(by_year.items())))
    cut = dates[len(dates) // 2]
    print(f"\n  a median split at {cut} gives "
          f"{sum(d < cut for d in dates)} earlier / {sum(d >= cut for d in dates)} later records")
    missing = [r["bug_id"] for r in rows.values() if not r["date"]]
    if missing:
        print(f"  unresolved: {len(missing)} ({', '.join(missing[:10])}"
              f"{', ...' if len(missing) > 10 else ''})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="resolve at most N new records")
    ap.add_argument("--summary", action="store_true", help="describe record_dates.csv and stop")
    args = ap.parse_args()

    rows = load_existing()
    if args.summary:
        if not rows:
            sys.exit(f"{OUT.name} does not exist yet")
        summarise(rows)
        return

    bugs = json.loads(MANIFEST.read_text())["bugs"]
    todo = [b for b in bugs if b["bug_id"] not in rows or not rows[b["bug_id"]]["date"]]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(bugs)} records, {len(rows)} already resolved, fetching {len(todo)}")

    for i, rec in enumerate(todo, 1):
        date, kind, source, note = resolve(rec)
        rows[rec["bug_id"]] = {
            "bug_id": rec["bug_id"],
            "framework": rec.get("framework", ""),
            "repo": repo_for(rec) or rec.get("repo", ""),
            "date": date, "date_kind": kind, "source": source, "note": note,
        }
        if i % 25 == 0 or i == len(todo):
            write(rows)
            done = sum(1 for r in rows.values() if r["date"])
            print(f"  {i}/{len(todo)} fetched, {done} dated in total")
    write(rows)
    print()
    summarise(rows)


if __name__ == "__main__":
    main()
