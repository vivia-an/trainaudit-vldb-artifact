#!/usr/bin/env python3
"""Check the public artifact's release-facing structure and terminology."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [p.decode() for p in result.stdout.split(b"\0") if p]


def main() -> int:
    tracked = tracked_files()
    tracked_set = set(tracked)
    errors: list[str] = []

    required = {
        "README.md",
        "LICENSE",
        "CITATION.cff",
        ".github/workflows/release-checks.yml",
        "scripts/check_release.sh",
        "scripts/check_public_release.py",
        "scripts/install_release_env.sh",
        "docs/CLAIM_TO_ARTIFACT_MAP.md",
        "docs/DATA_AVAILABILITY.md",
        "docs/RERUN_LIMITS.md",
        "docs/REVIEWER_REPRODUCTION.md",
        "benchmark/injection/overhead_h20.csv",
        "paper/main.tex",
        "paper/main.pdf",
        "paper/appendix.tex",
        "paper/appendix_supplement.pdf",
    }
    errors.extend(f"missing required release file: {p}" for p in sorted(required - tracked_set))

    forbidden_path_parts = (
        "/paper_drafts/", "/historical/", "/derivations/", "/preregistration/"
    )
    forbidden_names = {".trace", "AUDIT_STATE.md", "GAP_AUDIT.md", "PAPER_ACTIONS.md"}
    for p in tracked:
        wrapped = f"/{p}"
        if Path(p).name in forbidden_names or any(x in wrapped for x in forbidden_path_parts):
            errors.append(f"internal working-state path is tracked: {p}")
        if p.endswith(".listing"):
            errors.append(f"LaTeX intermediate is tracked: {p}")
        if "overhead_h800" in p.lower():
            errors.append(f"H20 microbenchmark has an H800 filename: {p}")

    public_text_suffixes = {".md", ".py", ".sh", ".yml", ".yaml", ".tex", ".toml", ".cff"}
    audit_id = re.compile(r"(?<![-\w])O\d{2,3}(?![-\w])")
    for p in tracked:
        path = ROOT / p
        if path.suffix.lower() not in public_text_suffixes and path.name != ".gitignore":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if p != "scripts/check_public_release.py" and re.search(r"(?i)\bH200\b", text):
            errors.append(f"obsolete H200 hardware label remains in: {p}")
        if p != "scripts/check_public_release.py" and (
            re.search(r"(?i)GAP_AUDIT|AUDIT_STATE", text) or audit_id.search(text)
        ):
            errors.append(f"internal audit notation remains in: {p}")

    main_tex = (ROOT / "paper/main.tex").read_text()
    if "1.2B/H20" not in main_tex:
        errors.append("paper/main.tex does not identify the H20 snapshot microbenchmark")
    if "H800" not in main_tex:
        errors.append("paper/main.tex does not identify the H800 replay/deployment hardware")

    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    if cff.get("repository-code") != "https://github.com/vivia-an/trainaudit-vldb-artifact":
        errors.append("CITATION.cff repository-code is not the public artifact URL")
    if len(cff.get("authors", [])) != 6:
        errors.append("CITATION.cff must list the six manuscript authors")
    if cff.get("license") != "MIT":
        errors.append("CITATION.cff license does not match LICENSE")

    markdown = [ROOT / p for p in tracked if p.endswith(".md")]
    link_pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
    for path in markdown:
        for target in link_pattern.findall(path.read_text(errors="replace")):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            destination = target.split("#", 1)[0]
            if destination and not (path.parent / destination).resolve().exists():
                errors.append(f"broken Markdown link in {path.relative_to(ROOT)}: {target}")

    if errors:
        print("public release hygiene failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        f"public release hygiene passed: {len(tracked)} tracked files, "
        f"{len(markdown)} Markdown files, six-author citation metadata"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
