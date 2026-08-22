#!/usr/bin/env python3
"""Measure the per-framework adapter sizes that fig:portability_matrix reports.

The figure labels each framework with an adapter cost — 150, 50, 150, 150 and 30 LoC. That is
the one column of paper_v2/portability.csv that matches the figure (GAP_AUDIT O17), and now
that the implementation ships (O45) it can be checked directly: the per-framework integration
code is trainaudit/adapters/<framework>.py.

    python3 benchmark/eval/adapter_loc.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ADAPTERS = ROOT / "core" / "trainaudit_pkg" / "trainaudit" / "adapters"
CLAIMED = {"megatron": ("Megatron-LM", 150), "deepspeed": ("DeepSpeed", 50),
           "olmo": ("OLMo", 150), "olmo_core": ("OLMo-core", 150), "fsdp": ("FSDP", 30)}


def main():
    if not ADAPTERS.is_dir():
        sys.exit(f"missing {ADAPTERS} — run scripts/assemble_from_workspace.sh")
    print(f"{'framework':<14}{'figure':>8}{'total':>8}{'code':>7}   note")
    flagged = []
    for stem, (label, claim) in CLAIMED.items():
        f = ADAPTERS / f"{stem}.py"
        if not f.exists():
            print(f"{label:<14}{claim:>8}{'—':>8}{'—':>7}   no adapter file")
            continue
        lines = f.read_text().splitlines()
        total = len(lines)
        code = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
        # generous: within a factor of 1.5 either way of the claim, on either count
        ok = any(claim / 1.5 <= n <= claim * 1.5 for n in (total, code))
        note = "" if ok else f"claim is {claim / max(total, 1):.1f}x the file"
        if not ok:
            flagged.append(label)
        print(f"{label:<14}{claim:>8}{total:>8}{code:>7}   {note}")

    print(f"\nCode lines exclude blanks and comment-only lines. `base.py` "
          f"({sum(1 for l in (ADAPTERS / 'base.py').read_text().splitlines())} lines) is shared "
          f"and not attributed to any framework.")
    if flagged:
        print(f"\nOut of range on both counts: {', '.join(flagged)}.")
        print("Nothing per-framework lives outside adapters/ — no other file is named for a\n"
              "framework — so this is the integration cost the figure is reporting.")
    else:
        print("\nAll adapters are within a reasonable rounding of the figure.")


if __name__ == "__main__":
    main()
