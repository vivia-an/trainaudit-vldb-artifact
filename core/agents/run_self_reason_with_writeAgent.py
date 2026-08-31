#!/usr/bin/env python3
"""Backward-compatible entry; prefer ``python3 ../run_miner.py`` from repo root of core."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "run_miner.py"),
    run_name="__main__",
)
