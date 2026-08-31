"""Minimal collector write-shape stub (not a production training hook)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


HOOKPOINTS = (
    "before-forward",
    "after-forward",
    "after-backward",
    "before-optimizer",
    "model-after-optimizer-step",
)


@dataclass
class CollectorStub:
    rows: List[tuple] = field(default_factory=list)

    def dump(
        self,
        step: int,
        stage: str,
        name: str,
        *,
        cksum: str,
        tp: int = 0,
        dp: int = 0,
        pp: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {"name": name, "cksum": cksum, "tp": tp, "dp": dp, "pp": pp}
        if extra:
            payload.update(extra)
        self.rows.append((step, stage, json.dumps(payload)))

    def to_duckdb(self, path: str) -> None:
        import duckdb

        con = duckdb.connect(path)
        con.execute(
            "CREATE TABLE IF NOT EXISTS coredump "
            "(step INTEGER, stage VARCHAR, data JSON)"
        )
        con.execute("DELETE FROM coredump")
        con.executemany("INSERT INTO coredump VALUES (?, ?, ?)", self.rows)
        con.close()


if __name__ == "__main__":
    c = CollectorStub()
    c.dump(1, "model-after-optimizer-step", "ln.weight", cksum="x", tp=0)
    c.dump(1, "model-after-optimizer-step", "ln.weight", cksum="x", tp=1)
    print("collector_stub rows=%d" % len(c.rows))
