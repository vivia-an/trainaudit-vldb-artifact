#!/usr/bin/env python3
"""Merge Phase-1 static/B-light seed rules into holdout lib (template-level, not case evidence)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "config" / "lib_holdout_mined.json"

SEEDS = {
    "model_integrity": {
        "layer-count-matches-config": {
            "name": "配置num_layers与参数名层索引数量一致性检查",
            "description": "静态结构检查(P6)：从coredump参数名解析transformer层索引个数，须与训练配置num_layers一致。不依赖T3注入；用于检测层数/分片结构静默错误。",
            "type": "validity",
            "logic": "",
            "tables": ["coredump"],
            "params": {},
            "applicable_conditions": {
                "stage": "IN ('model-after-backward', 'model-after-forward', 'initialization')",
                "check": "count(distinct layer_index from param_name) == config.num_layers",
            },
        },
        "emb-init-std-bound": {
            "name": "embedding初始化std与d_model关系检查",
            "description": "静态/初始化检查(P9)：embedding权重在initialization阶段的std应满足配置声明的缩放（如与sqrt(d_model)相关）。非T3注入路径。",
            "type": "validity",
            "logic": "",
            "tables": ["coredump"],
            "params": {},
            "applicable_conditions": {
                "stage": "= 'initialization' OR LIKE 'model-after-initialization%'",
                "param_name": "LIKE '%embed%'",
            },
        },
        "checkpoint-preserve-rng": {
            "name": "checkpoint save/load后RNG状态一致性检查",
            "description": "B-light：比较type=rng_state在checkpoint-save与checkpoint-load的cpu/cuda_rng_state_cksum须一致。字段来自dump_rng_state埋点。",
            "type": "consistency",
            "logic": "",
            "tables": ["coredump"],
            "params": {},
            "applicable_conditions": {
                "stage": "IN ('checkpoint-save', 'checkpoint-load')",
                "type": "= 'rng_state'",
            },
        },
        "process-group-size-correct": {
            "name": "并行process group规模与拓扑元数据一致性检查",
            "description": "B-light：parallel_metadata中的dp/tp/pp/world_size须与启动拓扑一致且组规模合法。字段来自dump_parallel_metadata。",
            "type": "validity",
            "logic": "",
            "tables": ["coredump"],
            "params": {},
            "applicable_conditions": {
                "stage": "IN ('parallel-metadata', 'initialization')",
                "type": "= 'parallel_metadata'",
            },
        },
    }
}


def main() -> int:
    lib_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LIB
    d = json.loads(lib_path.read_text())
    cons = d.setdefault("constraints", {})
    added = 0
    for cat, rules in SEEDS.items():
        bucket = cons.setdefault(cat, {})
        if not isinstance(bucket, dict):
            print(f"skip {cat}: not a dict")
            continue
        for key, rule in rules.items():
            if key in bucket:
                print(f"exists {cat}/{key} (keep existing)")
                continue
            bucket[key] = rule
            added += 1
            print(f"added {cat}/{key}")
    # Additive only: skip rewrite when nothing new (avoids racing category mining writers)
    if added == 0:
        print(f"done added=0 (no write) path={lib_path}")
        return 0
    bak = lib_path.with_suffix(lib_path.suffix + ".bak_pre_phase1_seed")
    if not bak.exists():
        bak.write_text(lib_path.read_text())
        print(f"backup {bak}")
    meta = d.setdefault("metadata", {})
    meta["phase1_template_seeds"] = True
    lib_path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
    print(f"done added={added} path={lib_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
