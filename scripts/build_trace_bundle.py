#!/usr/bin/env python3
"""Pack the trace databases for release, with a per-file checksum manifest.

Includes the `.db.wal` sidecars. DuckDB keeps uncommitted data there, so for 7 of the 43
runs the `.db` file alone is a 12 KB header with no rows — v1 of this bundle shipped those
as empty files. Merged databases are all checkpointed and self-contained, so the guard
ablation was unaffected, but per-rank data for those runs was not usable.

    python3 scripts/build_trace_bundle.py --workspace /path/to/lsk --out /tmp/bundle
"""
import argparse
import csv
import hashlib
import pathlib
import tarfile


def runs_under(mega):
    """Every directory holding a Collector/ with trace files, including the nested clean runs."""
    out = []
    for d in sorted(mega.iterdir()):
        if not d.is_dir():
            continue
        if (d / "Collector").is_dir() and (d.name.endswith("_test_db") or "normal" in d.name):
            out.append(d)
        elif d.name == "normal_db":
            out += [s for s in sorted(d.iterdir()) if s.is_dir() and (s / "Collector").is_dir()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="the research workspace root")
    ap.add_argument("--out", required=True, help="directory to write the bundle into")
    ap.add_argument("--name", default="trainaudit-trace-dbs.tar.gz")
    args = ap.parse_args()

    ws = pathlib.Path(args.workspace)
    mega = ws / "Megatron-LM"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rec = []
    tar_path = out / args.name
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tf:
        for d in runs_under(mega):
            for f in sorted((d / "Collector").glob("*.db")) + \
                     sorted((d / "Collector").glob("*.db.wal")):
                rel = f.relative_to(mega)
                rec.append({
                    "run": d.relative_to(mega).as_posix(),
                    "kind": ("merged" if f.name.startswith("merged") else "per_rank")
                            + ("_wal" if f.name.endswith(".wal") else ""),
                    "path": rel.as_posix(),
                    "size_bytes": f.stat().st_size,
                    "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                })
                tf.add(f, arcname=rel.as_posix())

    man = out / "trace_db_manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "kind", "path", "size_bytes", "sha256"])
        w.writeheader()
        w.writerows(rec)

    wal = sum(1 for r in rec if r["kind"].endswith("_wal"))
    print(f"{len(rec)} files from {len({r['run'] for r in rec})} runs ({wal} wal sidecars)")
    print(f"{tar_path.name}: {tar_path.stat().st_size / 2**20:.1f} MiB")
    print(f"manifest: {man}")
    print(f"bundle sha256: {hashlib.sha256(tar_path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
