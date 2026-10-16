"""
write_data_manifest.py
----------------------
Records the state of the pipeline's input data in a manifest (paths.manifest), and
checks the data against it later.

The manifest lists every file under the inputs below, with its size and SHA-256
checksum. If a file changes in any way, its checksum changes.

  Checksummed (inputs the pipeline reads):
    paths.inputs.training_set            training table
    paths.generated.climate_monthly_gcm  monthly climate per GCM (projections)
    paths.raw.static_vars                static layers
    paths.raw.fire_records               fire cell-months
    paths.inputs.boundaries              administrative boundaries for maps
  Size only (raw source data, too large to checksum routinely):
    paths.raw.climate_gcms               daily climate per GCM

Paths in the manifest are relative to paths.data_dir.

Usage:
    python write_data_manifest.py write    # (re)write the manifest
    python write_data_manifest.py verify   # compare the data with the manifest
"""

import csv
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "utils"))
from load_config import cfg

DATA_DIR = cfg.paths.data_dir
CHECKSUMMED = [
    cfg.paths.inputs.training_set,
    cfg.paths.generated.climate_monthly_gcm,
    cfg.paths.raw.static_vars,
    cfg.paths.raw.fire_records,
    cfg.paths.inputs.boundaries,
]
SIZE_ONLY = [cfg.paths.raw.climate_gcms]
FIELDS = ["path", "size_bytes", "sha256"]
CHUNK = 8 * 1024 * 1024


def list_files(root):
    """All files under `root` (or `root` itself), as paths relative to DATA_DIR."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input not found: {root}")
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    return sorted(str(p.relative_to(DATA_DIR)) for p in files)


def sha256(rel):
    h = hashlib.sha256()
    with open(DATA_DIR / rel, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def scan():
    """Current state of the inputs: {path: (size, sha256 or "")}."""
    checksummed = [f for root in CHECKSUMMED for f in list_files(root)]
    size_only = [f for root in SIZE_ONLY for f in list_files(root)]
    unfinished = [f for f in checksummed if f.endswith(".tmp")]
    if unfinished:
        raise RuntimeError(f"{len(unfinished)} unfinished .tmp files, e.g. {unfinished[0]}; "
                           "let the writing job finish first.")

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", min(8, os.cpu_count() or 1)))
    print(f"Checksumming {len(checksummed):,} files with {workers} threads; "
          f"recording sizes of {len(size_only):,} files")
    with ThreadPoolExecutor(workers) as pool:
        hashes = dict(zip(checksummed, pool.map(sha256, checksummed)))
    state = {f: ((DATA_DIR / f).stat().st_size, hashes[f]) for f in checksummed}
    state.update({f: ((DATA_DIR / f).stat().st_size, "") for f in size_only})
    return state


def write():
    state = scan()
    with open(cfg.paths.manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for path in sorted(state):
            w.writerow([path, *state[path]])
    total = sum(size for size, _ in state.values())
    print(f"Written: {cfg.paths.manifest} ({len(state):,} files, {total / 1e9:,.1f} GB)")


def verify():
    with open(cfg.paths.manifest, newline="") as f:
        recorded = {r["path"]: (int(r["size_bytes"]), r["sha256"]) for r in csv.DictReader(f)}
    current = scan()
    missing = sorted(set(recorded) - set(current))
    new = sorted(set(current) - set(recorded))
    changed = sorted(p for p in set(recorded) & set(current) if recorded[p] != current[p])

    for label, paths in [("MISSING", missing), ("NEW", new), ("CHANGED", changed)]:
        for p in paths:
            print(f"  {label:<8} {p}")
    if missing or new or changed:
        print(f"Data differ from the manifest: {len(changed)} changed, {len(missing)} missing, {len(new)} new.")
        sys.exit(1)
    print(f"Data match the manifest ({len(recorded):,} files).")


if __name__ == "__main__":
    modes = {"write": write, "verify": verify}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        print(__doc__)
        sys.exit(1)
    modes[sys.argv[1]]()
