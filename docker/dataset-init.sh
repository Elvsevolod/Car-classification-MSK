#!/bin/sh
set -eu

# Python is bundled in the runtime; content hashing and staged refresh also work on macOS.
exec python - <<'PY'
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

source = Path(os.environ.get("SOURCE_DATASET_DIR", "/source")).resolve()
target = Path(os.environ.get("TARGET_DATASET_DIR", "/dataset-cache/dataset")).resolve()
marker_name = ".vehicle-reid-source-fingerprint"

# Never replace the source, an ancestor of it, or a filesystem root.
if target == target.parent or target == source or target in source.parents or source in target.parents:
    raise SystemExit("Dataset cache must be separate from the source and cannot be a root directory")


def fingerprint(dataset):
    if not (dataset / "images").is_dir():
        raise ValueError(f"Dataset source is missing required directory: {dataset / 'images'}")
    files = [dataset / "test_query.csv", dataset / "test_gallery.csv"]
    if (dataset / "train.csv").exists():
        files.append(dataset / "train.csv")
    files.extend(path for path in (dataset / "images").rglob("*") if path.is_file())
    signature = hashlib.sha256()
    for path in sorted(files):
        signature.update(path.relative_to(dataset).as_posix().encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            signature.update(hashlib.file_digest(stream, "sha256").digest())
    return signature.hexdigest()


source_fingerprint = fingerprint(source)
marker = target / marker_name
if marker.is_file() and marker.read_text().strip() == source_fingerprint:
    print("Prepared dataset cache is current", flush=True)
    raise SystemExit(0)

print("Preparing dataset cache inside Docker volume", flush=True)
target.parent.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
backup = None
try:
    shutil.copytree(source, staging, dirs_exist_ok=True)
    if fingerprint(staging) != source_fingerprint:
        raise ValueError("Dataset changed while copying; the previous cache was preserved")
    (staging / marker_name).write_text(source_fingerprint + "\n")
    for path in [staging, *staging.rglob("*")]:
        if os.geteuid() == 0:
            os.chown(path, 10001, 10001)
        path.chmod(0o755 if path.is_dir() else 0o644)

    if target.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.previous-", dir=target.parent))
        backup.rmdir()
        target.rename(backup)
    try:
        staging.rename(target)
    except BaseException:
        if backup is not None:
            backup.rename(target)
        raise
    if backup is not None:
        shutil.rmtree(backup)
finally:
    if staging.exists():
        shutil.rmtree(staging)
print("Prepared dataset cache is ready", flush=True)
PY
