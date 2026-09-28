"""Portable experiment files. No historical trainer imports or Unix-only locks."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Frozen configuration changed: {path}. Use a NEW run name.")
    else:
        write(path, value)


def child(root, relative):
    """POSIX paths in manifests, including on Windows; reject traversal/symlinks."""
    root = Path(root).resolve()
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Expected a portable relative path")
    result = (root / relative).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError(f"Path escapes root: {relative}")
    return result


def verify(root, files):
    for relative, expected in files.items():
        path = child(root, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Missing/changed input: {path}")


@contextlib.contextmanager
def lock(directory):
    """OS releases this advisory lock on process exit, on Windows and Unix."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def runtime(device):
    import importlib.metadata as metadata
    import torch
    packages = {name: metadata.version(name) for name in
                ("torch", "torchvision", "numpy", "pillow", "pandas", "onnxruntime")}
    return {"packages": packages, "python": platform.python_version(), "platform": platform.platform(),
            "device": str(device), "gpu": torch.cuda.get_device_name() if str(device) == "cuda" else None,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "precision": "float32", "amp": False, "tf32": False}


def source_hashes():
    paths = [ROOT / "evaluate.py"]
    paths += list((ROOT / "training").rglob("*.py")) + list((ROOT / "backend").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def completed(directory, signature):
    path = Path(directory) / "complete.json"
    if not path.exists():
        return False
    marker = read(path)
    if marker["signature"] != signature:
        raise ValueError("Completed stage signature changed")
    verify(directory, marker["files"])
    return True


def finish(directory, signature):
    directory = Path(directory)
    files = {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name not in {"complete.json", ".run.lock"} and not p.name.endswith(".tmp")}
    write(directory / "complete.json", {"signature": signature, "files": files})


def archive(directory, output, *, light=False):
    """Return one verified ZIP; the full result includes checkpoints AND features."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.is_relative_to(directory):
        raise ValueError("Archive must be outside the source run")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    files = {p.relative_to(directory).as_posix(): p for p in sorted(directory.rglob("*"))
             if p.is_file() and not p.name.endswith((".tmp", ".lock"))
             and not p.name.startswith("resume_")
             and not (light and p.suffix in {".pt", ".pth", ".npy", ".npz"})}
    manifest = {name: sha(p) for name, p in files.items()}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as target:
        for name, path in files.items():
            target.write(path, name)
        target.writestr("PACKAGE_MANIFEST.json", json.dumps({"light": light, "files": manifest}, indent=2))
    with zipfile.ZipFile(temporary) as target:
        if target.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        for name, expected in manifest.items():
            with target.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Archive hash mismatch: {name}")
    os.replace(temporary, output)
    write(output.with_suffix(output.suffix + ".sha256.json"), {"name": output.name, "sha256": sha(output),
                                                            "bytes": output.stat().st_size, "files": len(files)})
    print(f"ARCHIVE: {output} ({output.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return output
