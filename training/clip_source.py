"""Pinned public source; downloads only the checkpoint, never a raw external dataset."""
import hashlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SOURCE_COMMIT = "eb1898b72c882875f478bebfc6d41644eece0a5d"
CHECKPOINT_URL = ("https://drive.usercontent.google.com/download?"
                  "id=1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e&export=download&confirm=t")
CHECKPOINT_BYTES = 505454649
CHECKPOINT_SHA256 = "71f6f2723be4d563f1df201133daa53f57bee2a67266486559d5ce453a60f805"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_checkpoint(destination):
    """Resumable HTTP ranges + atomic final replace; no credentials or pickle loading."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size == CHECKPOINT_BYTES:
        checksum = file_sha256(destination)
        if checksum == CHECKPOINT_SHA256:
            return checksum
        raise ValueError("Existing checkpoint checksum mismatch; do not overwrite it")
    if destination.exists():
        raise ValueError("Incomplete destination exists; use a new path, not an overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunk = 16 * 1024 * 1024
    ranges = [(i, min(i + chunk, CHECKPOINT_BYTES) - 1)
              for i in range(0, CHECKPOINT_BYTES, chunk)]

    def fetch(index):
        start, end = ranges[index]
        path = destination.with_suffix(destination.suffix + f".part{index:03d}")
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(3):
            try:
                request = urllib.request.Request(CHECKPOINT_URL, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {start}-{end}/{CHECKPOINT_BYTES}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the checkpoint byte range")
                    with path.open("wb") as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                if path.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete byte range")
                return path
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    paths = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, i): i for i in range(len(ranges))}
        for future in as_completed(futures):
            paths[futures[future]] = future.result()
            print(f"Checkpoint: {len(paths)}/{len(ranges)} parts", flush=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as target:
        for i in range(len(ranges)):
            with paths[i].open("rb") as source:
                while block := source.read(1024 * 1024):
                    target.write(block)
    checksum = file_sha256(temporary)
    if checksum != CHECKPOINT_SHA256:
        raise ValueError("Official download SHA-256 mismatch; no model was loaded")
    temporary.replace(destination)
    for path in paths.values():
        path.unlink()  # Only this downloader's exact, successfully assembled parts.
    return checksum
