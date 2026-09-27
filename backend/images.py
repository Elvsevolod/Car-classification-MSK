"""Resolve IDs to exactly one JPEG/PNG; filenames never enter the encoder."""
import re
from pathlib import Path


class ImageIndex:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.paths = {}
        for path in self.directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self.paths.setdefault(path.stem, []).append(path)

    def resolve(self, image_id):
        if not isinstance(image_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
            raise ValueError(f"Invalid image ID: {image_id!r}")
        matches = self.paths.get(image_id, [])
        if not matches:
            raise FileNotFoundError(f"No JPEG/PNG for image ID {image_id} in {self.directory}")
        if len(matches) != 1:
            raise ValueError(f"Ambiguous JPEG/PNG for image ID {image_id}: {[p.name for p in matches]}")
        return matches[0]
