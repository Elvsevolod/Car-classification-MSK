"""Separate test-only web service with a disposable file cache, no production DB."""
import argparse
from pathlib import Path
import uvicorn
from backend.app import create_app
from backend.cache_spaces import FileGallerySpaces
from backend.runtime import DEFAULT_PROFILE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    args = parser.parse_args()
    uvicorn.run(create_app(dataset=args.dataset, gallery_repository=FileGallerySpaces(args.cache),
                           profile=args.profile, provider="CPUExecutionProvider"), host="127.0.0.1", port=args.port)
