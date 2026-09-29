"""Build a ZIP from the verified file manifest; never include data or run outputs."""
import argparse
import zipfile
from pathlib import Path

from verify import ROOT, read_json, sha256, verify_kit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify_kit()
    files = [*read_json(ROOT / "KIT_SHA256.json"), "KIT_SHA256.json"]
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(files):
            path = ROOT / relative
            if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
                raise ValueError(f"Path escapes the kit: {relative}")
            archive.write(path, f"vehicle-reid-v25-reproduction/{relative}")
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(files):
            raise ValueError("ZIP integrity check failed")
    checksum = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{checksum}  {args.output.name}\n")
    print(f"{len(files)} files, {args.output.stat().st_size} bytes, SHA256 {checksum}")


if __name__ == "__main__":
    main()
