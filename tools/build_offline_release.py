"""One-command offline delivery: immutable image archive, source, licenses, hashes."""
import argparse
import gzip
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from backend.benchmark import weight_inventory
from backend.core import ROOT, sha256
from backend.runtime import write_json


def build(output, platform=None):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Use a new delivery directory outside the source repository")
    output.mkdir(parents=True, exist_ok=False)
    inventory = weight_inventory(ROOT / "models")
    if not inventory["passed"]:
        raise ValueError("Entire models directory exceeds 2 GB")
    write_json(output / "weight_inventory.json", inventory)
    tag = "vehicle-reid:release-integration"
    command = ["docker", "build", "--target", "runtime", "-t", tag]
    if platform:
        command += ["--platform", platform]
    subprocess.run([*command, "."], cwd=ROOT, check=True)
    info = json.loads(subprocess.check_output(["docker", "image", "inspect", tag], text=True))[0]
    subprocess.run(["docker", "save", "-o", str(output / "vehicle-reid.tar"), tag], check=True)
    with (output / "vehicle-reid.tar").open("rb") as source, gzip.open(output / "vehicle-reid.tar.gz", "wb") as target:
        shutil.copyfileobj(source, target)
    (output / "vehicle-reid.tar").unlink()  # Generated intermediate only; compressed copy is retained.
    paths = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    excluded = {".git", ".agents", ".idea", "artifacts", "output", "dataset", "node_modules",
                "__pycache__", ".venv", "test-results", "playwright-report"}
    with zipfile.ZipFile(output / "reproducible_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(paths)):
            path = ROOT / name
            if not name or not path.is_file() or excluded.intersection(Path(name).parts):
                continue
            if (path.suffix.lower() in {".onnx", ".pt", ".pth", ".npy", ".npz", ".zip", ".gz"}
                    and name != "example_submission/embeddings.npy"):
                continue
            if path.stat().st_size > 20_000_000:
                raise ValueError(f"Unexpected large source asset: {name}")
            archive.write(path, name)
    for source, name in ((ROOT / "docker/run-offline.sh", "run-offline.sh"),
                         (ROOT / "models/LICENSE.osnet", "LICENSE.osnet"),
                         (ROOT / "docs/RELEASE_INTEGRATION.md", "README.md"),
                         (ROOT / "release_decision.json", "release_decision.json")):
        shutil.copy2(source, output / name)
    write_json(output / "delivery_manifest.json", {
        "image": tag, "image_id": info["Id"], "architecture": info["Architecture"], "os": info["Os"],
        "native_linux_amd64_verified": False, "official_gpu_verified": False,
        "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        "weight_limit_checked_across_entire_models_directory": True,
        "dataset_included": False, "train_runtime_dependency": False})
    print(f"Offline delivery: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", help="e.g. linux/amd64; this alone does not certify native hardware")
    args = parser.parse_args()
    build(args.output, args.platform)
