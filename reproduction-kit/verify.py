"""Verify the portable kit and organizer training data without ML dependencies."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_kit(root=ROOT):
    manifest = read_json(root / "KIT_SHA256.json")
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Kit file missing or changed: {relative}")
    source = root / "source"
    historical = read_json(source / "SOURCE_SHA256.json")
    for relative, record in historical.items():
        expected = record.get("sha256", record.get("code_only_sha256"))
        if sha256(source / relative) != expected:
            raise ValueError(f"Historical source changed: {relative}")
    # These are the hashes recorded during the actual R1 training, not a new snapshot.
    for variant, run in [(16, "review_v1"), (17, "final_seeds_v1")]:
        folder = next((source / "OSNet-AIN-x1.0").glob(f"variant_{variant}_*"))
        training = read_json(folder / "runs" / run / "manifest.json")
        for relative, expected in training["source_sha256"].items():
            if sha256(source / relative) != expected:
                raise ValueError(f"Training-time source mismatch: {relative}")
    return {"kit_files": len(manifest), "historical_files": len(historical)}


def verify_dataset(dataset, root=ROOT):
    dataset = Path(dataset).resolve()
    split = read_json(root / "metadata/splits.json")
    if sha256(dataset / "train.csv") != split["train_csv_sha256"]:
        raise ValueError("train.csv differs from the frozen organizer annotations")
    with (dataset / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if {r["image_id"] for r in rows} != set(split["frame_sha256"]):
        raise ValueError("Training image IDs differ from the frozen split")
    partitions = {name: set(ids) for name, ids in split["identities"].items()}
    if sum(map(len, partitions.values())) != len(set.union(*partitions.values())):
        raise ValueError("An identity appears in multiple partitions")
    if {int(r["vehicle_id"]) for r in rows} != set.union(*partitions.values()):
        raise ValueError("Training identities differ from the frozen split")
    owner = {identity: name for name, ids in partitions.items() for identity in ids}
    frame_owner = {}
    for row in rows:
        identifier = row["image_id"]
        digest = sha256(dataset / "images" / f"{identifier}.jpg")
        if digest != split["frame_sha256"][identifier]:
            raise ValueError(f"Training image changed: {identifier}")
        partition = owner[int(row["vehicle_id"])]
        if frame_owner.setdefault(digest, partition) != partition:
            raise ValueError("An identical frame appears in multiple partitions")
    # No test CSV or test image is read here or by the training runner.
    return {"train_csv_rows": len(rows), "identities": {n: len(v) for n, v in partitions.items()},
            "train_csv_sha256": split["train_csv_sha256"], "verified_frames": len(frame_owner)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    result = {"kit": verify_kit()}
    if args.dataset:
        result["dataset"] = verify_dataset(args.dataset)
    print(json.dumps(result, indent=2))
