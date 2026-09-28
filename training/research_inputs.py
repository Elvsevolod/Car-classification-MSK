"""Build a LOCAL, hash-verified input bundle for the second computer; never upload data."""
import argparse
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
from pathlib import Path
import shutil
import json
import zipfile

from training import research_io as io

V = io.ROOT / "OSNet-AIN-x1.0"
DESTINATION = io.ROOT / "research_transfer/v41_inputs"


def build(destination=DESTINATION):
    from backend.core import read_rows
    from training.transreid_model import image_paths, verify_weights

    destination = Path(destination)
    old = V / "variant_16_review_protocol/runs/review_v1"
    m16 = io.read(old / "manifest.json")
    m33 = io.read(V / "variant_33_nive_low_aux/runs/low_aux_v1/manifest.json")
    r33 = io.read(V / "variant_33_nive_low_aux/runs/low_aux_v1/results.json")
    splits = io.read(io.ROOT / "artifacts/splits.json")
    partition = m16["inner"]["primary"]
    train, held = set(partition["train"]), set(partition["validation"])
    if train & held or train | held != set(splits["identities"]["train"]) or m33["inner"] != partition:
        raise ValueError("Historical primary identity partitions disagree")
    csv = io.ROOT / "dataset/train.csv"
    if io.sha(csv) != splits["train_csv_sha256"]:
        raise ValueError("Original annotations changed")
    rows = [r for r in read_rows(csv) if r["vehicle_id"] in train | held]
    paths = image_paths(io.ROOT / "dataset", rows)
    files = {}

    def copy(source, relative, expected=None):
        source = Path(source)
        checksum = io.sha(source)
        if expected is not None and checksum != expected:
            raise ValueError(f"Historical source changed: {source}")
        target = io.child(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if io.sha(target) != checksum:
                raise ValueError(f"Different existing transfer file: {target}")
        else:
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            if io.sha(temporary) != checksum:
                raise ValueError("Transfer copy checksum mismatch")
            temporary.replace(target)
        files[relative] = checksum
        return relative

    copy(csv, "provenance/original_train.csv", splits["train_csv_sha256"])
    copy(old / "manifest.json", "provenance/v16_manifest.json")
    copy(old / "selection.json", "provenance/v16_selection.json")
    copy(V / "variant_33_nive_low_aux/runs/low_aux_v1/manifest.json", "provenance/v33_manifest.json")
    for row in rows:
        row["path"] = copy(paths[row["image_id"]], f"images/{paths[row['image_id']].name}",
                           splits["frame_sha256"][row["image_id"]])
    # Full-frame hashes, not just IDs, must be disjoint across the fold.
    a, b = ({files[r["path"]] for r in rows if r["vehicle_id"] in ids} for ids in (train, held))
    if a & b:
        raise ValueError("Exact image leakage across primary fold")
    models = {}
    for name, variant, seed in [("B0", "B0_control", 20260915)] + [
            (f"R1_{s}", "R1_resolution256", s) for s in (20260915, 20260916, 20260917)]:
        summary_path = old / f"primary/{variant}/seed_{seed}/summary.json"
        summary = io.read(summary_path)
        if summary["fold"] != "primary" or summary["train_identities"] != len(train):
            raise ValueError("Baseline checkpoint is not fold-matched")
        entry = summary["checkpoints"]["800"]
        models[name] = {"path": copy(old / entry["path"], f"weights/{name}.pt", entry["sha256"]),
                        "sha256": entry["sha256"], "size": 208 if name == "B0" else 256,
                        "signature": summary["signature"], "step": 800, "train_ids": sorted(train)}
        copy(summary_path, f"provenance/{name}_summary.json")
    n1 = r33["training"]["N1_nive_mixed"]["checkpoints"]["1800"]
    n1path = V / "variant_33_nive_low_aux/runs/low_aux_v1/training/N1_nive_mixed/step_01800.pt"
    models["N1"] = {"path": copy(n1path, "weights/N1.pt", n1["sha256"]), "sha256": n1["sha256"],
                    "size": 256, "signature": r33["signature"], "step": 1800, "train_ids": sorted(train)}
    deit = verify_weights()
    copy(deit, "weights/deit_small_distilled_patch16_224-649709d9.pth")
    external = []
    confirmed = io.read(V / "variant_15_nive_transfer/runs/nive_pilot_v1/manifest.json")
    if not confirmed["nive"]["source"]["local_copy_source_confirmed_by_user"]:
        raise ValueError("NiVe provenance not confirmed")
    organizer_hashes = set(splits["frame_sha256"].values())
    for relative, checksum in sorted(m33["nive"]["files"].items()):
        if not relative.startswith("train/"):
            continue
        if checksum in organizer_hashes:
            raise ValueError("NiVe/organizer image overlap")
        path = Path(relative)
        identity, view = path.parent.name, path.stem.split("_")[0]
        external.append({"image_id": f"nive/{identity}/{path.stem}", "vehicle_id": f"nive:{identity}",
                         "camera_id": view, "path": copy(io.ROOT / "NiVe1303" / relative,
                                                         f"nive/{relative}", checksum)})
    if len(external) != 17070 or len({r["vehicle_id"] for r in external}) != 703:
        raise ValueError("Expected all 17070 NiVe train photos / 703 identities")
    manifest = {"schema": "reid-portable-inputs-v1", "rows": rows, "external": external,
                "train_ids": sorted(train), "holdout_ids": sorted(held), "models": models,
                "draws": m33["draws"], "config": m33["config"], "nive_source": m33["nive"]["source"],
                "files": files, "original_train_csv_sha256": splits["train_csv_sha256"],
                "baseline": "C_primary = B0_208 step800 + equal3 R1_256 step800, cosine 50/50; NOT release v25",
                "scope": "primary development only; no outer/test images, no promotion, no threshold fit"}
    io.freeze(destination / "inputs.json", manifest)
    io.verify(destination, files)
    print(f"INPUTS READY: {len(rows)} organizer + {len(external)} NiVe; {len(models)} fixed checkpoints", flush=True)
    return destination


def load(directory):
    directory = Path(directory).resolve()
    m = io.read(directory / "inputs.json")
    if m["schema"] != "reid-portable-inputs-v1" or set(m["train_ids"]) & set(m["holdout_ids"]):
        raise ValueError("Invalid portable identity protocol")
    allowed, held = set(m["train_ids"]), set(m["holdout_ids"])
    if {r["vehicle_id"] for r in m["rows"]} != allowed | held:
        raise ValueError("Missing/extra input identities")
    by_id = {r["image_id"]: r for r in m["rows"]}
    if len(by_id) != len(m["rows"]):
        raise ValueError("Duplicate image IDs")
    for protocol in m["draws"].values():
        if set(protocol["query_ids"]) & set(protocol["gallery_ids"]) or any(
                by_id[i]["vehicle_id"] not in held for i in protocol["query_ids"] + protocol["gallery_ids"]):
            raise ValueError("Train/holdout protocol leakage")
    for model in m["models"].values():
        if set(model["train_ids"]) != allowed or m["files"][model["path"]] != model["sha256"]:
            raise ValueError("Baseline provenance differs from fold")
    if any(not r["path"].startswith("nive/train/") for r in m["external"]):
        raise ValueError("Only external TRAIN images allowed")
    io.verify(directory, m["files"])
    return directory, m


def unpack(archive, destination=DESTINATION):
    """Safe resumable Windows extraction, verified against the package pin in Git."""
    archive, destination = Path(archive), Path(destination)
    pin = io.read(V / "variant_41_windows_queues/INPUT_PACKAGE.json")
    if archive.stat().st_size != pin["bytes"] or io.sha(archive) != pin["sha256"]:
        raise ValueError("Wrong/incomplete input archive; compare SHA256, do not bypass the guard")
    with zipfile.ZipFile(archive) as z:
        package = json.loads(z.read("PACKAGE_MANIFEST.json"))
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(package["files"]) | {"PACKAGE_MANIFEST.json"}:
            raise ValueError("Unexpected/duplicate archive members")
        for number,(name,expected) in enumerate(package["files"].items(),1):
            target = io.child(destination,name)
            if target.exists():
                if io.sha(target) != expected: raise ValueError(f"Different existing file: {target}")
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary = target.with_suffix(target.suffix+".tmp")
            with z.open(name) as source, temporary.open("wb") as result: shutil.copyfileobj(source,result)
            if io.sha(temporary) != expected: raise ValueError(f"Unpacked checksum mismatch: {name}")
            temporary.replace(target)
            if number % 1000 == 0: print(f"UNPACK {number}/{len(package['files'])}",flush=True)
    load(destination)
    print(f"VERIFIED INPUTS: {destination}",flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--unpack",type=Path)
    args = parser.parse_args()
    result = unpack(args.unpack,args.output) if args.unpack else build(args.output)
    if args.zip and not args.unpack:
        io.archive(result, result.with_suffix(".zip"))
