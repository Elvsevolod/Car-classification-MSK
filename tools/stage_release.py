"""Copy only frozen inference assets; never edit the research/application sources."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def stage(source, application, target):
    source, application, target = (p.resolve() for p in (source, application, target))
    if (target / "models/profiles.json").exists() and "MVP_dual_role_v24" in json.loads((target / "models/profiles.json").read_text()):
        raise ValueError("Do not restage legacy assets over an approved v24 profile; use a new target")
    inventory = {}

    def copy(path, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != sha(path):
            raise ValueError(f"Refusing to overwrite different asset: {destination}")
        shutil.copy2(path, destination)
        inventory[str(destination.relative_to(target))] = {
            "source": str(path), "sha256": sha(path), "bytes": path.stat().st_size}

    def bundle(path):
        data = json.loads(path.read_text())
        destination = target / "models/frozen" / path.relative_to(source)
        copy(path, destination)
        if data["schema"] == 2:
            for member in data["members"]:
                child = (path.parent / member["path"]).resolve()
                assert sha(child) == member["sha256"]
                bundle(child)
        else:
            child = (path.parent / data["model"]["path"]).resolve()
            assert sha(child) == data["model"]["sha256"]
            copy(child, target / "models/frozen" / child.relative_to(source))
        return str(destination.relative_to(target)), sha(path)

    for name in ("osnet_ain_x1_0_vehicle_reid_hpo_best_map.onnx",
                 "osnet_ain_x1_0_vehicle_reid.onnx", "calibration.json", "LICENSE.osnet"):
        copy(application / "models" / name, target / "models" / name)
    base = source / "OSNet-AIN-x1.0/variant_18_retrieval_policy/runs/policy_v1"
    final = json.loads((base / "final.json").read_text())
    profiles = {"MVP_legacy": {"kind": "legacy", "size": 208, "dimension": 512,
                "calibration": "models/calibration.json", "calibration_sha256": sha(application / "models/calibration.json"),
                "ranking": "legacy", "candidate_policy": "ranking_top1", "lambda": .5}}
    for name, key, dimension in (("RC_R1_equal3_v18", "R1_equal3", 1536),
                                 ("RC_R1_single_v18", "R1_resolution256_20260915", 512)):
        case = f"{key}/less_graph/raw_top1"
        path, checksum = bundle(base / "final" / case / "bundle.json")
        metrics = next(x for x in final["cases"] if x["case"] == case)
        assert metrics["bundle_sha256"] == checksum
        report = {"model": name, "threshold": metrics["threshold"],
                  "validation": {"mAP_at_10": metrics["ranking"]["mAP@10"],
                                 "candidate_F1": metrics["candidates"]["F1"],
                                 "TNR": metrics["candidates"]["TNR"],
                                 "known_queries": metrics["ranking"]["n_scored"],
                                 "unknown_queries": metrics["ranking"]["n_openset_excluded"]},
                  "provenance": {"source": str((base / "final.json").relative_to(source)),
                                 "source_sha256": sha(base / "final.json"), "case": case,
                                 "bundle_sha256": checksum, "kind": "historical local validation, not hidden test"}}
        report_path = target / "models" / f"{name}.metrics.json"
        write(report_path, report)
        profiles[name] = {"kind": "policy", "size": 256, "dimension": dimension,
                          "bundle": path, "bundle_sha256": checksum,
                          "ranking": "less_graph", "candidate_policy": "raw_top1", "lambda": .75,
                          "metrics": str(report_path.relative_to(target)), "metrics_sha256": sha(report_path)}
    write(target / "models/profiles.json", profiles)
    write(target / "docs/ASSET_PROVENANCE.json", inventory)
    # Snapshot code separately from runtime weights/data. This also captures dirty files.
    paths = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"],
                                    cwd=application).decode().split("\0")
    snapshot = {name: sha(application / name) for name in sorted(set(paths))
                if name and (application / name).is_file()}
    write(target / "docs/SOURCE_APPLICATION_SNAPSHOT.json", {
        "source": str(application), "head": subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=application).decode().strip(), "files": snapshot,
        "status": subprocess.check_output(["git", "status", "--porcelain"], cwd=application).decode()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    stage(args.research, args.application, args.target)
