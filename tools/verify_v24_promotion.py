"""Fresh image/export parity against v24; does not refit, alter source CSVs, or need a DB."""
import argparse
import csv
import json
from pathlib import Path

from backend.core import ROOT, read_rows, sha256
from backend.runtime import Runtime, export, write_json
from tools.release_verify import compare_exports


def verify(research, output, profile="MVP_dual_role_v24"):
    source = research / ("OSNet-AIN-x1.0/variant_25_map_search/runs/map_search_v1" if profile == "MVP_fusion_v25"
                         else "OSNet-AIN-x1.0/variant_24_dual_role/runs/dual_role_v1")
    manifest = json.loads((source/"manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    dataset = output/"validation_input"
    dataset.mkdir()
    (dataset/"images").symlink_to((research/"dataset/images").resolve(), target_is_directory=True)
    rows = {r["image_id"]: r for r in read_rows(research/"dataset/train.csv")}
    columns = ["image_id", "x", "y", "w", "h"]
    for name in ("query", "gallery"):
        with (dataset/f"test_{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: rows[i][key] for key in columns}
                             for i in manifest["protocols"]["validation"][f"{name}_ids"])
    runtime = Runtime(profile)
    counts = export(runtime, dataset, output/"export", progress=lambda n, total: print(f"{profile} images: {n}/{total}", flush=True))
    task = ("validation_" + json.loads((source/"frozen_selection.json").read_text())["selected"]["name"]
            if profile == "MVP_fusion_v25" else "fresh_validation")
    parity = compare_exports(source/"tasks"/task/"export", output/"export", atol=2e-5)
    evidence = {"profile": profile, "counts": counts, "parity": parity,
                "runtime": runtime.metadata(), "source_results_sha256": sha256(source/"results.json"),
                "fresh_images": True, "training": False, "threshold_fit": False,
                "scope": "full original validation CPU macOS; not GPU/Linux acceptance"}
    write_json(output/"verification.json", evidence)
    if not parity["passed"]:
        raise ValueError("Application does not reproduce research; do not promote or increase tolerances")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, default=ROOT.parent/"Car-classification-MSK")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("MVP_dual_role_v24", "MVP_fusion_v25"), default="MVP_dual_role_v24")
    args = parser.parse_args()
    verify(args.research.resolve(), args.output.resolve(), args.profile)
