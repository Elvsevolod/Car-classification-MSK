"""Check a frozen v25 export and package its three files with provenance."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from backend.artifacts import validate_artifacts
from backend.core import MODEL, ROOT, read_rows, sha256
from backend.frozen_encoder import POLICIES
from backend.images import ImageIndex
from backend.runtime import DEFAULT_PROFILE, ExactScorer, Runtime, encode_rows, validate_vectors, write_json

PAYLOAD = ("submission.csv", "candidates.csv", "embeddings.npy")
EXPORT_FILES = (*PAYLOAD, "export_manifest.json", "runtime_timing.json")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_export(dataset, source, runtime):
    manifest = load_json(source / "export_manifest.json")
    for key, value in runtime.metadata().items():
        if manifest.get(key) != value:
            raise ValueError(f"Export differs from the current frozen runtime: {key}")
    if manifest.get("cosine_threshold") != runtime.threshold:
        raise ValueError("Export used another refusal threshold")
    queries, gallery = (read_rows(dataset / name) for name in ("test_query.csv", "test_gallery.csv"))
    if len(gallery) < 10 or {r["image_id"] for r in queries} & {r["image_id"] for r in gallery}:
        raise ValueError("Require at least 10 gallery IDs and disjoint query/gallery IDs")
    for key, filename in (("query_csv_sha256", "test_query.csv"), ("gallery_csv_sha256", "test_gallery.csv")):
        if sha256(dataset / filename) != manifest[key]:
            raise ValueError(f"Input annotations changed: {filename}")
    rows = queries + gallery
    if manifest["embedding_ids"] != [r["image_id"] for r in rows]:
        raise ValueError("Embedding ID order differs from query CSV followed by gallery CSV")
    images = ImageIndex(dataset / "images")
    hashes = {r["image_id"]: sha256(images.resolve(r["image_id"])) for r in rows}
    if hashes != manifest["image_sha256"]:
        raise ValueError("Input images differ from the export manifest")
    counts = validate_artifacts(dataset, source)
    if counts != manifest["validation"]:
        raise ValueError("Artifact counts differ from the original export")
    vectors = np.load(source / "embeddings.npy", allow_pickle=False)
    validate_vectors(vectors, len(rows), runtime.dimension)

    # Recompute six real crops to check the stored feature rows against the model.
    sample = sorted({0, len(queries) // 2, len(queries) - 1,
                     len(queries), len(queries) + len(gallery) // 2, len(rows) - 1})
    fresh = encode_rows(runtime, [rows[i] for i in sample], dataset, batch_size=3)
    error = float(np.abs(fresh - vectors[sample]).max())
    if not np.allclose(fresh, vectors[sample], atol=2e-5, rtol=0):
        raise ValueError(f"Stored embeddings differ from model output: {error}")
    replay = replay_decisions(runtime, queries, gallery, vectors, source)
    return {"passed": True, "counts": counts, "input_images_sha256_checked": len(hashes),
            "input_csv_hashes_match": True, "embedding_id_order_matches": True,
            "runtime_metadata_matches": True, "embedding_blocks_are_unit_float32": True,
            "embedding_sample": {"rows": sample, "image_ids": [rows[i]["image_id"] for i in sample],
                                 "max_abs_error": error, "atol": 2e-5,
                                 "scope": "Six rows re-encoded; remaining rows checked structurally, not re-encoded"},
            "decision_replay": replay, "hidden_test_quality_evaluated": False}


def replay_decisions(runtime, queries, gallery, vectors, source):
    with (source / "submission.csv").open(newline="") as stream:
        submission = list(csv.reader(stream))
    with (source / "candidates.csv").open(newline="") as stream:
        candidate_rows = list(csv.DictReader(stream))
    candidates = {row["query_id"]: row for row in candidate_rows}
    if len(candidates) != len(candidate_rows):
        raise ValueError("The frozen v25 exporter produces at most one candidate per query")
    scorer = ExactScorer(runtime, gallery, vectors[len(queries):])
    accepted, max_error, margin = [], 0., float("inf")
    for query, vector, ranking in zip(queries, vectors, submission):
        decision = scorer.decide(vector)
        expected = [query["image_id"], *[r["image_id"] for r in decision["results"]]]
        if ranking != expected:
            raise ValueError(f"Ranking does not match the stored embeddings: {query['image_id']}")
        candidate = candidates.get(query["image_id"])
        predicted = decision["accepted_candidate"]
        margin = min(margin, abs(decision["confidence"] - runtime.threshold))
        if (predicted is None) != (candidate is None):
            raise ValueError(f"Refusal differs from the frozen threshold: {query['image_id']}")
        if predicted is not None:
            accepted.append(query["image_id"])
            error = abs(float(candidate["confidence"]) - decision["confidence"])
            if candidate["gallery_id"] != predicted["image_id"] or not np.isfinite(error) or error > 1e-7:
                raise ValueError(f"Candidate/confidence differs from raw R1 cosine: {query['image_id']}")
            max_error = max(max_error, error)
    if [r["query_id"] for r in candidate_rows] != accepted:
        raise ValueError("Candidate rows differ from accepted query order")
    return {"queries_checked": len(queries), "top10_exact_match": True,
            "candidates_and_refusals_match": True, "confidence_max_abs_error": max_error,
            "confidence_atol": 1e-7, "minimum_distance_to_threshold": margin}


def snapshot(runtime):
    paths = {MODEL, ROOT / "models/profiles.json", ROOT / runtime.spec["bundle"],
             ROOT / runtime.spec["metrics"]}
    for member in runtime.encoder.r1.members:
        paths.add(member.bundle_path)
        paths.add((member.bundle_path.parent / member.bundle["model"]["path"]).resolve())
    return {str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(paths)}


def verify_source_kit(source_kit, runtime, assets):
    """A historical training kit may accompany optimized code, not changed weights/policies."""
    prefix = "vehicle-reid-v25-reproduction/"
    with zipfile.ZipFile(source_kit) as archive:
        if archive.testzip() is not None:
            raise ValueError("Reproduction source ZIP is damaged")
        manifest = json.loads(archive.read(prefix + "KIT_SHA256.json"))
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(prefix + name)).hexdigest() != expected:
                raise ValueError(f"Source kit checksum mismatch: {name}")
        if "metadata/recipe.json" not in manifest:
            raise ValueError("Source kit manifest omits recipe")
        recipe = json.loads(archive.read(prefix + "metadata/recipe.json"))
        if recipe["profile"] != runtime.profile:
            raise ValueError("Source kit belongs to another profile")
        for name, record in assets.items():
            if name.endswith(".onnx"):
                expected = recipe["runtime_onnx_sha256"].get(name)
            else:
                expected = manifest.get("runtime/" + name)
            if expected != record["sha256"]:
                raise ValueError(f"Source kit references another frozen asset: {name}")
    return {"name": source_kit.name, "bytes": source_kit.stat().st_size,
            "sha256": sha256(source_kit), "historical_runtime_commit": recipe["runtime_commit"],
            "verification": "kit checksums and exact current frozen model/policy asset hashes"}


def package(dataset, source, output, source_kit):
    dataset, source, output, source_kit = [Path(p).resolve() for p in (dataset, source, output, source_kit)]
    archive_path = output.with_suffix(".zip")
    if output.exists() or archive_path.exists() or archive_path.with_suffix(".zip.sha256").exists():
        raise ValueError("Use a new package name; existing packages are preserved")
    if output.is_relative_to(dataset) or output.is_relative_to(source):
        raise ValueError("Package output must be outside the dataset and original export")
    original_hashes = {name: sha256(source / name) for name in EXPORT_FILES}
    runtime = Runtime(DEFAULT_PROFILE, "CPUExecutionProvider")
    if runtime.profile != "MVP_fusion_v25":
        raise ValueError("This packaging recipe is tied to the approved v25 profile")
    assets = snapshot(runtime)
    print("Checking input hashes, formats, six model outputs and all ranking decisions...", flush=True)
    validation = verify_export(dataset, source, runtime)
    code = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "backend").glob("*.py"))}
    code.update({name: sha256(ROOT / name) for name in ("evaluate.py", "requirements.txt")})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # Runtime source and model assets must really belong to the recorded revision.
    for name, expected in {**code, **{p: v["sha256"] for p, v in assets.items()}}.items():
        recorded = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
        if hashlib.sha256(recorded).hexdigest() != expected:
            raise ValueError(f"File differs from the recorded runtime commit: {name}")
    source_kit_record = verify_source_kit(source_kit, runtime, assets)
    timing = load_json(source / "runtime_timing.json")
    if timing["profile"] != runtime.profile or timing["provider"] != runtime.provider:
        raise ValueError("Timing record belongs to another profile/provider")
    if original_hashes != {name: sha256(source / name) for name in EXPORT_FILES} or assets != snapshot(runtime):
        raise ValueError("Export or model assets changed during packaging")
    output.mkdir(parents=True, exist_ok=False)
    for name in EXPORT_FILES:
        shutil.copyfile(source / name, output / name)
        if sha256(output / name) != original_hashes[name]:
            raise ValueError(f"Copied export checksum mismatch: {name}")
    passport = {"schema": 1, "package_id": output.name,
                "packaged_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "scope": "Technical results and provenance; presentation and hosted prototype are separate",
                "profile": runtime.profile, "runtime_commit": revision,
                "runtime_source_sha256": code, "frozen_assets": assets,
                "profile_spec": runtime.spec, "ranking_parameters": POLICIES[runtime.spec["ranking"]],
                "refusal_threshold_raw_cosine": runtime.threshold, "metadata": runtime.metadata(),
                "payload": {name: {"sha256": original_hashes[name], "bytes": (output / name).stat().st_size}
                            for name in PAYLOAD}, "counts": validation["counts"], "reproduction_source_archive": source_kit_record,
                "verification_environment": {"python": platform.python_version(), "platform": platform.platform(),
                                             "onnxruntime": ort.__version__, "numpy": np.__version__,
                                             "provider": runtime.provider},
                "export_reused_without_modification": True, "fresh_full_export_timing_measured": False,
                "original_export_timing": timing,
                "packaging_tool_sha256": sha256(Path(__file__)), "dataset_images_included": False,
                "quality_claim": "No ground-truth labels for the issued test; no test mAP/F1/TNR claimed"}
    write_json(output / "version.json", passport)
    write_json(output / "validation.json", validation)
    shutil.copyfile(ROOT / "docs/SUBMISSION_RESULTS_README.md", output / "README.md")
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, path.name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Package ZIP integrity check failed")
        for name in archive.namelist():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha256(output / name):
                raise ValueError(f"ZIP member differs from the checked package: {name}")
    checksum = sha256(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive_path.name}\n")
    return {"directory": str(output), "archive": str(archive_path), "sha256": checksum,
            "counts": validation["counts"], "validation": validation["decision_replay"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path, dest="source")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-kit", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.dataset, args.source, args.output, args.source_kit), ensure_ascii=False, indent=2))
