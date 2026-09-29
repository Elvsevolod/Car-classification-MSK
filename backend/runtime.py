"""One frozen, query-independent path for API, export and benchmarks. No training."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .calibration import DEFAULT_CALIBRATION, load_calibration
from .core import Encoder, MODEL_NAME, PREPROCESS, ROOT, bbox, normalize, read_rows, sha256
from .frozen_encoder import POLICIES, PolicyEncoder, digest
from .gallery_repository import GalleryBuildState, InMemoryGalleryRepository
from .images import ImageIndex
from .rerank import KReciprocalReranker
from .dual_role import DualRoleEncoder, LAYOUT, ranking_blocks, validate_blocks

DEFAULT_PROFILE = "MVP_fusion_v25"
PROFILE_NAMES = ("MVP_legacy", "RC_R1_equal3_v18", "RC_R1_single_v18", "MVP_dual_role_v24", DEFAULT_PROFILE)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class Runtime:
    def __init__(self, profile=DEFAULT_PROFILE, provider="CPUExecutionProvider",
                 calibration_path=DEFAULT_CALIBRATION):
        if profile not in PROFILE_NAMES:
            raise ValueError(f"Unknown frozen profile: {profile}")
        self.profile, self.provider = profile, provider
        self.spec = json.loads((ROOT / "models/profiles.json").read_text())[profile]
        if self.spec.get("r1_weight", 0.) != (.5 if profile == "MVP_fusion_v25" else 0.):
            raise ValueError("Frozen ranking mixture differs from the approved profile")
        if self.spec["kind"] == "legacy":
            self.encoder = Encoder(provider=provider)
            self.report = load_calibration(self.encoder, calibration_path)
            if Path(calibration_path).resolve() == DEFAULT_CALIBRATION.resolve():
                self._check(calibration_path, self.spec["calibration_sha256"])
            self.preprocessing = PREPROCESS
            self.name = MODEL_NAME
        else:
            if Path(calibration_path).resolve() != DEFAULT_CALIBRATION.resolve():
                raise ValueError("R1 uses its own frozen bundle, not an MVP calibration override")
            path = ROOT / self.spec["bundle"]
            self._check(path, self.spec["bundle_sha256"])
            self.encoder = (DualRoleEncoder(path, provider) if self.spec["kind"] == "dual_role"
                            else PolicyEncoder(path, provider))
            self._check(ROOT / self.spec["metrics"], self.spec["metrics_sha256"])
            self.report = json.loads((ROOT / self.spec["metrics"]).read_text())
            bundle = self.encoder.r1.bundle if self.spec["kind"] == "dual_role" else self.encoder.bundle
            candidate_ranking = "less_graph" if self.spec["kind"] == "dual_role" else self.spec["ranking"]
            if (bundle["ranking"] != candidate_ranking
                    or bundle["candidate_policy"] != self.spec["candidate_policy"]
                    or bundle["ranking_parameters"] != POLICIES[candidate_ranking]
                    or self.report["threshold"] != bundle["threshold"]):
                raise ValueError("Frozen profile policy/metrics mismatch")
            self.preprocessing = (self.encoder.preprocessing if self.spec["kind"] == "dual_role"
                                  else self.encoder.members[0].bundle["preprocessing"])
            self.name = profile
        if (self.encoder.dimension != self.spec["dimension"] or self.encoder.size != self.spec["size"]
                or self.spec["lambda"] != POLICIES[self.spec["ranking"]]["lambda"]):
            raise ValueError("Frozen profile dimension/preprocessing/ranking mismatch")
        self.threshold = self.report["threshold"]
        self.dimension = self.encoder.dimension
        self.implementation_sha256 = {name: sha256(ROOT / "backend" / name)
                                      for name in ("core.py", "frozen_encoder.py", "runtime.py", "rerank.py", "images.py", "dual_role.py")}
        self.fingerprint = digest({"profile": profile, "spec": self.spec,
                                   "encoder": self.encoder.fingerprint, "threshold": self.threshold,
                                   "implementation": self.implementation_sha256})

    @staticmethod
    def _check(path, expected):
        if sha256(path) != expected:
            raise ValueError(f"Frozen asset checksum mismatch: {path}")

    def metadata(self):
        return {"profile": self.profile, "model": self.name, "provider": self.provider,
                "dimension": self.dimension, "image_size": self.encoder.size,
                "profile_fingerprint": self.fingerprint, "encoder_fingerprint": self.encoder.fingerprint,
                "implementation_sha256": self.implementation_sha256,
                "ranking_policy": self.spec["ranking"], "candidate_policy": self.spec["candidate_policy"],
                "frozen_threshold": self.threshold, "preprocessing": self.preprocessing,
                "l2_normalized": self.spec["kind"] != "dual_role",
                "embedding_layout": LAYOUT if self.spec["kind"] == "dual_role" else "single unit vector",
                "ranking_model": ("MVP_legacy + R1_equal3 (50/50)" if self.spec.get("r1_weight") else
                                  "MVP_legacy" if self.spec["kind"] == "dual_role" else self.profile),
                "ranking_r1_weight": self.spec.get("r1_weight", 0.) if self.spec["kind"] == "dual_role" else None,
                "candidate_model": "RC_R1_equal3_v18" if self.spec["kind"] == "dual_role" else self.profile}


def encode_rows(runtime, rows, dataset, batch_size=16, progress=None, images=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    images = images if images is not None else ImageIndex(Path(dataset) / "images")
    vectors = []
    for start in range(0, len(rows), batch_size):
        batch = []
        for row in rows[start:start + batch_size]:
            with Image.open(images.resolve(row["image_id"])) as image:
                batch.append(runtime.encoder.preprocess(image, bbox(row)))
        vectors.append(runtime.encoder.encode_batch(batch))
        if progress:
            progress(min(start + batch_size, len(rows)), len(rows))
    result = np.concatenate(vectors)
    validate_vectors(result, len(rows), runtime.dimension)
    return result


def validate_vectors(vectors, count, dimension):
    if dimension == 2048:
        if vectors.shape != (count, dimension):
            raise ValueError("Dual-role vector count/dimension mismatch")
        validate_blocks(vectors)
        return
    if (vectors.dtype != np.float32 or vectors.shape != (count, dimension)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid gallery/encoder vectors: require finite unit float32 and matching dimensions")


class ExactScorer:
    """Static gallery only. Score one query at a time, irrespective of encoder batching."""
    def __init__(self, runtime, rows, vectors):
        self.runtime, self.rows = runtime, rows
        validate_vectors(vectors, len(rows), runtime.dimension)
        # Preserve the historical schema-2 extra normalization; legacy already stores unit vectors.
        self.dual_role = runtime.spec["kind"] == "dual_role"
        if self.dual_role:
            self.vectors, self.candidate_vectors = ranking_blocks(vectors, runtime.spec.get("r1_weight", 0.))
        else:
            self.vectors = normalize(vectors) if runtime.spec["kind"] == "policy" else vectors
        setting = POLICIES[runtime.spec["ranking"]]
        self.graph = (KReciprocalReranker(self.vectors, min(20, len(rows) - 1), min(3, len(rows)))
                      if len(rows) > 1 else None)
        self.lambda_value = setting["lambda"]

    def decide(self, vector, top_k=10, threshold=None):
        if vector.shape != (self.runtime.dimension,):
            raise ValueError("Query dimension does not match the active gallery space")
        if self.dual_role:
            ranking_vector, candidate_vector = ranking_blocks(vector[None], self.runtime.spec.get("r1_weight", 0.))
            vector = ranking_vector[0]
            candidate_scores = np.clip(self.candidate_vectors @ candidate_vector[0], -1, 1)
        else:
            vector = normalize(vector)
        scores = np.clip(self.vectors @ vector, -1, 1)
        distances = self.graph.distances(vector, self.lambda_value) if self.graph else -scores
        order = np.argsort(distances, kind="stable")
        candidate_scores = candidate_scores if self.dual_role else scores
        raw = int(np.argsort(-candidate_scores, kind="stable")[0])
        confidence = float(candidate_scores[raw])
        threshold = self.runtime.threshold if threshold is None else threshold
        if not np.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Demo threshold must be finite cosine in [-1, 1]")
        positions = {int(index): rank + 1 for rank, index in enumerate(order)}

        def item(index):
            row = self.rows[int(index)]
            return {"rank": positions[int(index)], "image_id": row["image_id"],
                    **{key: row[key] for key in ("x", "y", "w", "h")},
                    "similarity": float(scores[index]), "rerank_score": -float(distances[index]),
                    "crop_url": f"/api/images/gallery/{row['image_id']}?crop=true"}

        chosen = raw if self.runtime.spec["candidate_policy"] == "raw_top1" else int(order[0])
        accepted = item(chosen) if confidence >= threshold else None
        if accepted is not None and self.dual_role:
            accepted["similarity"] = confidence  # This image and confidence belong to R1, not the ranking branch.
        return {"results": [item(int(index)) for index in order[:top_k]],
                "accepted_candidate": accepted,
                "confidence": confidence, "refused": confidence < threshold}


class RuntimeGallery:
    def __init__(self, runtime, dataset, repository=None, batch_size=16, progress=None, images=None):
        self.runtime, self.dataset = runtime, Path(dataset)
        self.rows = read_rows(self.dataset / "test_gallery.csv")
        self.images = images if images is not None else ImageIndex(self.dataset / "images")
        hashes = {r["image_id"]: sha256(self.images.resolve(r["image_id"])) for r in self.rows}
        ordered = [{"image_id": r["image_id"], "bbox": bbox(r), "sha256": hashes[r["image_id"]]}
                   for r in self.rows]
        self.fingerprint = digest({"encoder": runtime.encoder.fingerprint,
                                   "preprocessing": runtime.preprocessing,
                                   "provider": runtime.provider, "onnxruntime": ort.__version__,
                                   "implementation": runtime.implementation_sha256,
                                   "dimension": runtime.dimension, "ordered_gallery": ordered})
        self.build_state = GalleryBuildState(runtime.encoder.fingerprint, digest(runtime.preprocessing),
                                            sha256(self.dataset / "test_gallery.csv"), hashes)
        self.repository = repository if repository is not None else InMemoryGalleryRepository()
        self.vectors = self.repository.load(self.fingerprint, [r["image_id"] for r in self.rows],
                                            runtime.dimension, self.build_state)
        self.cache_hit = self.vectors is not None
        if self.vectors is None:
            self.vectors = encode_rows(runtime, self.rows, self.dataset, batch_size, progress, self.images)
            if (hashes != {r["image_id"]: sha256(self.images.resolve(r["image_id"])) for r in self.rows}
                    or sha256(self.dataset / "test_gallery.csv") != self.build_state.csv_sha256):
                raise ValueError("Gallery inputs changed during build; cache was not published")
            self.repository.replace(self.fingerprint, self.rows, self.vectors, self.build_state)
        validate_vectors(self.vectors, len(self.rows), runtime.dimension)
        self.scorer = ExactScorer(runtime, self.rows, self.vectors)

    def decide(self, vector, top_k=10, threshold=None):
        return self.scorer.decide(vector, top_k, threshold)


def export(runtime, dataset, output, batch_size=16, repository=None, progress=None):
    """Known three-file contract. No test labels, DB, training or threshold fitting required."""
    from .artifacts import validate_artifacts
    started = time.perf_counter()
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory; existing results are preserved")
    queries, rows = (read_rows(dataset / name) for name in ("test_query.csv", "test_gallery.csv"))
    csv_hashes = {name: sha256(dataset / name) for name in ("test_query.csv", "test_gallery.csv")}
    if len(rows) < 10:
        raise ValueError("Contest inference requires at least 10 gallery images for Top-10")
    if {r["image_id"] for r in queries} & {r["image_id"] for r in rows}:
        raise ValueError("Query/gallery IDs must be disjoint")
    images = ImageIndex(dataset / "images")
    image_hashes = {r["image_id"]: sha256(images.resolve(r["image_id"])) for r in queries + rows}
    gallery = RuntimeGallery(runtime, dataset, repository, batch_size, progress, images)
    query_vectors = encode_rows(runtime, queries, dataset, batch_size, progress, images)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", np.concatenate([query_vectors, gallery.vectors]).astype(np.float32))
    with (output / "submission.csv").open("w", newline="") as ranking, (output / "candidates.csv").open("w", newline="") as candidates:
        writer, accepted = csv.writer(ranking), csv.writer(candidates)
        accepted.writerow(["query_id", "gallery_id", "confidence"])
        for row, vector in zip(queries, query_vectors):
            decision = gallery.decide(vector)
            writer.writerow([row["image_id"], *[x["image_id"] for x in decision["results"]]])
            if decision["accepted_candidate"] is not None:
                confidence = decision["confidence"]
                if runtime.spec["kind"] == "legacy":
                    confidence = (confidence + 1) / 2
                accepted.writerow([row["image_id"], decision["accepted_candidate"]["image_id"], confidence])
    validation = validate_artifacts(dataset, output)
    # Detect dataset mutation while building; do not publish a successful manifest for mixed inputs.
    if image_hashes != {r["image_id"]: sha256(images.resolve(r["image_id"])) for r in queries + rows}:
        raise ValueError("Input image bytes changed during export")
    if csv_hashes != {name: sha256(dataset / name) for name in csv_hashes}:
        raise ValueError("Input annotations changed during export")
    write_json(output / "export_manifest.json", {
        **runtime.metadata(), "validation": validation, "gallery_fingerprint": gallery.fingerprint,
        "cache_hit": gallery.cache_hit, "cosine_threshold": runtime.threshold,
        "query_csv_sha256": csv_hashes["test_query.csv"],
        "gallery_csv_sha256": csv_hashes["test_gallery.csv"], "image_sha256": image_hashes,
        "embedding_ids": [r["image_id"] for r in queries + rows],
        "candidate_confidence": "(raw cosine + 1) / 2" if runtime.spec["kind"] == "legacy" else "raw cosine",
        "refusal": "no candidate row; keep exactly 10 ranked IDs",
        "export_seconds_excluding_model_load": time.perf_counter() - started})
    return validation
