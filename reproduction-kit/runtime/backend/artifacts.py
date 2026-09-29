"""Submission validation only; no calibration, training or database imports."""
import csv
import numpy as np
from .core import DATASET, ARTIFACTS, read_rows

CANDIDATES_HEADER = ["query_id", "gallery_id", "confidence"]

def _read_strict_csv(path, expected_header):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != expected_header:
            raise ValueError(f"Invalid header in {path.name}: {header}; expected {expected_header}")
        rows = list(reader)
    if any(len(row) != len(expected_header) for row in rows):
        raise ValueError(f"Invalid column count in {path.name}")
    return rows


def validate_artifacts(dataset=DATASET, output=ARTIFACTS):
    """Validate the three submission artifacts against the published contract."""
    query_ids = [row["image_id"] for row in read_rows(dataset / "test_query.csv")]
    gallery_ids = [row["image_id"] for row in read_rows(dataset / "test_gallery.csv")]
    query_set = set(query_ids)
    gallery_set = set(gallery_ids)

    embeddings = np.load(output / "embeddings.npy", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(query_ids) + len(gallery_ids):
        raise ValueError("embeddings.npy must be a 2D array with query rows followed by gallery rows")
    if embeddings.dtype != np.float32 or embeddings.shape[1] < 1:
        raise ValueError("embeddings.npy must have dtype float32 and a non-empty embedding dimension")
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12):
        raise ValueError("embeddings.npy contains non-finite or zero vectors")

    with open(output / "submission.csv", newline="", encoding="utf-8-sig") as stream:
        submission = list(csv.reader(stream))
    top_k = min(10, len(gallery_ids))  # Organizer example has only eight gallery objects.
    if any(len(row) != top_k + 1 for row in submission):
        raise ValueError("submission.csv must have no header and query_id plus Top-K gallery IDs")
    if [row[0] for row in submission] != query_ids:
        raise ValueError("submission.csv must contain every query exactly once in test_query.csv order")
    for row in submission:
        candidates = row[1:]
        if len(set(candidates)) != top_k or not set(candidates).issubset(gallery_set):
            raise ValueError(f"submission.csv query {row[0]} must contain {top_k} distinct gallery IDs")

    candidates = _read_strict_csv(output / "candidates.csv", CANDIDATES_HEADER)
    seen_pairs = set()
    accepted_queries = set()
    for query_id, gallery_id, confidence in candidates:
        if query_id not in query_set or gallery_id not in gallery_set:
            raise ValueError("candidates.csv contains an unknown or empty query/gallery ID")
        try:
            score = float(confidence)
        except ValueError as error:
            raise ValueError("candidates.csv confidence must be numeric") from error
        if not np.isfinite(score):
            raise ValueError("candidates.csv confidence must be finite")
        pair = query_id, gallery_id
        if pair in seen_pairs:
            raise ValueError(f"Duplicate candidate pair: {query_id}, {gallery_id}")
        seen_pairs.add(pair)
        accepted_queries.add(query_id)
    return {"queries": len(query_ids), "gallery": len(gallery_ids),
            "embedding_shape": list(embeddings.shape), "embedding_dtype": str(embeddings.dtype),
            "submission_rows": len(submission), "candidate_rows": len(candidates),
            "accepted_queries": len(accepted_queries), "refused_queries": len(query_ids) - len(accepted_queries)}
