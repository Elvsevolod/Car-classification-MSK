"""SQLite/PostgreSQL parity verification for the persisted Vehicle ReID gallery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import ARTIFACTS, DATASET, Encoder, Gallery, rank
from .database import DatabaseSettings
from .gallery_repository import SQLiteGalleryRepository
from .postgres_gallery_repository import PostgresGalleryRepository

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "sqlite_baseline.json"


def _raw_top(gallery: Gallery, vector: np.ndarray, limit: int) -> tuple[list[str], np.ndarray]:
    scores = gallery._raw_scores(vector)
    selected = rank(scores, limit)
    return [gallery.rows[int(position)]["image_id"] for position in selected], scores


def verify(
    fixture_path: Path = DEFAULT_FIXTURE,
    dataset: Path = DATASET,
    artifacts: Path = ARTIFACTS,
    tolerance: float = 2e-6,
) -> dict:
    """Raise AssertionError when PostgreSQL changes a frozen SQLite baseline."""
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    encoder = Encoder()
    sqlite_gallery = Gallery(
        encoder, dataset, artifacts / "gallery.sqlite3", SQLiteGalleryRepository(artifacts / "gallery.sqlite3")
    )
    postgres_gallery = Gallery(
        encoder, dataset, artifacts / "gallery.sqlite3", PostgresGalleryRepository(DatabaseSettings.from_environment())
    )

    vector_difference = float(np.max(np.abs(sqlite_gallery.vectors - postgres_gallery.vectors)))
    if vector_difference > tolerance:
        raise AssertionError(f"Persisted gallery embeddings differ by {vector_difference}, tolerance {tolerance}")

    threshold = float(fixture["threshold"])
    checks = []
    for expected in fixture["queries"]:
        vector = np.asarray(expected["embedding"], dtype=np.float32)
        sqlite_raw_ids, sqlite_scores = _raw_top(sqlite_gallery, vector, 50)
        postgres_raw_ids, postgres_scores = _raw_top(postgres_gallery, vector, 50)
        expected_raw_ids = [item["image_id"] for item in expected["raw_top_50"]]
        if sqlite_raw_ids != expected_raw_ids or postgres_raw_ids != expected_raw_ids:
            raise AssertionError(f"Raw Top-50 mismatch for query {expected['query_id']}")

        raw_difference = float(np.max(np.abs(sqlite_scores - postgres_scores)))
        if raw_difference > tolerance:
            raise AssertionError(
                f"Raw cosine differs by {raw_difference} for {expected['query_id']}, tolerance {tolerance}"
            )

        sqlite_results = sqlite_gallery.search(vector, 10)
        postgres_results = postgres_gallery.search(vector, 10)
        expected_reranked_ids = [item["image_id"] for item in expected["reranked_top_10"]]
        sqlite_ids = [item["image_id"] for item in sqlite_results]
        postgres_ids = [item["image_id"] for item in postgres_results]
        if sqlite_ids != expected_reranked_ids or postgres_ids != expected_reranked_ids:
            raise AssertionError(f"Reranked Top-10 mismatch for query {expected['query_id']}")

        expected_refused = bool(expected["refused"])
        sqlite_refused = sqlite_gallery.confidence(vector) < threshold
        postgres_refused = postgres_gallery.confidence(vector) < threshold
        if sqlite_refused != expected_refused or postgres_refused != expected_refused:
            raise AssertionError(f"Refusal mismatch for query {expected['query_id']}")

        checks.append(
            {
                "query_id": expected["query_id"],
                "raw_top_50_identical": True,
                "reranked_top_10_identical": True,
                "refusal_identical": True,
                "max_abs_raw_cosine_difference": raw_difference,
            }
        )

    return {
        "fixture": str(fixture_path),
        "queries": len(checks),
        "embedding_tolerance": tolerance,
        "max_abs_gallery_embedding_difference": vector_difference,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SQLite/PostgreSQL gallery parity")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()
    print(json.dumps(verify(args.fixture, tolerance=args.tolerance), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
