"""Compatibility with the exact organizer files, including their edge cases."""
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import evaluate as official
from backend.core import ROOT, read_rows
from backend.evaluate import validate_artifacts
from backend.scoring import metrics, ranked_queries


def test_organizer_files_are_unmodified():
    expected = {
        "evaluate.py": "655c71db8c2e4d2cd7680c40c768afacfdffff360401111c1a46df921551ffa3",
        "example_submission/candidates.csv": "497e9056becf29b4247ff17f946982d72336c81001ed3b280ccd4ab44d60d681",
        "example_submission/embeddings.npy": "ec32cb17212bfac86e9d620f935991d5a34d2dcba313e9b0c5147df1a823c171",
        "example_submission/submission.csv": "d088c462ecb0596fa3e634ab85d459f44a2fc84ae258cf02f7045f6f41edb9e9",
        "example_submission/test_gallery.csv": "362205f61b6e0c503b9c7800f48f4d17259809023f966cdd6a40a7078e439c42",
        "example_submission/test_query.csv": "555f9584fdf141520c77b68b2c60bb4f226c0960463804f3322e1972e21cfb0f",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_official_example_passes_artifact_validation():
    example = ROOT / "example_submission"
    result = validate_artifacts(example, example)
    assert result == {"queries": 5, "gallery": 8, "embedding_shape": [13, 16],
                      "embedding_dtype": "float32", "submission_rows": 5,
                      "candidate_rows": 4, "accepted_queries": 4, "refused_queries": 1}


def test_ids_cannot_escape_image_directory(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("image_id,x,y,w,h\n../outside,0,0,1,1\n")
    with pytest.raises(ValueError, match="Invalid"):
        read_rows(path)


def test_top10_is_truncated_before_organizer_junk_filter():
    queries = [{"image_id": "q", "vehicle_id": 1, "camera_id": 1}]
    gallery = [{"image_id": "junk", "vehicle_id": 1, "camera_id": 1}]
    gallery += [{"image_id": f"n{i}", "vehicle_id": 2, "camera_id": 2} for i in range(9)]
    gallery += [{"image_id": "positive", "vehicle_id": 1, "camera_id": 2}]
    embeddings = {row["image_id"]: [1, 0] for row in queries + gallery}
    result = metrics(ranked_queries(queries, gallery, embeddings), .5)
    assert result["mAP_at_10"] == 0
    assert result["full_mAP"] == pytest.approx(.1)
    assert result["TP"] == 1  # Same-camera top candidate is accepted by official code.


def test_official_pr_auc_censoring_and_undefined_values():
    labels = np.array([1, 1, 0, 0])
    assert official.pr_auc(np.array([.9, .7, .8, .6]), labels) == pytest.approx(5/6)
    assert official.pr_auc(np.array([.9, -np.inf, -np.inf, -np.inf]), labels) == 1
    assert np.isnan(official.pr_auc(np.full(4, -np.inf), labels))


@pytest.mark.parametrize("seed", range(10))
def test_adapter_matches_official_functions_with_junk_and_reranking(seed):
    rng = np.random.default_rng(seed)
    queries = [{"image_id": f"q{i}", "vehicle_id": i, "camera_id": i % 2} for i in range(6)]
    gallery = [{"image_id": f"g{i}", "vehicle_id": i % 4, "camera_id": i % 3} for i in range(24)]
    vectors = rng.normal(size=(30, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    embeddings = {r["image_id"]: v for r, v in zip(queries + gallery, vectors)}
    scores = rng.random((6, 24))  # A reranked order independent from raw cosine.
    ranked = ranked_queries(queries, gallery, embeddings, scores)
    confidence = rng.random(6)
    candidates = {qid: [(ranked.predictions[qid][0], float(s))]
                  for qid, s in zip(ranked.query.index, confidence) if s >= .5}
    actual = metrics(ranked, .5, confidence)
    expected = official.candidate_metrics(ranked.query, ranked.gallery, candidates)
    for local, key in [("candidate_F1", "F1"), ("TNR", "TNR"), ("PR_AUC", "PR-AUC"),
                       ("TP", "TP"), ("FP", "FP"), ("FN", "FN"), ("TN", "TN")]:
        assert actual[local] == pytest.approx(expected[key])
    assert actual["mAP_at_10"] == official.ranking_metrics(
        ranked.query, ranked.gallery, ranked.predictions)["mAP@10"]
    q = vectors[:6] / np.linalg.norm(vectors[:6], axis=1, keepdims=True)
    g = vectors[6:] / np.linalg.norm(vectors[6:], axis=1, keepdims=True)
    full = official.full_ranking_metrics(q, g, list(ranked.query.index), list(ranked.gallery.index),
                                       ranked.query, ranked.gallery)
    assert actual["full_mAP"] == pytest.approx(full["mAP_full"])
    assert actual["mINP"] == pytest.approx(full["mINP"])
    all_refused = metrics(ranked, 2, confidence)
    assert all_refused["PR_AUC"] is None
    json.dumps(all_refused, allow_nan=False)


def test_official_cli_on_example_with_explicit_synthetic_ground_truth(tmp_path):
    # The organizer example has no GT. This is synthetic test data, not a quality claim.
    example = ROOT / "example_submission"
    q = pd.read_csv(example / "test_query.csv")
    g = pd.read_csv(example / "test_gallery.csv")
    gt = tmp_path / "synthetic_gt.csv"
    with gt.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "vehicle_id", "camera_id", "split"])
        writer.writerows([i, n, 0, "query"] for n, i in enumerate(q.image_id))
        writer.writerows([i, n % 4, 1, "gallery"] for n, i in enumerate(g.image_id))
    output = tmp_path / "report.json"
    subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--gt", str(gt),
                    "--submission", str(example / "submission.csv"),
                    "--candidates", str(example / "candidates.csv"),
                    "--embeddings", str(example / "embeddings.npy"),
                    "--query", str(example / "test_query.csv"),
                    "--gallery", str(example / "test_gallery.csv"), "--json", str(output)],
                   check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["ranking"]["n_scored"] == 4
    assert report["ranking"]["n_openset_excluded"] == 1
    assert set(report) == {"ranking", "full_ranking", "candidates"}
