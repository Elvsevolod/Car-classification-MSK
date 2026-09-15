import csv
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.core import Encoder, Gallery, normalize, preprocess, rank, read_rows
from backend.evaluate import calibrate, make_protocol, make_splits, metrics, ranked_queries


def test_preprocessing_bbox_rgb_and_normalization():
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:, 4:, 0] = 255
    tensor = preprocess(Image.fromarray(pixels), (4, 0, 4, 4))
    assert tensor.shape == (3, 208, 208)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[:, 0, 0], (np.array([1, 0, 0]) - [.485, .456, .406]) / [.229, .224, .225], rtol=1e-6)


@pytest.mark.parametrize("box", [(-1, 0, 2, 2), (0, 0, 0, 2), (0, 0, 11, 2), (0, 9, 2, 2), (0.5, 0, 2, 2)])
def test_invalid_bbox(box):
    with pytest.raises(ValueError):
        preprocess(Image.new("RGB", (10, 10)), box)


def test_ranking_is_cosine_and_ties_are_stable():
    vectors = normalize(np.array([[3, 0], [3, 3], [6, 0], [-1, 0]], dtype=np.float32))
    assert rank(vectors @ np.array([1, 0]), 4).tolist() == [0, 2, 1, 3]
    with pytest.raises(ValueError):
        normalize(np.zeros(512))


def test_hand_computed_metrics_and_refusal():
    ranked = [(np.array([.9, .8, .7]), np.array([True, False, True])),
              (np.array([.6, .5]), np.array([False, False]))]
    result = metrics(ranked, .75)
    assert result["mAP"] == pytest.approx((1 + 2 / 3) / 2)
    assert result["mINP"] == pytest.approx(2 / 3)
    assert result["Rank_1"] == result["Rank_5"] == 1
    assert result["candidate_F1"] == .5
    assert result["TNR"] == 1
    assert result["known_queries"] == result["unknown_queries"] == 1
    assert calibrate(ranked) == pytest.approx(.7)


def test_recall_counts_positives_outside_top10():
    result = metrics([(np.linspace(1, .1, 12), np.ones(12, dtype=bool))], -1)
    assert result["candidate_recall"] == pytest.approx(10 / 12)
    assert result["FN"] == 2


def test_camera_filter_removes_same_camera_even_other_vehicle():
    query = [{"image_id": "q", "vehicle_id": 1, "camera_id": 1}]
    gallery = [{"image_id": "a", "vehicle_id": 1, "camera_id": 1},
               {"image_id": "b", "vehicle_id": 2, "camera_id": 1},
               {"image_id": "c", "vehicle_id": 1, "camera_id": 2}]
    ranked = ranked_queries(query, gallery, {k: np.array([1., 0.]) for k in "qabc"})
    assert len(ranked[0][0]) == 1
    assert ranked[0][1].tolist() == [True]


def test_identity_and_frame_disjoint_splits():
    rows = [{"image_id": str(i), "vehicle_id": i} for i in range(30)]
    hashes = {str(i): str(i) for i in range(30)}
    hashes["1"] = hashes["2"] = hashes["3"]
    splits = make_splits(rows, hashes)
    assert splits == make_splits(rows, hashes)
    assert sorted(i for ids in splits.values() for i in ids) == list(range(30))
    assert any({1, 2, 3}.issubset(ids) for ids in splits.values())


def test_protocol_has_unknowns_and_cross_camera_positives():
    rows = [{"image_id": f"{i}-{c}", "vehicle_id": i, "camera_id": c} for i in range(10) for c in [1, 2]]
    queries, gallery = make_protocol(rows, list(range(10)))
    assert len(queries) == 10
    assert len({q["vehicle_id"] for q in queries} - {g["vehicle_id"] for g in gallery}) == 2
    for q in queries:
        assert all(q["image_id"] != g["image_id"] for g in gallery)
        assert all(q["camera_id"] != g["camera_id"] for g in gallery if q["vehicle_id"] == g["vehicle_id"])


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset")
    (root / "images").mkdir()
    rows = []
    for i in range(3):
        identifier = f"{i:032x}"
        pixels = np.random.default_rng(i).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / f"{identifier}.jpg")
        rows.append(dict(image_id=identifier, x=3, y=5, w=70, h=60))
    for name, values in [("test_gallery", rows[:2]), ("test_query", rows[2:])]:
        with open(root / f"{name}.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(values)
    return root


@pytest.fixture(scope="module")
def client(tiny_dataset, tmp_path_factory):
    app = create_app(tiny_dataset, tmp_path_factory.mktemp("artifacts"))
    with TestClient(app) as client:
        yield client


def test_api_actual_osnet_matches_upload_and_query(client, tiny_dataset):
    row = read_rows(tiny_dataset / "test_query.csv")[0]
    data = {k: row[k] for k in ("x", "y", "w", "h")}
    content = (tiny_dataset / "images" / f"{row['image_id']}.jpg").read_bytes()
    uploaded = client.post("/api/search", data=data, files={"image": ("q.jpg", content, "image/jpeg")})
    by_id = client.post("/api/search/query", json={"query_id": row["image_id"]})
    assert uploaded.status_code == by_id.status_code == 200
    assert uploaded.json()["results"] == by_id.json()["results"]
    embedded = client.post("/api/embedding", data=data, files={"image": ("q.jpg", content)})
    assert embedded.status_code == 200
    vector = np.array(embedded.json()["embedding"])
    assert vector.shape == (512,)
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    assert client.get(uploaded.json()["results"][0]["crop_url"]).status_code == 200


def test_api_rejects_bad_inputs(client):
    assert client.post("/api/search", data={"x": 0, "y": 0, "w": 1, "h": 1}, files={"image": ("fake.jpg", b"not an image")}).status_code == 422
    assert client.post("/api/search/query", json={"query_id": "../secret"}).status_code == 422
    assert client.post("/api/search/query", json={"query_id": "f" * 32}).status_code == 404
    assert client.post("/api/search/query", json={"query_id": "0" * 31 + "2", "top_k": 0}).status_code == 422
    assert client.post("/api/search/query", json={"query_id": "0" * 31 + "2", "threshold": 2}).status_code == 422
    assert client.get("/api/images/gallery/unknown").status_code == 404


def test_api_bbox_bounds_and_file_type(client, tiny_dataset):
    content = (tiny_dataset / "images" / ("0" * 32 + ".jpg")).read_bytes()
    assert client.post("/api/search", data={"x": 99, "y": 0, "w": 2, "h": 10}, files={"image": ("q.jpg", content)}).status_code == 422
    assert client.post("/api/search", data={"x": 0, "y": 0, "w": -1, "h": 10}, files={"image": ("q.jpg", content)}).status_code == 422


def test_api_refusal_and_no_fabricated_threshold(client):
    payload = {"query_id": "0" * 31 + "2", "mode": "candidates"}
    assert client.post("/api/search/query", json=payload).status_code == 409
    response = client.post("/api/search/query", json={**payload, "threshold": 1})
    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert response.json()["results"] == []


def test_frontend_and_openapi(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "SearchResponse" in schema["components"]["schemas"]


def test_gallery_cache_and_batch_parity(tiny_dataset, tmp_path):
    encoder = Encoder()
    gallery = Gallery(encoder, tiny_dataset, tmp_path / "gallery.db")
    reloaded = Gallery(encoder, tiny_dataset, tmp_path / "gallery.db")
    np.testing.assert_array_equal(gallery.vectors, reloaded.vectors)
    row = gallery.rows[0]
    with Image.open(tiny_dataset / "images" / f"{row['image_id']}.jpg") as image:
        single = encoder.encode(image, (3, 5, 70, 60))
    np.testing.assert_allclose(single, gallery.vectors[0], atol=2e-5)
    assert gallery.search(single, 2)[0]["image_id"] == row["image_id"]
    # Changing pixels invalidates the cache even if CSV and filename stay the same.
    path = tiny_dataset / "images" / f"{row['image_id']}.jpg"
    original = path.read_bytes()
    try:
        Image.new("RGB", (100, 80), "red").save(path)
        changed = Gallery(encoder, tiny_dataset, tmp_path / "gallery.db")
        assert changed.fingerprint != gallery.fingerprint
    finally:
        path.write_bytes(original)
