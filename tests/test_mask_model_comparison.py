"""Isolated 2 x 3 audit: fixed inputs/thresholds, no training or MVP writes."""
import copy
import json

import numpy as np
import pytest
from PIL import Image

from training import mask_model_comparison as audit


def sample(tmp_path):
    (tmp_path / "images").mkdir()
    rows = []
    images = {}
    for i in ("q", "g"):
        path = tmp_path / "images" / f"{i}.jpg"
        Image.new("RGB", (30, 25), (150, 120, 90)).save(path)
        rows.append({"image_id": i, "vehicle_id": 2, "camera_id": int(i == "g"),
                     "x": 3, "y": 2, "w": 20, "h": 20})
        images[i] = {"frame_sha256": audit.sha256(path), "bbox": [3, 2, 20, 20], "width": 20, "height": 20}
    plan = {"coordinates": "crop_xyxy_pixels_exclusive", "query_ids": ["q"], "gallery_ids": ["g"], "images": images}
    plan["fingerprint"] = audit.digest(plan)
    split = {"identities": {"validation": [2]}, "frame_sha256": {i: v["frame_sha256"] for i, v in images.items()}}
    return rows, plan, split


def test_manual_protocol_keeps_order_and_rejects_changed_bbox_and_leakage(tmp_path):
    rows, plan, split = sample(tmp_path)
    query, gallery = audit.check_sample(rows[::-1], plan, split, tmp_path)
    assert query == rows[:1] and gallery == rows[1:]
    changed = copy.deepcopy(rows)
    changed[0]["x"] += 1
    with pytest.raises(ValueError, match="bbox changed"):
        audit.check_sample(changed, plan, split, tmp_path)
    with pytest.raises(ValueError, match="outer validation"):
        audit.check_sample(rows, plan, {**split, "identities": {"validation": [3]}}, tmp_path)
    changed = copy.deepcopy(plan)
    changed["gallery_ids"] = ["q"]
    changed["fingerprint"] = audit.digest({k: v for k, v in changed.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="disjoint"):
        audit.check_sample(rows, changed, split, tmp_path)
    Image.new("RGB", (30, 25)).save(tmp_path / "images/q.jpg")
    with pytest.raises(ValueError, match="Source frame"):
        audit.check_sample(rows, plan, split, tmp_path)


def test_cached_masks_need_full_coverage_valid_geometry_and_stay_unreviewed(tmp_path):
    _, plan, _ = sample(tmp_path)
    images = {i: {"width": 20, "height": 20, "reviewed": False,
                  "rectangles": [[1, 2, 5, 6]], "confidences": [.9]} for i in ("q", "g")}
    before = copy.deepcopy(images)
    assert audit.cached_predictions(plan, {"images": images}) == before
    assert images == before
    with pytest.raises(KeyError):
        audit.cached_predictions(plan, {"images": {"q": images["q"]}})
    images["q"]["rectangles"] = [[1, 2, 21, 6]]
    with pytest.raises(ValueError, match="outside crop"):
        audit.cached_predictions(plan, {"images": images})
    images["q"] = {**before["q"], "width": 21}
    with pytest.raises(ValueError, match="dimensions"):
        audit.cached_predictions(plan, {"images": images})


def test_six_extractions_mask_query_and_gallery_before_resize_without_source_edits(tmp_path):
    rows, _, _ = sample(tmp_path)
    protected = {p: p.read_bytes() for p in (tmp_path / "images").iterdir()}
    class Encoder:
        def __init__(self):
            self.calls = []
        def encode_batch(self, batch):
            self.calls.append(np.stack(batch))
            return np.tile(np.eye(1, 512, dtype=np.float32), (len(batch), 1))
    encoders = {name: Encoder() for name in ("mvp", "masked_trained")}
    annotations = {"original": None, **{condition: {i: {"rectangles": [box]} for i in ("q", "g")}
        for condition, box in (("manual", [5, 5, 10, 10]), ("yolo", [3, 3, 12, 12]))}}
    arrays = audit.extract_matrix(encoders, rows, annotations, tmp_path)
    for name, encoder in encoders.items():
        assert set(arrays[name]) == set(audit.CONDITIONS)
        assert len(encoder.calls) == 3
        assert all(c.shape == (2, 3, 208, 208) for c in encoder.calls)
        for changed in encoder.calls[1:]:
            assert all(not np.array_equal(a, b) for a, b in zip(encoder.calls[0], changed))
            assert np.array_equal(encoder.calls[0][:, :, 0, 0], changed[:, :, 0, 0])
    assert all(np.array_equal(a, b) for a, b in zip(encoders["mvp"].calls, encoders["masked_trained"].calls))
    assert all(p.read_bytes() == value for p, value in protected.items())


def test_evaluation_keeps_one_threshold_per_model_and_unknowns_out_of_map():
    query = [{"image_id": "q", "vehicle_id": 0, "camera_id": 0},
             {"image_id": "u", "vehicle_id": 99, "camera_id": 0}]
    gallery = [{"image_id": f"g{i}", "vehicle_id": i, "camera_id": 1} for i in range(22)]
    basis = np.eye(24, 512, dtype=np.float32)
    vectors = np.concatenate([basis[[0, 22]], basis[:22]])
    altered = vectors.copy()
    altered[0] = .6 * basis[0] + .8 * basis[23]
    matrix = {name: {"original": vectors, "manual": altered, "yolo": altered.copy()}
              for name in ("mvp", "masked_trained")}
    result = audit.evaluate_matrix(query, gallery, matrix, {"mvp": .7, "masked_trained": .5})
    for model in matrix:
        for condition in audit.CONDITIONS:
            for mode in audit.MODES:
                scores = result["scores"][model][condition][mode]
                assert scores["known_queries"] == scores["unknown_queries"] == 1
                assert scores["mAP_at_10"] == scores["TNR"] == 1
                assert result["per_query"][model][condition][mode]["u"]["AP_at_10"] is None
        assert result["per_query"][model]["original"]["raw"]["q"]["accepted"]
        for condition in ("manual", "yolo"):
            assert result["per_query"][model][condition]["raw"]["q"]["accepted"] == (model == "masked_trained")
        assert result["within_model"][model]["manual_minus_original"]["raw"]["mAP_at_10_delta"] == 0


def test_audit_encoder_rejects_wrong_weights_before_loading(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="checksum"):
        audit.AuditEncoder(path, "wrong")


def test_notebook_is_valid_and_runs_only_the_isolated_audit():
    import nbformat
    notebook = nbformat.read(audit.EXPERIMENT / "compare_models_masks.ipynb", as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "report = run(OUTPUT)" in source and "pip install" not in source
    assert "train_masks(" not in source and "calibrate(" not in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []
            compile(cell.source, "comparison_notebook", "exec")


def test_run_reuses_completed_results_and_rejects_changed_artifacts(monkeypatch, tmp_path):
    query = [{"image_id": "q", "vehicle_id": 0, "camera_id": 0, "w": 20, "h": 20}]
    gallery = [{"image_id": f"g{i}", "vehicle_id": i, "camera_id": 1, "w": 20, "h": 20} for i in range(22)]
    annotations = {"original": None, **{c: {r["image_id"]: {"rectangles": [[1, 1, 5, 5]], "confidences": [.9]}
        for r in query + gallery} for c in ("manual", "yolo")}}
    signature = {"thresholds": {"mvp": .5, "masked_trained": .6}, "frames": {},
                 "mask_policy": {"confidence": .2, "margin": .1}}
    monkeypatch.setattr(audit, "prepare", lambda *_: (query, gallery, {}, annotations, signature, {}))
    values = np.concatenate([np.eye(1, 512, dtype=np.float32), np.eye(22, 512, dtype=np.float32)])
    monkeypatch.setattr(audit, "extract_matrix", lambda *_: {m: {c: values for c in audit.CONDITIONS}
                        for m in ("mvp", "masked_trained")})
    # This fixture has no unknown queries; the report renderer is tested with a separate formatter stub.
    monkeypatch.setattr(audit, "markdown_report", lambda report: "# synthetic audit\n")
    output = tmp_path / "audit"
    report = audit.run(output, tmp_path)
    assert report["protected_unchanged"]
    with np.load(output / "embeddings.npz", allow_pickle=False) as archive:
        assert archive["ids"].tolist() == [r["image_id"] for r in query + gallery]
        assert len(archive.files) == 8
    def forbidden(*_):
        raise AssertionError("Repeated run must not perform inference")
    monkeypatch.setattr(audit, "extract_matrix", forbidden)
    assert audit.run(output, tmp_path) == report
    (output / "RESULTS.md").write_text("changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        audit.run(output, tmp_path)
    saved = json.loads((output / "comparison.json").read_text())
    assert saved["signature"] == signature
