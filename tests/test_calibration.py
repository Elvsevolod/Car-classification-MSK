import json

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.calibration import CalibrationError, DEFAULT_CALIBRATION, load_calibration
from backend.core import Encoder


@pytest.fixture(scope="module")
def encoder():
    return Encoder()


def test_bundled_calibration_matches_actual_encoder(encoder):
    report = load_calibration(encoder)
    assert report["threshold"] == 0.5948754549026489
    assert report["validation"]["mAP_at_10"] == 0.8146886982413298
    assert report["model_sha256"] == encoder.model_sha256
    assert report["encoder_fingerprint"] == encoder.fingerprint


@pytest.mark.parametrize("field,value", [
    ("model_sha256", "wrong"),
    ("encoder_fingerprint", "wrong"),
    ("preprocessing", "changed-resize"),
    ("search", {"ranking": "cosine"}),
    ("threshold", None),
    ("threshold", float("nan")),
    ("threshold", float("inf")),
    ("threshold", 1.01),
    ("threshold", -1.01),
    ("threshold", True),
    ("schema_version", 2),
    ("schema_version", True),
    ("confidence_definition", "probability"),
    ("calibration", {}),
    ("validation", None),
    ("provenance", None),
])
def test_invalid_or_incompatible_manifest_fails_closed(encoder, tmp_path, field, value):
    report = json.loads(DEFAULT_CALIBRATION.read_text())
    report[field] = value
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report))
    with pytest.raises(CalibrationError, match="Invalid or incompatible calibration"):
        load_calibration(encoder, path)


@pytest.mark.parametrize("content", ["{", "[]", "null", '"not an object"'])
def test_malformed_manifest_is_actionable(encoder, tmp_path, content):
    path = tmp_path / "calibration.json"
    path.write_text(content)
    with pytest.raises(CalibrationError, match="calibration"):
        load_calibration(encoder, path)


def test_missing_manifest_fails_before_database_or_dataset_access(tmp_path):
    path = tmp_path / "missing-calibration.json"
    app = create_app(dataset=tmp_path / "no-dataset", calibration_path=path, profile="MVP_legacy")
    with pytest.raises(CalibrationError, match="Cannot read calibration manifest"):
        with TestClient(app):
            pytest.fail("Startup must fail without the approved calibration")


def test_reranker_parameter_drift_is_rejected(encoder, monkeypatch):
    monkeypatch.setattr("backend.calibration.ACTIVE_K1", 21)
    with pytest.raises(CalibrationError, match="search configuration"):
        load_calibration(encoder)
