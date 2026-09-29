"""Load frozen calibration shared by the web API and train-free batch inference."""

import json
import math
from pathlib import Path

from .core import PREPROCESS, ROOT
from .rerank import ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA

DEFAULT_CALIBRATION = ROOT / "models/calibration.json"


class CalibrationError(ValueError):
    """The bundled report is missing, malformed, or incompatible with runtime."""


def load_calibration(encoder, path=DEFAULT_CALIBRATION):
    """Fail closed; never infer a threshold from test data or mutable artifacts."""
    path = Path(path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise CalibrationError(f"Cannot read calibration manifest {path}: {exc}") from exc

    def require(condition, field):
        if not condition:
            raise CalibrationError(f"Invalid or incompatible calibration {path}: {field}")

    require(isinstance(report, dict), "report must be an object")
    require(type(report.get("schema_version")) is int and report["schema_version"] == 1, "schema_version")
    for key, expected in (("model_sha256", encoder.model_sha256),
                          ("encoder_fingerprint", encoder.fingerprint),
                          ("preprocessing", PREPROCESS)):
        require(report.get(key) == expected, key)
    require(report.get("search") == {"ranking": "streaming k-reciprocal", "k1": ACTIVE_K1,
                                    "k2": ACTIVE_K2, "lambda": ACTIVE_LAMBDA,
                                    "refusal": "maximum raw cosine"}, "search configuration")
    threshold = report.get("threshold")
    require(type(threshold) in (int, float) and math.isfinite(threshold) and -1 <= threshold <= 1,
            "threshold must be finite raw cosine in [-1, 1]")
    require(report.get("confidence_definition") == "maximum raw cosine over gallery, not a probability",
            "confidence_definition")
    for split in ("calibration", "validation"):
        values = report.get(split)
        require(isinstance(values, dict), split)
        for metric in ("mAP_at_10", "candidate_F1", "TNR", "candidate_score"):
            value = values.get(metric)
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    f"{split}.{metric}")
    provenance = report.get("provenance")
    require(isinstance(provenance, dict) and bool(provenance.get("source"))
            and isinstance(provenance.get("source_sha256"), str)
            and len(provenance["source_sha256"]) == 64, "provenance")
    return report
