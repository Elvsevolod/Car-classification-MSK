"""Calibration tests use synthetic rectangles, never the real holdout."""
from types import SimpleNamespace

import numpy as np
import pytest

from training import mask_calibration as cal


@pytest.fixture
def sample(monkeypatch, tmp_path):
    plan = {"images": {i: {"split": split, "width": 100, "height": 80}
        for i, split in (("v1", "val"), ("v2", "val"), ("h1", "holdout"), ("h2", "holdout"))}}
    manual = {i: {"rectangles": [[10, 10, 30, 30]], "reviewed": True} for i in plan["images"]}
    signature = {"version": 1}
    monkeypatch.setattr(cal, "_context", lambda *args: (plan, manual, signature))
    calls = []

    def tensor(values):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.asarray(values)))

    class Detector:
        def predict(self, path, **kwargs):
            calls.append((str(path), kwargs))
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=tensor([[12.4, 12.4, 28.4, 28.4], [70, 50, 90, 70]]), conf=tensor([.8, .1])))]

    monkeypatch.setattr(cal, "runtime", lambda: lambda path: Detector())
    return plan, manual, signature, calls, tmp_path


def test_float_coordinates_expand_before_rounding_and_clip(sample):
    plan, manual, _, _, _ = sample
    raw = {i: {"xyxy": [[12.4, 12.4, 28.4, 28.4]], "confidences": [.8]} for i in ("v1", "v2")}
    result = cal.score_predictions(plan, manual, raw, .25, .1)
    assert result["predictions"]["v1"]["rectangles"] == [[10, 10, 30, 30]]
    assert result["coverage"]["reference_pixel_coverage"] == 1
    assert result["coverage"]["outside_reference_pixels"] == 0
    raw["v1"]["xyxy"] = [[0.2, 0.2, 99.8, 79.8]]
    assert cal.score_predictions(plan, manual, raw, .25, .2)["predictions"]["v1"]["rectangles"] == [[0, 0, 100, 80]]


def test_empty_detections_and_negative_images(sample):
    plan, manual, _, _, _ = sample
    raw = {i: {"xyxy": [], "confidences": []} for i in ("v1", "v2")}
    result = cal.score_predictions(plan, manual, raw, .25, 0)["coverage"]
    assert result["reference_pixel_coverage"] == 0 and result["regions_zero_coverage"] == 2
    assert result["negative_image_false_mask_rate"] is None
    manual["v2"]["rectangles"] = []
    raw["v2"] = {"xyxy": [[1, 1, 3, 3]], "confidences": [.6]}
    result = cal.score_predictions(plan, manual, raw, .25, 0)["coverage"]
    assert result["negative_images"] == 1 and result["negative_image_false_mask_rate"] == 1


@pytest.mark.parametrize("confidence,margin", [(0, 0), (1, 0), (.25, -.1), (.25, .26), (float('nan'), 0)])
def test_bad_policy_rejected(sample, confidence, margin):
    plan, manual, _, _, _ = sample
    with pytest.raises(ValueError, match="Invalid"):
        cal.score_predictions(plan, manual, {}, confidence, margin)


def test_score_rejects_other_split(sample):
    plan, manual, _, _, _ = sample
    with pytest.raises(ValueError, match="exactly"):
        cal.score_predictions(plan, manual, {"h1": {}}, .25, 0)


def test_selection_caps_global_and_single_crop_area():
    def row(regions, outside, worst):
        return {"confidence": .25, "margin": .1, "coverage": {
            "regions_covered_90pct": regions, "reference_pixel_coverage": .9,
            "outside_reference_fraction_of_crop": outside, "max_outside_reference_fraction_of_crop": worst}}
    valid = row(8, .02, .08)
    assert cal.select_candidate([row(10, .04, .08), row(10, .02, .2), valid]) == valid
    with pytest.raises(ValueError, match="No candidate"):
        cal.select_candidate([row(10, .04, .08)])


def test_calibration_infers_val_only_and_reuses_result(sample):
    plan, manual, signature, calls, tmp_path = sample
    report = cal.calibrate_masks("best.pt", "masks.json", tmp_path, "cpu")
    assert len(report["leaderboard"]) == 49 and len(calls) == 2
    assert all('/images/val/' in p and opts['conf'] == .05 for p, opts in calls)
    assert report["calibrated"]["coverage"]["regions_covered_90pct"] == 2
    assert report["calibrated"]["coverage"]["predicted_regions"] == 2
    assert not (tmp_path / 'raw_holdout.json').exists()
    assert cal.calibrate_masks("best.pt", "masks.json", tmp_path, "cpu") == report
    assert len(calls) == 2
    signature['version'] = 2
    with pytest.raises(ValueError, match="Different"):
        cal.calibrate_masks("best.pt", "masks.json", tmp_path, "cpu")


def test_holdout_requires_confirm_and_is_frozen_and_idempotent(sample):
    _, _, _, calls, tmp_path = sample
    with pytest.raises(ValueError, match="confirm"):
        cal.evaluate_frozen_holdout("best.pt", "masks.json", tmp_path)
    assert not calls
    validation = cal.calibrate_masks("best.pt", "masks.json", tmp_path, "cpu")
    result = cal.evaluate_frozen_holdout("best.pt", "masks.json", tmp_path, confirm=True, device="cpu")
    assert result['selected'] == validation['selected']
    assert set(result['calibrated']['predictions']) == {'h1', 'h2'}
    assert len(calls) == 4 and all('/images/test/' in p for p, _ in calls[2:])
    assert cal.evaluate_frozen_holdout("best.pt", "masks.json", tmp_path, confirm=True, device="cpu") == result
    assert len(calls) == 4
    changed = cal._load(tmp_path / 'calibration.json')
    changed['selected']['margin'] = .2
    cal._save(tmp_path / 'calibration.json', changed)
    with pytest.raises(ValueError, match="Different"):
        cal.evaluate_frozen_holdout("best.pt", "masks.json", tmp_path, confirm=True, device="cpu")


def test_incomplete_cache_resumes_only_missing_images(sample):
    plan, _, signature, calls, tmp_path = sample
    cal._save(tmp_path / 'raw_val.json', {'signature': {**signature, 'split': 'val'}, 'device': 'cpu',
        'images': {'v1': {'xyxy': [[10, 10, 30, 30]], 'confidences': [.8]}}})
    raw = cal._predict_split('best.pt', plan, 'val', tmp_path / 'raw_val.json', signature, tmp_path, 'cpu')
    assert len(calls) == 1 and calls[0][0].endswith('/v2.png')
    assert set(raw['images']) == {'v1', 'v2'}


def test_cache_tampering_and_split_mismatch_rejected(sample):
    plan, _, signature, _, tmp_path = sample
    path = tmp_path / 'raw.json'
    cal._save(path, {'signature': {**signature, 'split': 'val'}, 'device': 'cpu', 'images': {'h1': {}}})
    with pytest.raises(ValueError, match="different split"):
        cal._predict_split('best.pt', plan, 'val', path, signature, tmp_path, 'cpu')
    path.write_text(path.read_text().replace('cpu', 'mps'))
    with pytest.raises(ValueError, match="Changed"):
        cal._load(path)


def test_cannot_recalibrate_after_holdout_opened(sample):
    _, _, _, calls, tmp_path = sample
    (tmp_path / 'raw_holdout.json').write_text('{}')
    with pytest.raises(ValueError, match="Cannot recalibrate"):
        cal.calibrate_masks('best.pt', 'masks.json', tmp_path, 'cpu')
    assert not calls


def test_new_notebook_is_clean_and_cannot_train():
    import nbformat
    nb = nbformat.read(cal.EXPERIMENT / 'calibrate_yolo11_masks.ipynb', as_version=4)
    nbformat.validate(nb)
    source = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    assert 'RUN_CALIBRATION = True' in source and 'RUN_HOLDOUT = False' in source
    assert 'train_masks' not in source and '.train(' not in source
    for cell in nb.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, 'notebook', 'exec')
