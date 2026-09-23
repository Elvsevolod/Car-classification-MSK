"""Small, synthetic tests: no model downloads, images, or training required."""
import numpy as np
import pytest
from PIL import Image

from training import mask_detection as detection
from training.audit import validate_annotations


@pytest.mark.parametrize("size", [(321, 100), (100, 321), (640, 640)])
def test_letterbox_rgb_padding_and_inverse_coordinates(size):
    pytest.importorskip("cv2")
    image = Image.new("RGB", size, (255, 0, 0))
    tensor, gain, fractional, integer = detection.letterbox(image)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor[0, :, 320, 320], [1, 0, 0])
    if size[0] != size[1]:
        np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255)
        assert any(abs(a - b) == .5 for a, b in zip(fractional, integer))
    for pad in (fractional, integer):
        original = np.array([[10, 10, 90, 80]], dtype=float)
        boxes, scores = detection.restore_boxes(original * gain + np.tile(pad, 2), [.75], gain, pad, size)
        np.testing.assert_allclose(boxes, original, atol=1)
        assert scores == [.75]


def test_yolov9_end2end_column_order_and_confidence():
    output = np.array([[0, 10, 20, 30, 40, 0, .9], [0, 1, 2, 3, 4, 0, .1]])
    boxes, scores = detection.decode(output, "yolov9_end2end")
    assert boxes.tolist() == [[10, 20, 30, 40]]
    assert scores.tolist() == [.9]
    assert detection.decode(np.empty((0, 7)), "yolov9_end2end")[0].shape == (0, 4)
    with pytest.raises(ValueError, match="N x 7"):
        detection.decode(np.zeros((2, 6)), "yolov9_end2end")
    output[0, 5] = 1
    with pytest.raises(ValueError, match="batch/class"):
        detection.decode(output, "yolov9_end2end")


def test_yolo11_xywh_nms_and_empty_output():
    raw = np.array([[[20, 30, 20, 20, .9], [21, 30, 20, 20, .8],
                     [100, 100, 10, 10, .7], [200, 200, 10, 10, .1]]]).transpose(0, 2, 1)
    boxes, scores = detection.decode(raw, "yolo11_raw")
    assert boxes.tolist() == [[10, 20, 30, 40], [95, 95, 105, 105]]
    assert scores.tolist() == [.9, .7]
    assert detection.decode(raw * 0, "yolo11_raw")[0].shape == (0, 4)
    with pytest.raises(ValueError, match="1 x 5 x N"):
        detection.decode(np.zeros((1, 6, 10)), "yolo11_raw")
    with pytest.raises(ValueError, match="Non-finite"):
        detection.decode(raw * np.nan, "yolo11_raw")


def test_restore_clips_boxes_and_removes_degenerate():
    boxes, scores = detection.restore_boxes([[-1, 2.8, 50.2, 11], [100, 1, 120, 5]],
                                            [.9, .7], 1, (0, 0), (20, 10))
    assert boxes == [[0, 2, 20, 10]]
    assert scores == [.9]


def test_reference_matching_is_one_to_one_and_pixel_union_is_not_double_counted():
    result = detection.compare_regions((20, 10), [[0, 0, 10, 10]],
                                       [[0, 0, 10, 10], [0, 0, 10, 10]], [.9, .8])
    assert result["matched_regions"] == 1
    assert result["predicted_regions"] == 2
    assert result["intersection_pixels"] == result["predicted_pixels"] == 100
    assert result["regions_covered_90pct"] == 1
    totals = detection.summarize({"a": result})
    assert totals["agreement_precision_at_05"] == .5
    assert totals["agreement_recall_at_05"] == 1
    assert totals["unmatched_predictions"] == 1


def test_coverage_partial_excess_and_no_prediction():
    result = detection.compare_regions((20, 10), [[0, 0, 10, 10]], [[5, 0, 15, 10]], [.9])
    assert result["matched_regions"] == 0  # IoU is 1/3, even though 50% pixels are covered.
    assert result["reference_pixels"] == result["predicted_pixels"] == 100
    assert result["intersection_pixels"] == result["outside_reference_pixels"] == 50
    assert result["region_coverages"] == [.5]
    missed = detection.compare_regions((20, 10), [[0, 0, 10, 10]], [], [])
    assert missed["intersection_pixels"] == missed["predicted_pixels"] == 0
    summary = detection.summarize({"partial": result, "missed": missed})
    assert summary["reference_pixel_coverage"] == .25
    assert summary["outside_reference_fraction_of_crop"] == .125
    assert summary["crops_with_predictions"] == 1


def test_automatic_predictions_must_not_pass_as_manual_reference():
    plan = {"fingerprint": "a", "images": {"q": {"width": 10, "height": 10}}}
    auto = {"fingerprint": "a", "images": {"q": {
        "reviewed": False, "rectangles": [[0, 0, 5, 5]], "confidences": [.9]}}}
    with pytest.raises(ValueError, match="Unreviewed"):
        validate_annotations(plan, auto)
