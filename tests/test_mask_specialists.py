import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("cv2")
from training import mask_specialists as specialists


def test_mosaic_and_egoblur_input_use_bgr_but_different_scale_and_shape():
    image = Image.new("RGB", (720, 360), (255, 100, 0))
    mosaic = specialists.mosaic_input(image)
    assert mosaic.shape == (1, 3, 360, 720) and mosaic.dtype == torch.float32
    torch.testing.assert_close(mosaic[0, :, 0, 0], torch.tensor([0, 100 / 255, 1]))
    ego = specialists.egoblur_input(image)
    assert ego.shape == (3, 360, 720) and ego.dtype == torch.uint8
    assert ego[:, 0, 0].tolist() == [0, 100, 255]


def test_mosaic_threshold_quantization_all_components_and_exclusive_bounds():
    probability = np.zeros((8, 12), np.float32)
    probability[1:3, 2:5] = .9
    probability[5:7, 8:10] = .8
    probability[0, 0] = 64 / 255  # author's strict uint8 > 64 threshold
    result = specialists.mosaic_rectangles(probability, (24, 16))
    assert result["reviewed"] is False
    assert result["rectangles"] == [[4, 2, 10, 6], [16, 10, 20, 14]]
    assert len(result["confidences"]) == 2


def test_mosaic_expansion_is_separate_and_empty_detections_remain_empty():
    probability = np.zeros((100, 100), np.float32)
    probability[40:60, 40:60] = 1
    raw = specialists.mosaic_rectangles(probability, (100, 100))
    expanded = specialists.mosaic_rectangles(probability, (100, 100), True)
    assert raw["rectangles"] == [[40, 40, 60, 60]]
    box = expanded["rectangles"][0]
    assert box[0] < 40 and box[1] < 40 and box[2] > 60 and box[3] > 60
    assert specialists.mosaic_rectangles(probability * 0, (100, 100), True)["rectangles"] == []
    with pytest.raises(ValueError, match="probability"):
        specialists.mosaic_rectangles(probability * np.nan, (100, 100))


def test_egoblur_output_filter_nms_and_clipping():
    boxes = torch.tensor([[1., 2., 20., 21.], [2, 2, 20, 21], [50, 50, 60, 60], [-1, 1, 5, 10]])
    scores = torch.tensor([.8, .7, .25, .5])
    output = (boxes, torch.zeros(4), scores, torch.tensor([100, 100]))
    result = specialists.egoblur_rectangles(output, (100, 100))
    assert result["rectangles"] == [[1, 2, 20, 21], [0, 1, 5, 10]]
    assert result["reviewed"] is False


def test_combined_predictions_preserve_classes_and_reference_status():
    left = {"a": {"rectangles": [[0, 0, 5, 5]], "confidences": [.6], "reviewed": False}}
    right = {"a": {"rectangles": [[10, 10, 20, 20]], "confidences": [.8], "reviewed": False}}
    result = specialists.combine_predictions(left, right)
    assert result["a"]["rectangles"] == [[10, 10, 20, 20], [0, 0, 5, 5]]
    assert result["a"]["reviewed"] is False
    assert left["a"]["rectangles"] == [[0, 0, 5, 5]]
    with pytest.raises(ValueError, match="crop sets"):
        specialists.combine_predictions(left, {})
