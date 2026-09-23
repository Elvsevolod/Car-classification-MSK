"""Synthetic paired-audit tests; never instantiate the MVP gallery database."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from training import mask_reid_ablation as audit


def tensor(values):
    return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.asarray(values, dtype=float)))


@pytest.mark.parametrize('collision', ['image', 'identity', 'frame'])
def test_detector_overlap_is_rejected(collision):
    selected = [{'image_id': 'q', 'vehicle_id': 1}]
    key = 'q' if collision == 'image' else 'train'
    plan = {'images': {key: {'vehicle_id': 1 if collision == 'identity' else 2,
                             'frame_sha256': 'q-frame' if collision == 'frame' else 'other-frame'}}}
    with pytest.raises(ValueError, match='overlaps'):
        audit.check_detector_separation(selected, {'q': 'q-frame'}, plan)


def test_disjoint_detector_data_is_accepted():
    audit.check_detector_separation([{'image_id': 'q', 'vehicle_id': 1}], {'q': 'q-frame'},
        {'images': {'train': {'vehicle_id': 2, 'frame_sha256': 'other-frame'}}})


def test_detector_uses_frozen_policy_and_expands_float_boxes():
    calls = []
    class Detector:
        def predict(self, image, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=tensor([[12.4, 12.4, 28.4, 28.4]]), conf=tensor([.8])))]
    image = Image.new('RGB', (40, 40), (123, 45, 67))
    before = image.tobytes()
    result = audit.detect_mask(Detector(), image, {'confidence': .2, 'margin': .1},
                              {'conf': .05, 'iou': .45, 'imgsz': 640, 'save': False}, 'cpu')
    assert result['rectangles'] == [[10, 10, 30, 30]]
    assert calls[0]['conf'] == .2 and calls[0]['iou'] == .45
    assert image.tobytes() == before


def test_cache_checks_order_signature_normalization_and_labels(tmp_path):
    path = tmp_path / 'cache.npz'
    vectors = {k: np.eye(2, 512, dtype=np.float32) for k in ('original', 'masked')}
    metadata = {'fingerprint': 'version1', 'predictions': {'q': {}, 'g': {}}}
    audit.save_cache(path, ['q', 'g'], vectors, metadata)
    ids, arrays, loaded = audit.load_cache(path, 'version1', ['q', 'g', 'g2'])
    assert ids == ['q', 'g'] and loaded == metadata
    assert np.array_equal(arrays['original'], vectors['original'])
    with pytest.raises(ValueError, match='different protocol'):
        audit.load_cache(path, 'version2', ['q', 'g'])
    with pytest.raises(ValueError, match='different protocol'):
        audit.load_cache(path, 'version1', ['g', 'q'])
    vectors['masked'][0] = 0
    audit.save_cache(path, ['q', 'g'], vectors, metadata)
    with pytest.raises(ValueError, match='Invalid cached'):
        audit.load_cache(path, 'version1', ['q', 'g'])


def test_paired_extraction_masks_query_and_gallery_and_preserves_files(monkeypatch, tmp_path):
    (tmp_path / 'images').mkdir()
    rows = []
    for image_id in ('q', 'g'):
        Image.new('RGB', (30, 30), (123, 45, 67)).save(tmp_path / 'images' / f'{image_id}.jpg')
        rows.append({'image_id': image_id, 'x': 5, 'y': 5, 'w': 20, 'h': 20})
    protected = {p: p.read_bytes() for p in (tmp_path / 'images').iterdir()}
    detector_calls, encoder_calls = [], []
    class Detector:
        def predict(self, image, **kwargs):
            assert image.size == (20, 20)
            detector_calls.append(image.tobytes())
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor([[5, 5, 15, 15]]), conf=tensor([.9])))]
    class Encoder:
        def encode_batch(self, batch):
            encoder_calls.append(np.stack(batch))
            return np.tile(np.eye(1, 512, dtype=np.float32), (len(batch), 1))
    monkeypatch.setattr(audit, 'runtime', lambda: lambda path: Detector())
    output = tmp_path / 'result'
    output.mkdir()
    signature = {'mask_policy': {'confidence': .2, 'margin': .1}, 'inference': {'save': False}}
    vectors, meta = audit.paired_embeddings(Encoder(), rows, signature, output, 'cpu', tmp_path)
    assert len(detector_calls) == 2 and set(meta['predictions']) == {'q', 'g'}
    assert encoder_calls[0].shape == encoder_calls[1].shape == (2, 3, 208, 208)
    assert all(not np.array_equal(a, b) for a, b in zip(*encoder_calls))
    assert np.array_equal(encoder_calls[0][:, :, 0, 0], encoder_calls[1][:, :, 0, 0])
    assert all(p.read_bytes() == data for p, data in protected.items())
    audit.paired_embeddings(Encoder(), rows, signature, output, 'cpu', tmp_path)
    assert len(detector_calls) == 2 and len(encoder_calls) == 2


def test_official_metrics_keep_unknowns_out_of_map_and_threshold_fixed():
    queries = [{'image_id': 'q', 'vehicle_id': 0, 'camera_id': 0},
               {'image_id': 'unknown', 'vehicle_id': 99, 'camera_id': 0}]
    gallery = [{'image_id': f'g{i}', 'vehicle_id': i, 'camera_id': 1} for i in range(22)]
    basis = np.eye(23, 512, dtype=np.float32)
    vectors = np.concatenate([basis[[0, 22]], basis[:22]])
    scores, details = audit.evaluate_embeddings(queries, gallery, vectors, .5)
    for mode in ('raw', 'reranked'):
        assert scores[mode]['mAP_at_10'] == 1 and scores[mode]['known_queries'] == 1
        assert scores[mode]['unknown_queries'] == 1 and scores[mode]['TNR'] == 1
        assert details[mode]['q']['accepted'] and not details[mode]['unknown']['accepted']
        assert details[mode]['unknown']['AP_at_10'] is None


def test_paired_deltas_are_reproducible_and_exclude_unknowns():
    def item(ap, correct):
        return {'known': ap is not None, 'AP_at_10': ap, 'top1_correct': correct,
                'top10': ['g'], 'accepted': correct}
    before = {'q1': item(.5, False), 'q2': item(.8, True), 'u': item(None, False)}
    after = {'q1': item(1., True), 'q2': item(.6, False), 'u': item(None, False)}
    result = audit.paired_deltas(before, after)
    assert result == audit.paired_deltas(before, after)
    assert result['mAP_at_10_delta'] == pytest.approx(.15)
    assert result['AP_improved'] == result['AP_worsened'] == 1
    assert result['top1_improved'] == ['q1'] and result['top1_worsened'] == ['q2']
    with pytest.raises(ValueError, match='same queries'):
        audit.paired_deltas(before, {'q1': after['q1']})
