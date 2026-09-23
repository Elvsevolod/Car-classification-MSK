from dataclasses import asdict
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch
from PIL import Image

from training import masked_hpo as masked
from training.hpo import ExperimentConfig, hpo_identities, split_hpo_identities
from training.pipeline import VehicleDataset
from training.preprocessing import mask_crop


def test_masks_are_crop_local_exclusive_and_leave_source_unchanged():
    source = Image.new('RGB', (8, 6), 'white')
    result = mask_crop(source, [[2, 1, 5, 4]])
    pixels = np.asarray(result)
    assert np.all(pixels[1:4, 2:5] == 0)
    assert np.all(pixels[0] == 255) and np.all(pixels[:, 5:] == 255)
    assert np.all(np.asarray(source) == 255)
    for rect in ([1, 1, 9, 4], [1, 1, 1, 2], [.5, 1, 2, 3]):
        with pytest.raises(ValueError):
            mask_crop(source, [rect])


def test_vehicle_dataset_masks_both_consistency_views_and_inference(tmp_path):
    (tmp_path / 'images').mkdir()
    image_path = tmp_path / 'images/example.jpg'
    Image.new('RGB', (16, 16), 'white').save(image_path)
    original = image_path.read_bytes()
    row = {'image_id': 'example', 'x': 4, 'y': 4, 'w': 8, 'h': 8,
           'label': 3, 'mask_rectangles': [[0, 0, 4, 4]]}
    dataset = VehicleDataset([row], tmp_path, augment=True)
    dataset.clean_transform = lambda image: np.asarray(image).copy()
    dataset.robust_transform = lambda image: np.asarray(image).copy()
    clean, robust, label, _ = dataset[0]
    assert label == 3 and np.all(clean[:4, :4] == 0) and np.array_equal(clean, robust)
    assert np.all(clean[4:, 4:] == 255)
    masked_tensor = VehicleDataset([row], tmp_path)[0][0]
    expected_black = -torch.tensor([.485, .456, .406]) / torch.tensor([.229, .224, .225])
    assert torch.allclose(masked_tensor[:, 0, 0], expected_black)
    unmasked_row = {k: v for k, v in row.items() if k != 'mask_rectangles'}
    assert not torch.equal(masked_tensor, VehicleDataset([unmasked_row], tmp_path)[0][0])
    assert image_path.read_bytes() == original


def test_attach_masks_requires_exact_coverage_and_dimensions():
    rows = [{'image_id': 'a', 'w': 10, 'h': 8}, {'image_id': 'b', 'w': 10, 'h': 8}]
    cache = {'images': {i: {'width': 10, 'height': 8, 'rectangles': [[1, 1, 3, 4]]}
                        for i in ('a', 'b')}}
    result = masked.attach_masks(rows, cache)
    assert all('mask_rectangles' in r for r in result)
    assert all('mask_rectangles' not in r for r in rows)
    cache['images']['a']['rectangles'] = []
    assert masked.attach_masks(rows, cache)[0]['mask_rectangles'] == []
    cache['images']['b']['width'] = 11
    with pytest.raises(ValueError, match='dimensions'):
        masked.attach_masks(rows, cache)
    del cache['images']['b']
    with pytest.raises(ValueError, match='every'):
        masked.attach_masks(rows, cache)


def test_inner_split_excludes_transitive_detector_frames_and_is_reproducible():
    rows = [{'image_id': str(i), 'vehicle_id': i} for i in range(20)]
    hashes = {str(i): str(i) for i in range(20)}
    # Identity 2 bridges frame A (identity 1) and frame B (identity 3).
    rows += [{'image_id': 'bridge', 'vehicle_id': 2}]
    hashes.update({'1': 'A', '2': 'A', 'bridge': 'B', '3': 'B'})
    plan = {'images': {'1': {'vehicle_id': 1, 'frame_sha256': 'A'}}}
    split = masked.inner_split(rows, list(range(20)), hashes, plan)
    assert split == masked.inner_split(rows, list(range(20)), hashes, plan)
    assert {1, 2, 3}.issubset(split['train'])
    assert len(split['validation']) == 4
    assert set(split['train']).isdisjoint(split['validation'])
    assert sorted(split['train'] + split['validation']) == list(range(20))
    with pytest.raises(ValueError, match='outer train'):
        masked.inner_split(rows, [i for i in range(20) if i != 1], hashes, plan)


def test_hpo_uses_explicit_split_but_preserves_old_default():
    split = {'identities': {'train': list(range(20))}}
    assert hpo_identities(split) == split_hpo_identities(list(range(20)))
    split['hpo_identities'] = {'train': list(range(3, 20)), 'validation': [0, 1, 2]}
    assert hpo_identities(split) == (list(range(3, 20)), [0, 1, 2])
    split['hpo_identities']['validation'].append(3)
    with pytest.raises(ValueError, match='Invalid explicit'):
        hpo_identities(split)


def test_mask_cache_resumes_rejects_stale_signature_and_never_redetects(monkeypatch, tmp_path):
    rows = [{'image_id': i, 'w': 10, 'h': 10} for i in ('a', 'b', 'c')]
    signature = {'policy': {'confidence': .2, 'margin': .1}, 'inference': {}}
    calls = []
    fail = [True]
    monkeypatch.setattr(masked, 'runtime', lambda: lambda _: object())
    monkeypatch.setattr(masked, 'load_crop', lambda row, _: Image.new('RGB', (10, 10), 'white'))
    def detect(*args):
        calls.append(1)
        if len(calls) == 2 and fail[0]:
            raise RuntimeError('interrupted')
        return {'rectangles': [[1, 1, 4, 4]], 'confidences': [.8], 'reviewed': False}
    monkeypatch.setattr(masked, 'detect_mask', detect)
    path = tmp_path / 'masks.json'
    with pytest.raises(RuntimeError, match='interrupted'):
        masked.cache_masks(rows, signature, path, save_every=1)
    assert set(masked._load(path)['images']) == {'a'}
    fail[0] = False
    result = masked.cache_masks(rows, signature, path, save_every=1)
    assert set(result['images']) == {'a', 'b', 'c'} and len(calls) == 4
    masked.cache_masks(rows, signature, path)
    assert len(calls) == 4
    with pytest.raises(ValueError, match='Different weights'):
        masked.cache_masks(rows, {**signature, 'policy': {'confidence': .3}}, path)


def test_run_manifest_locks_cache_and_budget(tmp_path):
    cache = tmp_path / 'cache.json'
    cache.write_text('{}')
    results = tmp_path / 'run'
    signature = {'policy': {'confidence': .2, 'margin': .1}}
    budget = {'epochs': 8}
    first = masked.freeze_run(results, signature, cache, budget)
    assert masked.freeze_run(results, signature, cache, budget) == first
    with pytest.raises(ValueError, match='Different weights'):
        masked.freeze_run(results, signature, cache, {'epochs': 9})
    cache.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Different weights'):
        masked.freeze_run(results, signature, cache, budget)


def test_selected_training_selects_on_calibration_not_outer_validation(monkeypatch, tmp_path):
    observed = {}
    split = {'identities': {'train': [1, 2], 'calibration': [3], 'validation': [4]}}
    monkeypatch.setattr(masked, 'prepare_experiment', lambda rows, ids, config: ([], {1: 0, 2: 1}, 'sampler', 'loader'))
    monkeypatch.setattr(masked, 'initialize_experiment', lambda *args: ('model', 1))
    def fit(model, rows, validation_ids, *args):
        observed['ids'] = validation_ids
        return {'best_epoch': 1}
    monkeypatch.setattr(masked, 'fit_inner_candidate', fit)
    result = masked.fit_masked_selected([], split, torch.device('cpu'), ExperimentConfig(),
                                       tmp_path / 'weights', tmp_path / 'results')
    assert observed['ids'] == [3] and result['best_epoch'] == 1


def test_final_evaluation_keeps_masks_on_both_sides_and_uses_calibration(monkeypatch):
    rows = [{'image_id': f'{i}_{camera}', 'vehicle_id': i, 'camera_id': camera,
             'mask_rectangles': [[1, 1, 3, 4]]} for i in range(60) for camera in (0, 1)]
    split = {'identities': {'calibration': list(range(30)), 'validation': list(range(30, 60))}}
    seen = []
    def encode(model, selected, device):
        assert all(r['mask_rectangles'] for r in selected)
        seen.extend(r['image_id'] for r in selected)
        return np.stack([np.eye(60, 512, dtype=np.float32)[r['vehicle_id']] for r in selected])
    monkeypatch.setattr(masked, 'encode_experiment', encode)
    result = masked.evaluate_selected(None, rows, split, 'cpu')
    assert result['outer_validation_used_for_selection'] is False
    assert len(seen) == len(set(seen))
    for mode in ('raw', 'reranked'):
        assert result['scores'][mode]['validation']['mAP_at_10'] == 1.
        assert result['scores'][mode]['validation']['known_queries'] == 24
        assert result['scores'][mode]['validation']['unknown_queries'] == 6


def test_export_uses_masked_input_separate_report_and_reuses_result(monkeypatch, tmp_path):
    from training import pipeline

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                               torch.nn.Linear(3, 4))
        def embedding(self, images):
            return self.encoder(images)
        def inference_module(self):
            return self.encoder

    model = Tiny()
    weights, results = tmp_path / 'weights', tmp_path / 'results'
    weights.mkdir()
    results.mkdir()
    training_summary = results / 'summary.json'
    training_summary.write_text('{"best_epoch": 1}')
    torch.save({'model': model.state_dict(), 'config': asdict(ExperimentConfig()), 'epoch': 1},
               weights / 'best_map.pt')
    monkeypatch.setattr(masked, 'ROOT', tmp_path)
    monkeypatch.setattr(masked, 'initialize_experiment', lambda *args: (Tiny(), 0))
    seen = []
    def dataset(rows):
        assert rows[0]['mask_rectangles'] == [[0, 0, 8, 8]]
        seen.append('masked-input')
        return [(torch.zeros(3, 208, 208), 0, 'sample')]
    monkeypatch.setattr(pipeline, 'VehicleDataset', dataset)
    def evaluate(*args):
        seen.append('outer-evaluation')
        return {'scores': {'test': True}}
    monkeypatch.setattr(masked, 'evaluate_selected', evaluate)
    rows = [{'vehicle_id': 2, 'mask_rectangles': [[0, 0, 8, 8]]}]
    split = {'identities': {'train': [1], 'calibration': [2], 'validation': [3]}}
    signature = {'mask_policy': {'confidence': .2, 'margin': .1}}
    result = masked.export_selected(rows, split, weights, results, signature)
    assert result['checkpoint_epoch'] == 1 and result['parity']['max_absolute_difference'] < 1e-3
    assert (results / 'evaluation.json').exists() and (weights / 'osnet_masked_best_map.onnx').exists()
    assert training_summary.read_text() == '{"best_epoch": 1}'
    masked.export_selected(rows, split, weights, results, signature)
    assert seen == ['masked-input', 'outer-evaluation']


def test_notebook_is_valid_clean_and_compiles():
    path = Path(__file__).resolve().parents[1] / 'OSNet-AIN-x1.0/variant_08_masked_hpo/train_osnet_masked_hpo.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            assert cell.execution_count is None and cell.outputs == []
            compile(cell.source, f'notebook-cell-{i}', 'exec')
    all_code = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    assert 'TARGET_TRIALS = 16' in all_code and 'RUN_MASKS = True' in all_code
    assert "'evaluation.json'" in all_code
    assert "variant_02_hpo_bnneck_supcon" not in all_code
