"""v21 synthetic tests: no organizer data or old weights are changed."""
import numpy as np
import pytest
import torch

from backend.core import normalize, sha256
from training import model_search as search
from training.stage6 import write_json


@pytest.fixture
def protocol():
    q = [{"image_id": f"q{i}", "vehicle_id": i, "camera_id": 0} for i in [0, 1, 2, 3, 99, 100]]
    g = [{"image_id": f"g{i}", "vehicle_id": i//2, "camera_id": 1} for i in range(12)]
    rng = np.random.default_rng(12)
    gv = normalize(rng.normal(size=(12, 512)).astype(np.float32))
    qv = normalize(np.stack([gv[0]+.01, gv[2]+.01, gv[4]+.01, gv[6]+.01, *rng.normal(size=(2, 512))]).astype(np.float32))
    return q, g, qv, gv


def test_bounded_grid_and_specific_seed_can_win():
    seeds = (15, 16, 17)
    assert len(search.components(seeds)) == 13
    specs = search.screening_systems(seeds)
    assert len(specs) == 20 and len({s['name'] for s in specs}) == 20
    mixtures = search.ensemble_systems(seeds, 'R1_full1700_avg_15')
    assert len(mixtures) == 5
    assert all(sum(s['weights']) == pytest.approx(1) for s in mixtures)
    assert {len(s['members']) for s in mixtures} == {3, 4}
    reports = {'baseline': {'mean_map': .85}, 'new_seed15': {'mean_map': .86},
               'new_seed16': {'mean_map': .80}, 'new_seed17': {'mean_map': .79}}
    assert search.best(reports, list(reports)) == 'new_seed15'
    reports['new_seed15']['mean_map'] = .85
    assert search.best(reports, list(reports)) == 'baseline'


def test_incomplete_or_nan_results_never_select_a_winner():
    for bad in [None, {'mean_map': float('nan')}]:
        with pytest.raises(ValueError):
            search.best({'a': {'mean_map': .9}, 'b': bad}, ['a', 'b'])
    with pytest.raises(ValueError):
        search.best({}, [])


def test_weighted_concat_has_the_correct_cosine_geometry():
    rng = np.random.default_rng(3)
    values = [normalize(rng.normal(size=(9, 512)).astype(np.float32)) for _ in range(4)]
    weights = [.25, .25, .25, .25]
    combined = search.combine(values, weights)
    expected = sum(w * (v @ v.T) for w, v in zip(weights, values))
    np.testing.assert_allclose(combined @ combined.T, expected, rtol=0, atol=4e-7)
    equal = search.combine(values[:3], [1/3]*3)
    np.testing.assert_array_equal(equal, search.old.policy.combine_members(values[:3]))
    # Test geometry in float64, without float32 BLAS accumulation error over 2048 coordinates.
    mixed = search.combine(values, [.3, .3, .3, .1]).astype(np.float64)
    np.testing.assert_allclose(mixed @ mixed.T,
                               sum(w * (v.astype(np.float64) @ v.astype(np.float64).T)
                                   for w, v in zip([.3, .3, .3, .1], values)), atol=4e-7, rtol=0)


@pytest.mark.parametrize('weights', [[], [0., 1.], [-.1, 1.1], [.2, .2], [float('nan'), 1.]])
def test_invalid_mixture_weights(weights):
    with pytest.raises(ValueError):
        search.combine([np.ones((4, 512), np.float32)]*2, weights)


def test_invalid_or_fabricated_features_are_rejected():
    for array in [np.zeros((3, 512), np.float32), np.ones((3, 512), np.float32),
                  np.ones((3, 512), np.float64), np.full((3, 512), np.nan, np.float32)]:
        with pytest.raises(search.old.IntegrityError):
            search.validate_vectors(array, 3)
    with pytest.raises(ValueError):
        search.combine([np.zeros((3, 512), np.float32)], [1.])


@pytest.mark.parametrize('lam', search.LAMBDAS)
def test_rank_query_independence_permutation_deletion_and_batches(protocol, lam):
    _, _, qv, gv = protocol
    whole, _ = search.rank(qv, gv, lam)
    for batch in (1, 8, 16, 32):
        pieces = [search.rank(qv[i:i+batch], gv, lam)[0] for i in range(0, len(qv), batch)]
        np.testing.assert_array_equal(whole['order'], np.concatenate([p['order'] for p in pieces]))
    reverse, _ = search.rank(qv[::-1], gv, lam)
    np.testing.assert_array_equal(reverse['order'][::-1], whole['order'])
    for i in (0, 2, 5):
        one, _ = search.rank(qv[i:i+1], gv, lam)
        np.testing.assert_array_equal(one['order'][0], whole['order'][i])
    reference = search.old.policy.rank_vectors(qv, gv, 'less_graph' if lam == .75 else 'legacy')
    if lam in (.75, .50):
        np.testing.assert_array_equal(whole['order'], reference['order'])
    np.testing.assert_array_equal(whole['raw_order'], reference['raw_order'])
    np.testing.assert_array_equal(whole['confidence'], reference['confidence'])


def test_source_receipt_rejects_corruption_and_path_escape(tmp_path):
    p = tmp_path / 'old';p.mkdir()
    file = p / 'weights';file.write_bytes(b'original')
    receipt = p / 'complete.json'
    write_json(receipt, {'signature': 's', 'artifacts': {'weights': sha256(file)}})
    assert str(file) in search.verify_task_source(p, 's', receipt)
    with pytest.raises(search.old.IntegrityError, match='signature'):
        search.verify_task_source(p, 'other', receipt)
    file.write_bytes(b'changed')
    with pytest.raises(search.old.IntegrityError, match='artifact'):
        search.verify_task_source(p, 's', receipt)
    outside = tmp_path / 'outside';outside.write_bytes(b'valid')
    write_json(receipt, {'signature': 's', 'artifacts': {'../outside': sha256(outside)}})
    with pytest.raises(search.old.IntegrityError, match='artifact'):
        search.verify_task_source(p, 's', receipt)


def test_head_cannot_be_attached_to_ensemble_or_new_encoder(tmp_path):
    for members in [['R1_full1700_15'], ['R1_control_15', 'R1_control_16']]:
        with pytest.raises(search.old.IntegrityError, match='original single'):
            search.load_head_data({}, search.system('bad', members, head_weight=.1), [], np.empty((0, 512)))


def test_evaluate_saves_real_top10_and_raw_candidate_and_no_outer(protocol, tmp_path, monkeypatch):
    q, g, qv, gv = protocol
    rows = q + g
    context = {'rows': rows, 'manifest': {'draws': {'primary': {'episode': {
        'selection_eligible': True, 'query_ids': [r['image_id'] for r in q], 'gallery_ids': [r['image_id'] for r in g]}}}}}
    monkeypatch.setattr(search, 'reference_score', lambda *_: None)
    spec = search.system('single', ['x'])
    report = search.evaluate_system(context, spec, {'x': np.vstack([qv, gv])}, tmp_path, .75, True)
    assert report['mean_map'] == report['draws']['episode']['ranking']['mAP@10']
    diagnostic = report['draws']['episode']['candidate_diagnostic']
    assert not set(diagnostic['calibration_identities']) & set(diagnostic['evaluation_identities'])
    with np.load(tmp_path/'episode.npz', allow_pickle=False) as data:
        assert data['top10'].shape == (len(q), 10)
        assert all(len(set(r)) == 10 for r in data['top10'])
        np.testing.assert_array_equal(data['raw_top1'], np.array([r['image_id'] for r in g])[np.argmax(qv @ gv.T, axis=1)])
    monkeypatch.setattr(search, 'reference_score', lambda *_: report['mean_map'] - .01)
    with pytest.raises(search.old.IntegrityError, match='Historical score'):
        search.evaluate_system(context, spec, {'x': np.vstack([qv, gv])}, tmp_path)


def test_missing_average_uses_only_train_and_keeps_old_weights(tmp_path, monkeypatch):
    """Exercise the real average/BN branch without any training optimizer."""
    old_output, output = tmp_path/'old', tmp_path/'new'
    old_output.mkdir();output.mkdir()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2), torch.nn.Dropout(.9))
    summary = {'signature': 'saved', 'checkpoints': {}}
    for step, value in zip(search.AVERAGE_STEPS, (1., 2., 3.)):
        with torch.no_grad():
            for p in model.parameters(): p.fill_(value)
        path = old_output / f'{step}.pt'
        torch.save({'signature': 'saved', 'step': step, 'model': model.state_dict()}, path)
        summary['checkpoints'][str(step)] = {'path': path.name, 'sha256': sha256(path)}
    initial = {p: sha256(p) for p in old_output.iterdir()}
    context = {'output': output, 'quality_output': old_output, 'quality_manifest': {},
               'manifest': {'quality_source_signature': 'old_signature'}, 'protected': {}, 'device': torch.device('cpu'),
               'rows': [{'vehicle_id': i} for i in (0, 0, 1, 1, 9, 10)], 'dataset': tmp_path}
    monkeypatch.setattr(search, 'load_summary', lambda *_: summary)
    monkeypatch.setattr(search.clock, 'job_context', lambda c, _: c)
    monkeypatch.setattr(search.quality.training, 'load_job_model', lambda *_: (model, None))
    monkeypatch.setattr(search.quality.training, 'fold_training_ids', lambda *_: {0, 1})
    seen = []
    class Dataset:
        def __init__(self, rows, *args, **kw):
            assert kw['augment'] is False
            seen.extend(r['vehicle_id'] for r in rows);self.rows=rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return torch.tensor([float(i), 1.]), 0, str(i)
    monkeypatch.setattr(search, 'AblationDataset', Dataset)
    loaded, _, evidence = search.load_encoder(context, {'case': 'R1_full1700_avg', 'seed': 16}, output)
    assert seen == [0, 0, 1, 1] and evidence['bn_train_images'] == 4
    assert evidence['steps'] == [1400, 1600, 1700]
    assert all(torch.allclose(p, torch.full_like(p, 2.)) for p in loaded.parameters())
    assert not any(m.training for m in loaded.modules())
    assert all(sha256(p) == h for p, h in initial.items())
    assert (output/'derived.pt').is_file()


def test_complete_queue_resume_selection_failure_and_corruption(tmp_path, monkeypatch):
    seeds = (15, 16, 17)
    rows = [{'image_id': 'q', 'vehicle_id': 8}]
    context = {'output': tmp_path, 'signature': 'v21', 'seeds': seeds, 'protected': {},
               'manifest': {'source_sha256': {}, 'search_plan': {
                   'components': search.components(seeds), 'screening': search.screening_systems(seeds),
                   'lambdas': list(search.LAMBDAS), 'scope': 'development'}}}
    monkeypatch.setattr(search.old.review, 'check_inputs', lambda *a, **k: None)
    monkeypatch.setattr(search.old, 'fold_rows', lambda *a: rows)
    counters = {'features': 0, 'eval': 0}
    fail = {'enabled': True}
    def features(_context, name, spec, directory):
        counters['features'] += 1
        path = directory/'features.npz'
        np.savez_compressed(path, ids=np.array(['q']), vectors=np.full((1, 512), 1/np.sqrt(512), np.float32))
        return {'path': str(path), 'sha256': sha256(path), 'model': {'path': name, 'sha256': 'fake'}}
    def evaluate(_context, spec, _features, directory, lam=.75, candidate_diagnostics=False):
        counters['eval'] += 1
        if fail['enabled'] and spec['name'] == 'R1_full1700_16': raise RuntimeError('temporary failure')
        score = .82 if spec['name'] == 'R1_full1700_avg_15' else .8
        if spec['name'] == 'add_w10': score = .84 + (.01 if lam == .65 else 0)
        return {'system': spec, 'lambda': lam, 'mean_map': score,
                'draws': {'draw': {'per_query': {'q': {'vehicle_id': 8, 'ap': score}}}}}
    monkeypatch.setattr(search, 'feature_task', features)
    monkeypatch.setattr(search, 'evaluate_system', evaluate)
    with pytest.raises(ValueError, match='every planned'):
        search.run(context)
    assert not (tmp_path/'selected_candidate.json').exists()
    assert counters['features'] == 13
    fail['enabled'] = False
    result = search.run(context)
    assert result['status'] == 'complete'
    assert result['selection']['selected'] == 'add_w10_lambda65'
    assert result['selection']['system']['weights'] == [.3, .3, .3, .1]
    assert result['selection']['threshold'] is None and result['promoted'] is False
    assert counters['features'] == 13
    before = counters.copy()
    again = search.run(context)
    assert again['selection'] == result['selection'] and before == counters
    path = next(tmp_path.glob('tasks/features_*/features.npz'))
    path.write_bytes(b'corrupt')
    with pytest.raises(search.old.IntegrityError, match='Changed task artifact'):
        search.run(context)


def test_notebook_valid_after_execution_and_has_no_automatic_training():
    import nbformat
    path = search.VARIANT / 'search_best_model.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code': compile(cell.source, f'v21_cell_{i}', 'exec')
    text = '\n'.join(c.source for c in notebook.cells)
    assert "RUN_NAME = 'search_v1'" in text
    assert 'experiment.run(context)' in text and 'experiment.prepare' in text
    assert 'не среднее по seed' in text and 'Общего ограничения времени нет' in text
    assert 'fit_job(' not in text and 'allow_outer=True' not in text
