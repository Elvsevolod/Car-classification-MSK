"""Ranking-only search keeps candidate/threshold frozen and selects before validation."""
from copy import deepcopy

import nbformat
import numpy as np
import pytest

from backend.core import normalize
from training import map_search as search
from tests.test_dual_role_experiment import context


def test_grid_is_unique_fixed_and_baseline_first():
    grid = search.systems()
    assert len(grid) == len({s['name'] for s in grid}) == 84
    assert grid[0] == {"name": "r1w00_k20_q3_l50", "k1": 20, "k2": 3, "lambda": .5, "r1_weight": 0.}
    assert sum(s['r1_weight'] == 0 for s in grid) == 72
    assert all(set(s) == set(grid[0]) for s in grid)


def test_all_ranking_variants_keep_exact_candidate_and_confidence():
    rng = np.random.default_rng(10)
    values = search.dual.pack(*[normalize(rng.normal(size=(25, d)).astype(np.float32)) for d in (512, 1536)])
    reference = search.dual.rank(values[:5], values[5:])
    for spec in search.systems():
        actual = search.rank(values, 5, spec)
        np.testing.assert_array_equal(actual['raw_order'], reference['raw_order'])
        np.testing.assert_array_equal(actual['confidence'], reference['confidence'])
        if spec == search.systems()[0]:
            np.testing.assert_array_equal(actual['order'], reference['order'])


@pytest.mark.parametrize('spec', [search.systems()[0], search.systems()[20], search.systems()[-1]])
def test_query_independence_for_ranking(spec):
    rng = np.random.default_rng(45)
    values = search.dual.pack(*[normalize(rng.normal(size=(48, d)).astype(np.float32)) for d in (512, 1536)])
    q, g = values[:25], values[25:]
    expected = search.rank(values, len(q), spec)
    for selection in (np.arange(len(q))[::-1], np.array([4])):
        actual = search.rank(np.concatenate([q[selection], g]), len(selection), spec)
        np.testing.assert_array_equal(actual['order'], expected['order'][selection])
        np.testing.assert_array_equal(actual['raw_order'], expected['raw_order'][selection])
    for size in (1, 8, 16, 32):
        results = [search.rank(np.concatenate([q[i:i+size], g]), len(q[i:i+size]), spec)
                   for i in range(0, len(q), size)]
        np.testing.assert_array_equal(np.concatenate([r['order'] for r in results]), expected['order'])


@pytest.fixture
def search_context(context, monkeypatch):
    search.previous.run(context, allow_outer=True)
    specs = [search.systems()[0], search.systems()[-1]]
    return {**context, 'output': context['output'].parent/'v25', 'signature': 'v25', 'v24_output': context['output'],
            'manifest': {**context['manifest'], 'map_search_plan': {'source_signature': context['signature'], 'systems': specs}}}


def test_run_selects_only_after_calibration_and_resumes(search_context, monkeypatch):
    def denied(*a, **kw):
        raise AssertionError('No new calibration or encoder inference in v25')
    monkeypatch.setattr(search.dual.policy, 'calibrate_policy', denied)
    monkeypatch.setattr(search.dual, 'DualRoleEncoder', denied)
    with pytest.raises(ValueError, match='allow_outer'):
        search.run(search_context)
    with pytest.raises(FileNotFoundError):
        search.read_selection(search_context)
    result = search.run(search_context, allow_outer=True)
    assert result['status'] == 'complete' and result['protected_unchanged']
    assert result['threshold_fit'] is False and result['encoder_forwards'] == 0 and not result['promoted']
    assert result['selection']['selection_split'] == 'calibration'
    assert len(result['evaluations']) <= 2 and all(r['candidate_unchanged'] for r in result['evaluations'].values())
    resumed = search.run(search_context, allow_outer=True)
    assert result['evaluations'] == resumed['evaluations']
    assert all(event['status'] == 'cached' for event in resumed['events'])
    reports = deepcopy(result['calibration'])
    for r in reports.values():
        r['ranking']['mAP@10'] = .8
    assert search.selection(search_context, reports)['selected'] == search_context['manifest']['map_search_plan']['systems'][0]
    reports[next(iter(reports))]['split'] = 'validation'
    with pytest.raises(search.old.IntegrityError):
        search.selection(search_context, reports)


def test_notebook_has_no_training_or_time_limit():
    notebook = nbformat.read(search.VARIANT/'search_map.ipynb', as_version=4)
    nbformat.validate(notebook)
    code = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    compile(code, 'notebook', 'exec')
    assert 'experiment.run(context' in code and 'ALLOW_OUTER_EVALUATION = True' in code
    assert not any(word in code for word in ('calibrate_policy', 'WALL_HOURS', 'optimizer.step'))
