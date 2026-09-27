"""v26 extends only ranking and keeps the concrete v25 baseline reproducible."""
from copy import deepcopy

import nbformat
import numpy as np
import pytest

from backend.core import normalize
from training import fusion_extension as experiment
from tests.test_map_search import search_context
from tests.test_dual_role_experiment import context


def test_frozen_grid_covers_boundary_with_v25_first():
    grid = experiment.systems()
    assert len(grid) == len({s['name'] for s in grid}) == 84
    assert grid[0] == experiment.BASELINE
    assert {s['r1_weight'] for s in grid} == {.4, .5, .6, .7, .8, .9, 1.}
    assert all(s['k2'] == 3 for s in grid)


def test_every_variant_preserves_candidates():
    rng = np.random.default_rng(26)
    values = np.concatenate([normalize(rng.normal(size=(30,d)).astype(np.float32)) for d in (512,1536)], axis=1)
    base = experiment.previous.rank(values,5,experiment.BASELINE)
    for spec in experiment.systems():
        r = experiment.previous.rank(values,5,spec)
        np.testing.assert_array_equal(r['raw_order'],base['raw_order'])
        np.testing.assert_array_equal(r['confidence'],base['confidence'])


def test_reference_export_rank_matches_research():
    from training.map_inference import rank
    rng = np.random.default_rng(325)
    values = np.concatenate([normalize(rng.normal(size=(36,d)).astype(np.float32)) for d in (512,1536)],axis=1)
    ref = experiment.previous.rank(values, 6, experiment.BASELINE)
    actual = rank(values[:6], values[6:])
    for key in ('order','raw_order','confidence'):
        np.testing.assert_array_equal(actual[key],ref[key])


@pytest.mark.parametrize('weight', [.4, .7, 1.])
def test_query_independence_and_batching(weight):
    spec = {**experiment.BASELINE, 'r1_weight':weight}
    rng = np.random.default_rng(75)
    values = np.concatenate([normalize(rng.normal(size=(60,d)).astype(np.float32)) for d in (512,1536)],axis=1)
    q,g = values[:33],values[33:]
    ref = experiment.previous.rank(values,len(q),spec)
    for select in (np.arange(len(q))[::-1],np.array([4])):
        actual = experiment.previous.rank(np.concatenate([q[select],g]),len(select),spec)
        np.testing.assert_array_equal(actual['order'],ref['order'][select])
        np.testing.assert_array_equal(actual['raw_order'],ref['raw_order'][select])
    for size in (1,8,16,32):
        parts = [experiment.previous.rank(np.concatenate([q[i:i+size],g]),len(q[i:i+size]),spec)
                 for i in range(0,len(q),size)]
        np.testing.assert_array_equal(np.concatenate([x['order'] for x in parts]),ref['order'])


@pytest.fixture
def extension_context(search_context):
    # Produce a genuine synthetic v25 baseline receipt through the unchanged controller.
    search_context['manifest']['map_search_plan']['systems'] = [dict(experiment.BASELINE)]
    experiment.previous.run(search_context,allow_outer=True)
    return {**search_context,'output':search_context['output'].parent/'v26','signature':'v26',
            'v25_output':search_context['output'], 'manifest':{**search_context['manifest'],
            'fusion_extension_plan':{'source_signature':search_context['signature']},
            'map_search_plan':{**search_context['manifest']['map_search_plan'],
                              'systems':[dict(experiment.BASELINE),experiment.systems()[-1]]}}}


def test_frozen_selection_baseline_resume_and_no_training(extension_context,monkeypatch):
    def forbidden(*a,**kw):
        raise AssertionError('No model inference or threshold fitting')
    monkeypatch.setattr(experiment.previous.dual,'DualRoleEncoder',forbidden)
    monkeypatch.setattr(experiment.previous.dual.policy,'calibrate_policy',forbidden)
    with pytest.raises(ValueError,match='allow_outer'):
        experiment.run(extension_context)
    with pytest.raises(FileNotFoundError):
        experiment.previous.read_selection(extension_context)
    r = experiment.run(extension_context,allow_outer=True)
    assert r['status']=='complete' and r['protected_unchanged'] and not r['promoted']
    assert r['encoder_forwards']==r['optimizer_updates']==0 and not r['threshold_fit']
    assert list(r['evaluations'])[0]==experiment.BASELINE['name']
    assert len(r['evaluations'])<=2 and r['selected_vs_v25']['C_delta']==0
    again=experiment.run(extension_context,allow_outer=True)
    assert r['evaluations']==again['evaluations']
    assert all(x['status']=='cached' for x in again['events'])
    reports=deepcopy(r['calibration'])
    for x in reports.values(): x['ranking']['mAP@10']=.8
    assert experiment.previous.selection(extension_context,reports)['selected']==experiment.BASELINE
    reports[experiment.BASELINE['name']]['split']='validation'
    with pytest.raises(experiment.old.IntegrityError):
        experiment.previous.selection(extension_context,reports)


def test_changed_v25_reference_rejected(extension_context,tmp_path):
    path=extension_context['v25_output']/'tasks'/f"calibration_{experiment.BASELINE['name']}"/'result.json'
    path.write_bytes(path.read_bytes()+b' ')
    values=experiment.previous.previous.load_vectors(extension_context,'calibration')
    with pytest.raises(experiment.old.IntegrityError):
        experiment.evaluate(extension_context,'calibration',experiment.BASELINE,values,tmp_path/'check')


def test_notebook_is_clean_run_all():
    nb=nbformat.read(experiment.VARIANT/'search_fusion.ipynb',as_version=4)
    nbformat.validate(nb)
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    compile(code,'v26','exec')
    assert 'ALLOW_OUTER_EVALUATION = True' in code and 'experiment.run(context' in code
    assert not any(s in code for s in ('WALL_HOURS','optimizer.step','calibrate_policy'))
