"""v27: exact control, fixed candidates, independent queries and fail-closed selection."""
from copy import deepcopy
from pathlib import Path

import nbformat
import numpy as np
import pytest

from backend.core import ROOT, normalize, sha256
from training import late_fusion as experiment
from training.late_fusion_inference import fusion_order, branch_distances
from tests.test_dual_role_experiment import context
from tests.test_map_search import search_context
from tests.test_fusion_extension import extension_context


def values(n=45, seed=27):
    rng = np.random.default_rng(seed)
    return np.concatenate([normalize(rng.normal(size=(n, d)).astype(np.float32)) for d in (512, 1536)], axis=1)


def test_grid_is_fixed_unique_with_v25_first():
    grid = experiment.systems()
    assert len(grid) == len({s['name'] for s in grid}) == 43
    assert grid[0] == experiment.BASELINE
    assert {s['r1_weight'] for s in grid[1:]} == {.1, .25, .4, .5, .6, .75, .9}
    assert {s['r1_lambda'] for s in grid[1:]} == {.5, .75}
    assert {s['fusion'] for s in grid[1:]} == {'distance', 'rrf10', 'rrf60'}


@pytest.mark.parametrize('method', ['distance', 'rrf10', 'rrf60'])
def test_fusion_formula_rank_one_and_stable_ties(method):
    a = np.array([[.4, .1, .8, .2], [1., 1., 1., 1.]])
    b = np.array([[.1, .8, .4, .2], [1., 1., 1., 1.]])
    if method == 'distance':
        expected = np.argsort(.75*a+.25*b, axis=1, kind='stable')
    else:
        k = 10 if method == 'rrf10' else 60
        expected = np.array([sorted(range(4), key=lambda j: -(.75/(k+[3,1,4,2][j])+.25/(k+[1,4,3,2][j]))),
                             [0,1,2,3]])
    np.testing.assert_array_equal(fusion_order(a,b,method,.25),expected)
    np.testing.assert_array_equal(fusion_order(a,b,method,0),np.argsort(a,axis=1,kind='stable'))
    np.testing.assert_array_equal(fusion_order(a,b,method,1),np.argsort(b,axis=1,kind='stable'))
    with pytest.raises(ValueError): fusion_order(a,b,method,float('nan'))
    with pytest.raises(ValueError): fusion_order(a,b[:1],method,.5)


def test_baseline_and_all_candidates_match_original_v25():
    v = values()
    reference = experiment.search.rank(v, 7, experiment.v26.BASELINE)
    for spec in experiment.systems():
        r = experiment.rank(v,7,spec)
        np.testing.assert_array_equal(r['raw_order'],reference['raw_order'])
        np.testing.assert_array_equal(r['confidence'],reference['confidence'])
        if spec == experiment.BASELINE:
            np.testing.assert_array_equal(r['order'],reference['order'])


def test_graphs_are_separate_and_fusion_uses_beyond_top10():
    v=values(40)
    mvp,r1=experiment.dual.unpack(v)
    left=branch_distances(mvp,5,.5)
    right=branch_distances(r1,5,.75)
    spec=next(s for s in experiment.systems() if s.get('fusion')=='distance'
              and s['r1_lambda']==.75 and s['r1_weight']==.5)
    actual=experiment.rank(v,5,spec)
    np.testing.assert_array_equal(actual['order'],fusion_order(left,right,'distance',.5))
    changed=v.copy()
    changed[:,:512]=values(40,seed=30)[:,:512]
    _,same_r1=experiment.dual.unpack(changed)
    np.testing.assert_array_equal(branch_distances(same_r1,5,.75),right)
    # Consensus at rank 11 in both lists must survive: never fuse truncated top-10 lists.
    a=np.array([list(range(10))+[100]*10+list(range(11,90))+[10]],dtype=float)
    b=np.array([[100]*10+list(range(10))+list(range(11,90))+[10]],dtype=float)
    assert fusion_order(a,b,'distance',.5)[0,0]==99
    assert fusion_order(a,b,'rrf60',.5)[0,0]==99


@pytest.mark.parametrize('method', ['features','distance','rrf10','rrf60'])
def test_query_permutation_removal_and_batch_sizes(method):
    spec=next(s for s in experiment.systems() if s['fusion']==method)
    v=values(58,seed=29); q,g=v[:33],v[33:]
    ref=experiment.rank(v,len(q),spec)
    for select in (np.arange(len(q))[::-1],np.array([4])):
        r=experiment.rank(np.concatenate([q[select],g]),len(select),spec)
        np.testing.assert_array_equal(r['order'],ref['order'][select])
        np.testing.assert_array_equal(r['raw_order'],ref['raw_order'][select])
        np.testing.assert_allclose(r['confidence'],ref['confidence'][select],atol=2e-6,rtol=0)
    for size in (1,8,16,32):
        parts=[experiment.rank(np.concatenate([q[i:i+size],g]),len(q[i:i+size]),spec)
               for i in range(0,len(q),size)]
        np.testing.assert_array_equal(np.concatenate([r['order'] for r in parts]),ref['order'])
        np.testing.assert_array_equal(np.concatenate([r['raw_order'] for r in parts]),ref['raw_order'])


def test_invalid_spec_and_vectors_rejected():
    v=values()
    with pytest.raises(ValueError): experiment.rank(v,0,experiment.BASELINE)
    with pytest.raises(ValueError): experiment.rank(v,5,{'name':'invented'})
    with pytest.raises(ValueError): experiment.rank(v[:,:100],5,experiment.BASELINE)
    with pytest.raises(ValueError): fusion_order(np.zeros((2,3)),np.zeros((2,3)),'unknown',.5)
    v[0,0]=np.nan
    with pytest.raises(ValueError): experiment.rank(v,5,experiment.systems()[1])


@pytest.fixture
def late_context(extension_context):
    experiment.v26.run(extension_context,allow_outer=True)
    return {**extension_context,'output':extension_context['output'].parent/'v27','signature':'v27',
            'reference_output':extension_context['output'], 'manifest':{**extension_context['manifest'],
            'source_sha256':{'training/late_fusion_inference.py':sha256(ROOT/'training/late_fusion_inference.py')},
            'late_fusion_plan':{'source_signature':extension_context['signature']},
            'map_search_plan':{**extension_context['manifest']['map_search_plan'],
                              'systems':[experiment.BASELINE,experiment.systems()[1],experiment.systems()[-1]]}}}


def test_controller_selection_export_resume_and_fixed_candidates(late_context,monkeypatch):
    def denied(*a,**kw): raise AssertionError('No encoder inference or threshold fitting')
    monkeypatch.setattr(experiment.dual,'DualRoleEncoder',denied)
    monkeypatch.setattr(experiment.dual.policy,'calibrate_policy',denied)
    with pytest.raises(ValueError,match='allow_outer'): experiment.run(late_context)
    with pytest.raises(FileNotFoundError): experiment.search.read_selection(late_context)
    # Exercise a late-fusion winner/export regardless of the synthetic dataset's preferred method.
    evaluate=experiment.evaluate
    def forced(context,split,spec,values,directory):
        r=evaluate(context,split,spec,values,directory)
        if split=='calibration': r['ranking']['mAP@10']=.99 if spec==experiment.systems()[-1] else .1
        return r
    monkeypatch.setattr(experiment,'evaluate',forced)
    r=experiment.run(late_context,allow_outer=True)
    assert r['status']=='complete' and r['protected_unchanged']
    assert r['encoder_forwards']==r['optimizer_updates']==0 and not r['threshold_fit'] and not r['promoted']
    assert r['selection']['selected']==experiment.systems()[-1] and len(r['evaluations'])==2
    assert r['selected_vs_v25']['C_delta']==0
    control=r['evaluations'][experiment.BASELINE['name']]
    winner=r['evaluations'][r['selection']['selected']['name']]
    for name in ('candidates.csv','embeddings.npy'):
        assert sha256(Path(control['export'])/name)==sha256(Path(winner['export'])/name)
    again=experiment.run(late_context,allow_outer=True)
    assert r['evaluations']==again['evaluations'] and all(x['status']=='cached' for x in again['events'])
    reports=deepcopy(r['calibration'])
    for x in reports.values(): x['ranking']['mAP@10']=.8
    assert experiment.search.selection(late_context,reports)['selected']==experiment.BASELINE
    reports[experiment.BASELINE['name']]['split']='validation'
    with pytest.raises(experiment.old.IntegrityError): experiment.search.selection(late_context,reports)
    with pytest.raises(experiment.old.IntegrityError):
        experiment.evaluate(late_context,'validation',experiment.systems()[2],
                            experiment.source.load_vectors(late_context,'validation'),late_context['output']/'denied')


def test_failure_cannot_freeze_selection(late_context,monkeypatch):
    evaluate=experiment.evaluate
    def failed(context,split,spec,values,directory):
        if spec!=experiment.BASELINE: raise ValueError('Synthetic incomplete calibration')
        return evaluate(context,split,spec,values,directory)
    monkeypatch.setattr(experiment,'evaluate',failed)
    with pytest.raises(ValueError,match='Complete every calibration'):
        experiment.run(late_context,allow_outer=True)
    assert not (late_context['output']/'frozen_selection.json').exists()
    assert experiment.old.read(late_context['output']/'results.json')['status']=='incomplete'


def test_source_corruption_and_threshold_change_rejected(late_context,tmp_path,monkeypatch):
    vectors=experiment.source.load_vectors(late_context,'calibration')
    original=experiment.source.frozen_profile
    monkeypatch.setattr(experiment.source,'frozen_profile',lambda c:{'threshold':.2})
    with pytest.raises(experiment.old.IntegrityError,match='threshold'):
        experiment.evaluate(late_context,'calibration',experiment.BASELINE,vectors,tmp_path/'threshold')
    monkeypatch.setattr(experiment.source,'frozen_profile',original)
    source=late_context['reference_output']/'tasks'/f"calibration_{experiment.v26.BASELINE['name']}"/'result.json'
    source.write_bytes(source.read_bytes()+b' ')
    with pytest.raises(experiment.old.IntegrityError):
        experiment.evaluate(late_context,'calibration',experiment.BASELINE,vectors,tmp_path/'corruption')


def test_notebook_run_all_without_training_or_time_limit():
    nb=nbformat.read(experiment.VARIANT/'search_late_fusion.ipynb',as_version=4)
    nbformat.validate(nb)
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    compile(code,'v27','exec')
    assert 'ALLOW_OUTER_EVALUATION = True' in code and 'experiment.run(context' in code
    assert not any(s in code for s in ('WALL_HOURS','optimizer.step','calibrate_policy'))
