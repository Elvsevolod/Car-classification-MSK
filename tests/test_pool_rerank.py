"""v30: fixed raw pools, unchanged candidate role, safe selection/export/resume."""
from copy import deepcopy
from pathlib import Path
import csv
import os
import subprocess
import sys

import nbformat
import numpy as np
import pytest

from backend.core import ROOT, normalize, sha256
from training import pool_rerank as e, pool_rerank_inference as inference
from training.audit import digest
from tests.test_dual_role_experiment import context
from tests.test_map_search import search_context
from tests.test_fusion_extension import extension_context
from tests.test_late_fusion import late_context, values
from tests.test_multi_hypothesis import multi_context, mock_flip
from tests.test_family_finalists import finalists_context, fake_validation_flip


def test_grid_and_pool_membership_tail_and_graph_priority():
    assert e.systems() == [e.BASELINE, {'name':'raw_pool10','pool_k':10},
                          {'name':'raw_pool20','pool_k':20}, {'name':'raw_pool50','pool_k':50}]
    raw = np.array([[5, 1, 4, 3, 0, 2], [0, 2, 3, 1, 5, 4]])
    graph = np.array([[2, 4, 0, 1, 3, 5], [4, 5, 1, 3, 2, 0]])
    np.testing.assert_array_equal(inference.restrict_order(raw, graph, 3),
                                  [[4,1,5,3,0,2], [3,2,0,1,5,4]])
    np.testing.assert_array_equal(inference.restrict_order(raw, graph, 50), graph)
    np.testing.assert_array_equal(inference.restrict_order(raw, graph, 1), raw)
    for k in (0, -1, 2.5, True):
        with pytest.raises(ValueError): inference.restrict_order(raw, graph, k)


def test_control_exact_candidates_fixed_pool_is_fusion_not_r1():
    v = values(76, seed=37); n = 9
    ref = inference.v25.rank(v[:n], v[n:])
    mvp, r1 = e.dual.unpack(v)
    mixed = normalize(np.concatenate([normalize(mvp)*np.float32(np.sqrt(.5)),
                                      normalize(r1)*np.float32(np.sqrt(.5))], axis=1))
    q, g = normalize(mixed[:n]), normalize(mixed[n:])
    raw = np.argsort(-np.stack([g @ x for x in q]), axis=1, kind='stable')
    assert not np.array_equal(raw[:, :10], ref['raw_order'][:, :10])
    for spec in e.systems():
        ranked = e.rank(v, n, spec)
        for key in ('raw_order', 'confidence'):
            np.testing.assert_array_equal(ranked[key], ref[key])
        if spec == e.BASELINE:
            np.testing.assert_array_equal(ranked['order'], ref['order'])
        else:
            k = spec['pool_k']
            np.testing.assert_array_equal(np.sort(ranked['order'][:, :k]), np.sort(raw[:, :k]))
            np.testing.assert_array_equal(ranked['order'][:, k:], raw[:, k:])
            np.testing.assert_array_equal(ranked['order'], inference.restrict_order(raw, ref['order'], k))


@pytest.mark.parametrize('spec', e.systems(), ids=lambda s:s['name'])
def test_query_permutation_removal_and_batches(spec):
    v = values(96, seed=38); q, g = v[:33], v[33:]
    ref = e.rank(v, len(q), spec)
    for indices in (np.arange(33)[::-1], np.array([4])):
        r = e.rank(np.concatenate([q[indices], g]), len(indices), spec)
        for key in ('order', 'raw_order'):
            np.testing.assert_array_equal(r[key], ref[key][indices])
        np.testing.assert_array_equal(r['confidence'] >= .534365177154541,
                                      ref['confidence'][indices] >= .534365177154541)
    for size in (1, 8, 16, 32):
        parts = [e.rank(np.concatenate([q[i:i+size],g]), len(q[i:i+size]), spec) for i in range(0,len(q),size)]
        for key in ('order', 'raw_order'):
            np.testing.assert_array_equal(np.concatenate([p[key] for p in parts]), ref[key])
        np.testing.assert_array_equal(np.concatenate([p['confidence'] for p in parts]) >= .534365177154541,
                                      ref['confidence'] >= .534365177154541)


def test_stable_ties_and_invalid_vectors_or_spec():
    v = np.repeat(values(1), 66, axis=0)
    for spec in e.systems():
        ranked = e.rank(v, 3, spec)
        np.testing.assert_array_equal(ranked['order'], np.tile(np.arange(63), (3,1)))
    for n, spec, data in ((0,e.BASELINE,v), (len(v),e.BASELINE,v), (1,{'name':'extra'},v),
                           (1,{'name':'raw_pool10','pool_k':20},v), (1,e.BASELINE,v[:,:100])):
        with pytest.raises(ValueError): e.rank(data,n,spec)
    v[0,0] = np.nan
    with pytest.raises(ValueError): e.rank(v,3,e.systems()[1])


@pytest.fixture
def pool_context(finalists_context, fake_validation_flip, monkeypatch):
    c = finalists_context
    # Complete a synthetic v29 through the real controller; no research artifacts touched.
    c['manifest']['protected'] = {}
    c['signature'] = digest(c['manifest'])
    e.write_json(c['output']/'manifest.json', c['manifest'])
    e.write_json(c['output']/'frozen_comparison.json', e.previous.frozen_value(c))
    e.previous.run(c, allow_outer=True)
    sm, signature, protected = e.completed_source(c['output'])
    manifest = {**sm, 'version':30, 'protected':protected,
                'pool_rerank_plan':{'source_directory':str(c['output']), 'source_signature':signature, 'systems':e.systems()},
                'map_search_plan':{**sm['map_search_plan'], 'systems':e.systems()},
                'source_sha256':{**sm['source_sha256'],
                                'training/pool_rerank_inference.py':sha256(ROOT/'training/pool_rerank_inference.py')}}
    result = {**c, 'output':c['output'].parent/'v30', 'manifest':manifest,
              'signature':digest(manifest), 'protected':protected}
    e.old.review.old.freeze_json(result['output']/'manifest.json', manifest)
    def verify(context, rehash=False):
        for path, expected in context['protected'].items():
            if sha256(path) != expected: raise e.old.IntegrityError('Protected source changed')
        for path, expected in context['manifest']['source_sha256'].items():
            if sha256(ROOT/path) != expected: raise e.old.IntegrityError('Source code changed')
    monkeypatch.setattr(e.old.review, 'check_inputs', verify)
    return result


@pytest.mark.parametrize('winner', range(4))
def test_selection_export_resume_and_no_training(pool_context, monkeypatch, winner):
    c = pool_context; evaluate = e.evaluate
    def forced(ctx, split, spec, v, directory):
        r = evaluate(ctx,split,spec,v,directory)
        # Force each selection path using synthetic calibration only, not real data.
        if split == 'calibration': r['ranking']['mAP@10'] = .99 if spec == e.systems()[winner] else .1
        return r
    def denied(*a, **kw): raise AssertionError('No new encoder or fitting calls')
    monkeypatch.setattr(e,'evaluate',forced)
    monkeypatch.setattr(e.dual,'DualRoleEncoder',denied)
    monkeypatch.setattr(e.dual.policy,'calibrate_policy',denied)
    with pytest.raises(ValueError,match='allow_outer'): e.run(c)
    r = e.run(c,allow_outer=True)
    assert r['status']=='complete' and r['protected_unchanged']
    assert r['encoder_forwards']==r['optimizer_updates']==0 and not r['threshold_fit'] and not r['promoted']
    assert r['selection']['selected']==e.systems()[winner]
    names = ['V25_control'] + ([e.systems()[winner]['name']] if winner else [])
    assert list(r['evaluations'])==names and len(r['events'])==4+len(names)
    assert r['selected_vs_v25']['C_delta']==0
    src = Path(c['manifest']['pool_rerank_plan']['source_directory'])/'tasks/validation_V25_control/export'
    for name in names:
        export = Path(r['evaluations'][name]['export'])
        for artifact in ('candidates.csv','embeddings.npy'):
            assert sha256(export/artifact)==sha256(src/artifact)
        with (export/'submission.csv').open(newline='') as stream:
            rows = list(csv.reader(stream))
        assert len(rows)==6
        assert all(len(row[1:])==len(set(row[1:]))==10 for row in rows)
    assert sha256(Path(r['evaluations']['V25_control']['export'])/'submission.csv')==sha256(src/'submission.csv')
    monkeypatch.setattr(e,'evaluate',denied)
    again = e.run(c,allow_outer=True)
    assert r['evaluations']==again['evaluations'] and all(x['status']=='cached' for x in again['events'])
    assert 'не независимый тест' in (c['output']/'REPORT.md').read_text()
    reports = deepcopy(r['calibration'])
    for report in reports.values(): report['ranking']['mAP@10']=.8
    assert e.search.selection(c,reports)['selected']==e.BASELINE


def test_failed_calibration_blocks_validation_but_continues_and_resumes(pool_context,monkeypatch):
    c = pool_context; evaluate = e.evaluate; load = e.source.load_vectors; seen = []
    def fail(ctx,split,spec,v,d):
        seen.append((split,spec['name']))
        if spec==e.systems()[1]: raise ValueError('Synthetic failed calibration')
        return evaluate(ctx,split,spec,v,d)
    def calibration_only(ctx,split):
        assert split=='calibration'
        return load(ctx,split)
    monkeypatch.setattr(e,'evaluate',fail)
    monkeypatch.setattr(e.source,'load_vectors',calibration_only)
    with pytest.raises(ValueError,match='Complete every calibration'): e.run(c,allow_outer=True)
    assert seen==[('calibration',s['name']) for s in e.systems()]
    assert not (c['output']/'frozen_selection.json').exists()
    assert e.old.read(c['output']/'results.json')['evaluations']=={}
    monkeypatch.setattr(e,'evaluate',evaluate)
    monkeypatch.setattr(e.source,'load_vectors',load)
    assert e.run(c,allow_outer=True)['status']=='complete'


def test_selection_gate_and_tampering(pool_context):
    c=pool_context; v=e.source.load_vectors(c,'validation')
    with pytest.raises(FileNotFoundError): e.evaluate(c,'validation',e.BASELINE,v,c['output']/'denied')
    r=e.run(c,allow_outer=True)
    excluded=next(s for s in e.systems() if s not in r['selection']['evaluations'])
    with pytest.raises(e.old.IntegrityError): e.evaluate(c,'validation',excluded,v,c['output']/'denied')
    frozen=deepcopy(r['selection']); frozen['selected']=excluded
    e.write_json(c['output']/'frozen_selection.json',frozen)
    with pytest.raises(ValueError,match='Changed protocol/configuration'): e.run(c,allow_outer=True)
    assert e.old.read(c['output']/'frozen_selection.json')==frozen  # Never overwrite a changed selection.


def test_validation_failure_retains_frozen_choice_and_resumes(pool_context,monkeypatch):
    c=pool_context; evaluate=e.evaluate
    def fail(ctx,split,spec,v,d):
        if split=='validation' and spec==e.systems()[1]:
            raise ValueError('Synthetic validation error')
        r=evaluate(ctx,split,spec,v,d)
        if split=='calibration': r['ranking']['mAP@10']=.99 if spec==e.systems()[1] else .1
        return r
    monkeypatch.setattr(e,'evaluate',fail)
    with pytest.raises(ValueError,match='Incomplete validation'): e.run(c,allow_outer=True)
    r=e.old.read(c['output']/'results.json')
    assert r['status']=='incomplete' and r['evaluations']['raw_pool10'] is None
    assert r['selection']['selected']==e.systems()[1]
    monkeypatch.setattr(e,'evaluate',evaluate)
    resumed=e.run(c,allow_outer=True)
    assert resumed['status']=='complete' and resumed['selection']==r['selection']
    assert [event['status'] for event in resumed['events']]==['cached']*5+['complete']


@pytest.mark.parametrize('fault', ['runtime','threshold','grid','source','export'])
def test_changed_inputs_fail_closed(pool_context,monkeypatch,fault):
    c=pool_context
    if fault=='runtime': c['manifest']['analysis_runtime']['numpy']='changed'
    elif fault=='threshold': monkeypatch.setattr(e.source,'frozen_profile',lambda c:{'threshold':.2})
    elif fault=='grid': c['manifest']['map_search_plan']['systems'].append({'name':'unplanned','pool_k':100})
    elif fault=='source':
        p=c['source_output']/'tasks/features_calibration_mvp/features.npz'
        p.write_bytes(p.read_bytes()+b'changed')
    else:
        e.run(c,allow_outer=True)
        p=c['output']/'tasks/validation_V25_control/export/submission.csv'
        p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(e.old.IntegrityError): e.run(c,allow_outer=True)


def test_completed_source_aggregate_must_match_receipt(pool_context):
    directory=Path(pool_context['manifest']['pool_rerank_plan']['source_directory'])
    result=e.old.read(directory/'results.json')
    result['evaluations']['V25_control']['ranking']['mAP@10']+=.01
    e.write_json(directory/'results.json',result)
    with pytest.raises(e.old.IntegrityError,match='aggregate'): e.completed_source(directory)


def test_pure_inference_has_no_training_imports():
    code="from training import pool_rerank_inference; import sys; assert 'torch' not in sys.modules; assert 'training.pool_rerank' not in sys.modules; assert 'training.family_finalists' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True,env={**os.environ,'ORT_DISABLE_TELEMETRY':'1'},cwd=ROOT)


def notebook():
    return nbformat.read(e.VARIANT/'search_pool_rerank.ipynb',as_version=4)


@pytest.mark.parametrize('executed', [False, True])
def test_run_all_notebook_and_failing_test_gate(monkeypatch, executed):
    nb=notebook(); nbformat.validate(nb)
    # Saved outputs are legitimate after Run All and must not block a resume.
    for cell in nb.cells:
        if cell.cell_type=='code':
            cell.execution_count=1 if executed else None
            cell.outputs=[nbformat.v4.new_output('stream', name='stdout', text='saved result\n')] if executed else []
    nbformat.validate(nb)
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    compile(code,'v30','exec')
    assert 'ALLOW_OUTER_EVALUATION = True' in code and 'experiment.run(context' in code
    assert not any(x in code for x in ('WALL_HOURS','optimizer.step','calibrate_policy'))
    cell=next(c.source for c in nb.cells if c.id=='v30-tests')
    def fail(command,*,check,env):
        assert check is True and env['ORT_DISABLE_TELEMETRY']=='1'
        assert 'tests/test_pool_rerank.py' in command
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError): exec(cell,{'subprocess':subprocess,'sys':sys,'os':os})


@pytest.mark.parametrize('loaded,disabled',[(False,None),(False,'0'),(True,None),(True,'1')])
def test_telemetry_disabled_before_import(loaded,disabled):
    bootstrap=next(c.source for c in notebook().cells if c.id=='v30-config').split('candidates = ',1)[0]
    setup='import sys\n'+("sys.modules['onnxruntime'] = object()\n" if loaded else '')
    env={k:v for k,v in os.environ.items() if k!='ORT_DISABLE_TELEMETRY'}
    if disabled is not None: env['ORT_DISABLE_TELEMETRY']=disabled
    r=subprocess.run([sys.executable,'-c',setup+bootstrap+"\nprint(os.environ['ORT_DISABLE_TELEMETRY'])"],env=env,capture_output=True,text=True)
    if loaded and disabled!='1': assert r.returncode!=0 and 'Restart Kernel' in r.stderr
    else: assert r.returncode==0 and r.stdout.strip()=='1'
