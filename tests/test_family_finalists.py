"""v29: replay source calibration, compare exactly three systems, never retune."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys

import nbformat
import numpy as np
import pytest

from backend.core import ROOT, sha256
from training import family_finalists as e
from training.audit import digest
from tests.test_dual_role_experiment import context
from tests.test_map_search import search_context
from tests.test_fusion_extension import extension_context
from tests.test_late_fusion import late_context
from tests.test_multi_hypothesis import multi_context, mock_flip, values


def test_shortlist_is_exactly_authorized_systems():
    assert [s['name'] for s in e.systems()]==['V25_control','power_50_l50','flip_mvp_w75_l50']
    assert e.systems()[1]=={'name':'power_50_l50','family':'power','gamma':.5,'lambda':.5}
    assert e.systems()[2]=={'name':'flip_mvp_w75_l50','family':'flip','scope':'mvp','flip_weight':.75,'lambda':.5}


@pytest.fixture
def finalists_context(multi_context,mock_flip,monkeypatch):
    # Completed synthetic v28. Only non-shortlisted calibration metrics are lowered
    # to force the exact authorized family winners; the three replayed values are real.
    m=multi_context['manifest']
    m['version']=28
    m['map_search_plan']['systems']=e.inference.systems()
    multi_context['signature']=digest(m)
    evaluate=e.previous.evaluate
    def force(c,split,spec,v,d):
        r=evaluate(c,split,spec,v,d)
        if split=='calibration' and spec not in e.systems(): r['ranking']['mAP@10']=0.
        return r
    with monkeypatch.context() as patch:
        patch.setattr(e.previous,'evaluate',force)
        r=e.previous.run(multi_context,allow_outer=True)
    e.old.review.old.freeze_json(multi_context['output']/'manifest.json',m)
    plan={'source_directory':str(multi_context['output']),'source_signature':multi_context['signature'],
          'source_result_sha256':sha256(multi_context['output']/'results.json'),'systems':e.systems(),
          'calibration_reports':{s['name']:r['calibration'][s['name']] for s in e.systems()}}
    manifest={**m,'version':29,'finalists_plan':plan,'source_sha256':{**m['source_sha256'],
              'training/family_finalists.py':sha256(ROOT/'training/family_finalists.py')}}
    c={**multi_context,'output':multi_context['output'].parent/'v29','manifest':manifest,'signature':digest(manifest)}
    e.old.review.old.freeze_json(c['output']/'manifest.json',manifest)
    e.old.review.old.freeze_json(c['output']/'frozen_comparison.json',e.frozen_value(c))
    return c


@pytest.fixture
def fake_validation_flip(monkeypatch):
    calls=[]
    def fake(c,d):
        e.validation_ready(c)
        calls.append('validation')
        q,g=e.old.review.old.protocol_rows(c,'validation')
        path=d/'features.npy';np.save(path,values(len(q)+len(g),seed=39))
        return {'path':str(path),'sha256':sha256(path),'split':'validation','ids':[r['image_id'] for r in q+g]}
    monkeypatch.setattr(e,'flip_validation',fake)
    return calls


def test_family_eligibility_uses_complete_calibration_only(finalists_context):
    r=e.source_report(finalists_context)
    assert e.family_shortlist(r)==e.systems()
    for fault in ('incomplete','validation','nan','different_winner'):
        changed=deepcopy(r)
        if fault=='incomplete': changed['calibration'].pop('power_50_l50')
        elif fault=='validation': changed['calibration']['power_50_l50']['split']='validation'
        elif fault=='nan': changed['calibration']['power_50_l50']['ranking']['mAP@10']=float('nan')
        else: changed['calibration']['flip_mvp_w25_l50']['ranking']['mAP@10']=2.
        with pytest.raises(e.old.IntegrityError): e.family_shortlist(changed)


def test_full_comparison_exports_resumes_and_never_fits(finalists_context,fake_validation_flip,monkeypatch):
    def denied(*a,**kw): raise AssertionError('No training, calibration fitting or calibration image inference')
    monkeypatch.setattr(e.dual.policy,'calibrate_policy',denied)
    monkeypatch.setattr(e.inference,'encode_rows',denied)
    with pytest.raises(ValueError,match='allow_outer'): e.run(finalists_context)
    r=e.run(finalists_context,allow_outer=True)
    assert r['status']=='complete' and r['protected_unchanged']
    assert r['optimizer_updates']==0 and not r['threshold_fit'] and not r['promoted']
    assert list(r['evaluations'])==[s['name'] for s in e.systems()]
    assert len(r['events'])==5 and fake_validation_flip==['validation']
    assert r['calibration_replay']['encoder_forwards']==0
    assert all(d['C_delta']==0 for d in r['vs_v25'].values())
    assert r['best_observed_validation'] in e.systems()
    control=Path(r['evaluations']['V25_control']['export'])
    src=Path(finalists_context['manifest']['finalists_plan']['source_directory'])/'tasks/validation_V25_control/export'
    for name in ('submission.csv','candidates.csv','embeddings.npy'):
        assert sha256(control/name)==sha256(src/name)
    for s in e.systems():
        p=Path(r['evaluations'][s['name']]['export']);v=np.load(p/'embeddings.npy')
        assert v.shape[1]==(4096 if s['family']=='flip' else 2048)
        np.testing.assert_array_equal(v[:,:2048],np.load(control/'embeddings.npy'))
        assert sha256(p/'candidates.csv')==sha256(control/'candidates.csv')
    again=e.run(finalists_context,allow_outer=True)
    assert r['evaluations']==again['evaluations'] and all(x['status']=='cached' for x in again['events'])
    assert fake_validation_flip==['validation']
    assert 'не независимый тест' in (finalists_context['output']/'REPORT.md').read_text()


def test_validation_gates_and_immutable_shortlist(finalists_context,fake_validation_flip):
    c=finalists_context;original=e.source.load_vectors(c,'validation')
    with pytest.raises(FileNotFoundError): e.evaluate(c,e.BASELINE,original,c['output']/'denied')
    e.old.Queue(c).task('verify_calibration',lambda d:e.verify_calibration(c,d))
    extra=next(s for s in e.inference.systems() if s not in e.systems())
    with pytest.raises(e.old.IntegrityError): e.evaluate(c,extra,original,c['output']/'denied')
    frozen=e.read_frozen(c);frozen['evaluations'].append(extra)
    e.write_json(c['output']/'frozen_comparison.json',frozen)
    with pytest.raises(e.old.IntegrityError): e.run(c,allow_outer=True)
    assert e.old.read(c['output']/'results.json')['status']=='incomplete'
    assert not fake_validation_flip


def test_source_and_flip_cache_corruption_stop_before_validation(finalists_context,fake_validation_flip):
    c=finalists_context;src=Path(c['manifest']['finalists_plan']['source_directory'])
    cache=src/'tasks/flip_calibration/features.npy'
    cache.write_bytes(cache.read_bytes()+b'changed')
    with pytest.raises(e.old.IntegrityError): e.run(c,allow_outer=True)
    assert not fake_validation_flip
    assert e.old.read(c['output']/'results.json')['evaluations']=={}


def test_failed_power_does_not_skip_flip_or_claim_complete(finalists_context,fake_validation_flip,monkeypatch):
    evaluate=e.evaluate
    def fail(c,s,v,d):
        if s['family']=='power': raise ValueError('Synthetic isolated power error')
        return evaluate(c,s,v,d)
    monkeypatch.setattr(e,'evaluate',fail)
    with pytest.raises(ValueError,match='Incomplete finalist'): e.run(finalists_context,allow_outer=True)
    r=e.old.read(finalists_context['output']/'results.json')
    assert r['status']=='incomplete' and r['evaluations']['power_50_l50'] is None
    assert r['evaluations']['flip_mvp_w75_l50'] and 'best_observed_validation' not in r
    monkeypatch.setattr(e,'evaluate',evaluate)
    assert e.run(finalists_context,allow_outer=True)['status']=='complete'
    assert fake_validation_flip==['validation']


def test_changed_threshold_and_calibration_replay_rejected(finalists_context,monkeypatch):
    monkeypatch.setattr(e.source,'frozen_profile',lambda c:{'threshold':.2})
    with pytest.raises(e.old.IntegrityError,match='Calibration replay'): e.run(finalists_context,allow_outer=True)
    assert e.old.read(finalists_context['output']/'results.json')['evaluations']=={}


def test_actual_flip_blocks_resume_and_reject_corruption(finalists_context,monkeypatch,tmp_path):
    c=finalists_context
    e.old.Queue(c).task('verify_calibration',lambda d:e.verify_calibration(c,d))
    original=e.source.load_vectors(c,'validation')
    q,g=e.old.review.old.protocol_rows(c,'validation');index={r['image_id']:i for i,r in enumerate(q+g)}
    calls=[]
    monkeypatch.setattr(e.dual,'DualRoleEncoder',lambda path:object())
    def encode(encoder,rows,dataset,*,flip,batch_size):
        positions=[index[r['image_id']] for r in rows]
        calls.append((flip,positions))
        if flip and positions[0]==8: raise KeyboardInterrupt('Synthetic interruption')
        return original[positions].copy()
    monkeypatch.setattr(e.inference,'encode_rows',encode)
    directory=tmp_path/'flip'
    with pytest.raises(KeyboardInterrupt): e.flip_validation(c,directory)
    assert len(list(directory.glob('block_*.json')))==2
    calls.clear()
    def resumed(encoder,rows,dataset,*,flip,batch_size):
        positions=[index[r['image_id']] for r in rows];calls.append((flip,positions))
        return original[positions].copy()
    monkeypatch.setattr(e.inference,'encode_rows',resumed)
    report=e.flip_validation(c,directory)
    assert report['cached_flip_images_this_attempt']==8 and report['fresh_flip_images_this_attempt']==10
    assert [ids[0] for flip,ids in calls if flip]==[8,12,16]
    p=directory/'block_00000.npy';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(e.old.IntegrityError): e.flip_validation(c,directory)


@pytest.mark.parametrize('spec',e.systems(),ids=lambda s:s['name'])
def test_fixed_finalists_query_independence(spec):
    v=values(48)
    if spec['family']=='flip': v=np.concatenate([v,values(48,seed=35)],axis=1)
    q,g=v[:33],v[33:];ref=e.inference.rank(v,len(q),spec)
    for indices in (np.arange(33)[::-1],np.array([4])):
        r=e.inference.rank(np.concatenate([q[indices],g]),len(indices),spec)
        np.testing.assert_array_equal(r['order'],ref['order'][indices])
        np.testing.assert_array_equal(r['raw_order'],ref['raw_order'][indices])
    for size in (1,8,16,32):
        parts=[e.inference.rank(np.concatenate([q[i:i+size],g]),len(q[i:i+size]),spec) for i in range(0,len(q),size)]
        for key in ('order','raw_order'):
            np.testing.assert_array_equal(np.concatenate([p[key] for p in parts]),ref[key])
        np.testing.assert_array_equal(np.concatenate([p['confidence'] for p in parts])>=.534365177154541,
                                      ref['confidence']>=.534365177154541)


def test_notebook_run_all_without_search_or_time_limit():
    nb=nbformat.read(e.VARIANT/'compare_family_finalists.ipynb',as_version=4)
    nbformat.validate(nb)
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    compile(code,'v29','exec')
    assert 'ALLOW_OUTER_EVALUATION = True' in code and 'experiment.run(context' in code
    assert not any(s in code for s in ('WALL_HOURS','optimizer.step','calibrate_policy'))


@pytest.mark.parametrize('loaded,disabled',[(False,None),(False,'0'),(True,None),(True,'1')])
def test_notebook_disables_telemetry_before_native_import(loaded,disabled):
    nb=nbformat.read(e.VARIANT/'compare_family_finalists.ipynb',as_version=4)
    config=next(c.source for c in nb.cells if c.id=='v29-config')
    bootstrap=config.split('candidates = ',1)[0]
    setup='import sys\n'
    if loaded: setup+="sys.modules['onnxruntime'] = object()\n"
    env={k:v for k,v in os.environ.items() if k!='ORT_DISABLE_TELEMETRY'}
    if disabled is not None: env['ORT_DISABLE_TELEMETRY']=disabled
    result=subprocess.run([sys.executable,'-c',setup+bootstrap+"\nprint(os.environ['ORT_DISABLE_TELEMETRY'])"],
                          env=env,capture_output=True,text=True)
    if loaded and disabled!='1':
        assert result.returncode!=0 and 'Restart Kernel' in result.stderr
    else:
        assert result.returncode==0 and result.stdout.strip()=='1'


def test_notebook_test_subprocess_keeps_failure_gate(monkeypatch):
    nb=nbformat.read(e.VARIANT/'compare_family_finalists.ipynb',as_version=4)
    cell=next(c.source for c in nb.cells if c.id=='v29-tests')
    def fail(command,*,check,env):
        assert check is True and env['ORT_DISABLE_TELEMETRY']=='1'
        assert command==[sys.executable,'-m','pytest','-q','tests/test_family_finalists.py',
                         'tests/test_multi_hypothesis.py','tests/test_dual_role_inference.py']
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setenv('ORT_DISABLE_TELEMETRY','0')
    monkeypatch.setattr(subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):
        exec(cell,{'subprocess':subprocess,'sys':sys,'os':os})
