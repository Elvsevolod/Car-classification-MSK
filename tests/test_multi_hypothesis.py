"""v28: fixed families, streaming independence, isolated failures and restartable TTA."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import nbformat
import numpy as np
import pytest
from PIL import Image

from backend.core import ROOT, normalize, sha256
from training import multi_hypothesis as e
from training import multi_hypothesis_inference as inf
from tests.test_dual_role_experiment import context
from tests.test_map_search import search_context
from tests.test_fusion_extension import extension_context
from tests.test_late_fusion import late_context


def values(n=45, seed=28):
    rng = np.random.default_rng(seed)
    parts = [normalize(rng.normal(size=(n, 512)).astype(np.float32)) for _ in range(4)]
    return inf.dual.pack(parts[0], inf.dual.policy.combine_members(parts[1:]))


def test_grid_counts_unique_names_and_control_first():
    grid = inf.systems()
    assert len(grid) == len({s['name'] for s in grid}) == 57
    assert grid[0] == inf.BASELINE
    assert Counter(s['family'] for s in grid) == dict(control=1, members=24, power=6, center=8, flip=18)
    assert {s['lambda'] for s in grid} == {.5, .75}


def test_every_system_preserves_raw_candidate_and_input_arrays():
    original, flipped = values(), values(seed=29)
    saved = original.copy()
    ref = inf.v25.rank(original[:5], original[5:])
    for spec in inf.systems():
        v = np.concatenate([original, flipped], axis=1) if spec['family']=='flip' else original
        actual = inf.rank(v, 5, spec)
        np.testing.assert_array_equal(actual['raw_order'], ref['raw_order'])
        np.testing.assert_array_equal(actual['confidence'], ref['confidence'])
        assert all(len(set(row))==40 for row in actual['order'])
        if spec==inf.BASELINE:
            np.testing.assert_array_equal(actual['order'], ref['order'])
    np.testing.assert_array_equal(original, saved)


@pytest.mark.parametrize('family', ['control', 'members', 'power', 'center', 'flip'])
def test_query_permutation_removal_and_batches(family):
    spec = next(s for s in inf.systems() if s['family']==family)
    v = values(48)
    if family=='flip': v = np.concatenate([v, values(48, seed=30)], axis=1)
    q,g = v[:33], v[33:]
    ref = inf.rank(v, len(q), spec)
    for indices in (np.arange(33)[::-1], np.array([4])):
        r = inf.rank(np.concatenate([q[indices],g]),len(indices),spec)
        np.testing.assert_array_equal(r['order'],ref['order'][indices])
        np.testing.assert_array_equal(r['raw_order'],ref['raw_order'][indices])
        np.testing.assert_array_equal(r['confidence']>=.534365177154541,ref['confidence'][indices]>=.534365177154541)
    for size in (1,8,16,32):
        parts=[inf.rank(np.concatenate([q[i:i+size],g]),len(q[i:i+size]),spec) for i in range(0,len(q),size)]
        np.testing.assert_array_equal(np.concatenate([x['order'] for x in parts]),ref['order'])
        np.testing.assert_array_equal(np.concatenate([x['raw_order'] for x in parts]),ref['raw_order'])


def test_member_power_center_and_flip_formulas():
    v=values(); m,r=inf.dual.unpack(v)
    for family in inf.FAMILIES:
        spec=next(s for s in inf.systems() if s['family']==family)
        source=v
        if family=='members':
            rm=inf.members(r)
            expected=inf.mix(m,normalize(np.concatenate([rm[i] for i in spec['members']],axis=1)),spec['r1_weight'])
        elif family=='power':
            expected=inf.mix(*[normalize(np.sign(b)*np.abs(b)**np.float32(spec['gamma'])) for b in (m,r)])
        elif family=='center':
            expected=inf.mix(*[normalize(b-np.float32(spec['strength'])*b[5:].mean(axis=0)) for b in (m,r)])
        else:
            flip=values(seed=29);fm,fr=inf.dual.unpack(flip)
            source=np.concatenate([v,flip],axis=1)
            assert spec['scope']=='mvp'
            expected=inf.mix(normalize(.75*m+.25*fm),r)
        np.testing.assert_array_equal(inf.ranking_features(source,5,spec),expected)
    # The saved R1 is actually a concatenation, not three coordinates of one model.
    np.testing.assert_allclose(inf.dual.policy.combine_members(inf.members(r)),r,rtol=0,atol=2e-7)
    pair=next(s for s in inf.systems() if s['family']=='members' and len(s['members'])==2)
    rm=inf.members(r)
    expected=inf.mix(m,normalize(np.concatenate([rm[0],rm[1]],axis=1)),.5)
    np.testing.assert_array_equal(inf.ranking_features(v,5,pair),expected)
    center=next(s for s in inf.systems() if s['family']=='center')
    changed=v.copy();changed[:5]=values(5,seed=90)
    np.testing.assert_array_equal(inf.ranking_features(v,5,center)[5:],inf.ranking_features(changed,5,center)[5:])


def test_bad_specs_shapes_and_nonfinite_vectors():
    v=values()
    with pytest.raises(ValueError): inf.rank(v,0,inf.BASELINE)
    with pytest.raises(ValueError): inf.rank(v,5,{'name':'invented'})
    with pytest.raises(ValueError): inf.rank(v,5,inf.systems()[-1])
    with pytest.raises(ValueError): inf.rank(np.concatenate([v,v],axis=1),5,inf.BASELINE)
    v[0,0]=np.nan
    with pytest.raises(ValueError): inf.rank(v,5,inf.systems()[1])


def test_flip_mirrors_pixels_not_labels_or_bbox(tmp_path,monkeypatch):
    folder=tmp_path/'images';folder.mkdir()
    pixels=np.zeros((5,7,3),np.uint8);pixels[:,:3]=60;pixels[:,3:]=190
    Image.fromarray(pixels).save(folder/'a.png')
    rows=[{'image_id':'a','x':1,'y':1,'w':5,'h':3,'camera_id':999,'vehicle_id':999}]
    prep=lambda image,box: np.asarray(image.crop((box[0],box[1],box[0]+box[2],box[1]+box[3])),dtype=np.float32).transpose(2,0,1)
    monkeypatch.setattr(inf,'preprocess',prep)
    class Encoder:
        def __init__(self): self.inputs=[]
        preprocess=staticmethod(prep)
        def encode_batch(self,batch):
            self.inputs.append(np.asarray(batch).copy())
            return normalize(np.ones((len(batch),512),np.float32))
    class Bundle: pass
    b=Bundle();b.mvp=Encoder();b.members=[Encoder() for _ in range(3)]
    original=inf.encode_rows(b,rows,tmp_path,flip=False)
    flipped=inf.encode_rows(b,rows,tmp_path,flip=True)
    for model in [b.mvp,*b.members]:
        np.testing.assert_array_equal(model.inputs[1],model.inputs[0][...,::-1])
        assert model.inputs[0].shape==(1,3,3,5)
    assert original.shape==flipped.shape==(1,2048)
    with pytest.raises(ValueError): inf.encode_rows(b,rows,tmp_path,flip=True,batch_size=0)


@pytest.fixture
def multi_context(late_context):
    e.previous.run(late_context,allow_outer=True)
    grid=[e.BASELINE]+[next(s for s in e.systems() if s['family']==f) for f in inf.FAMILIES]
    return {**late_context,'output':late_context['output'].parent/'v28','signature':'v28',
            'reference_output':late_context['output'],'manifest':{**late_context['manifest'],
            'source_sha256':{p:sha256(ROOT/p) for p in ('training/multi_hypothesis.py','training/multi_hypothesis_inference.py')},
            'map_search_plan':{**late_context['manifest']['map_search_plan'],'systems':grid},
            'multi_hypothesis_plan':{'source_signature':late_context['signature'],'flip_batch_size':2,
                                     'flip_chunk_size':4,'flip_parity_atol':2e-5}}}


@pytest.fixture
def mock_flip(monkeypatch):
    calls=[]
    def fake(c,split,d):
        calls.append(split)
        q,g=e.old.review.old.protocol_rows(c,split)
        v=values(len(q)+len(g),seed=33)
        path=d/'features.npy';np.save(path,v)
        return {'path':str(path),'sha256':sha256(path),'split':split,'ids':[r['image_id'] for r in q+g]}
    monkeypatch.setattr(e,'flip_features',fake)
    return calls


@pytest.mark.parametrize('winner_family', ['control','members','power','center','flip'])
def test_controller_selection_export_resume(multi_context,mock_flip,monkeypatch,winner_family):
    original=e.evaluate
    def force(c,split,spec,v,d):
        r=original(c,split,spec,v,d)
        if split=='calibration': r['ranking']['mAP@10']=.99 if spec['family']==winner_family else .1
        return r
    monkeypatch.setattr(e,'evaluate',force)
    with pytest.raises(ValueError,match='allow_outer'): e.run(multi_context)
    r=e.run(multi_context,allow_outer=True)
    assert r['status']=='complete' and r['protected_unchanged'] and not r['promoted']
    assert not r['threshold_fit'] and r['optimizer_updates']==0
    assert r['selection']['selected']['family']==winner_family
    assert len(r['evaluations'])==(1 if winner_family=='control' else 2)
    assert mock_flip==(['calibration','validation'] if winner_family=='flip' else ['calibration'])
    assert r['selected_vs_v25']['C_delta']==0
    exports=[Path(v['export']) for v in r['evaluations'].values()]
    assert len({sha256(p/'candidates.csv') for p in exports})==1
    winner=r['evaluations'][r['selection']['selected']['name']]
    v=np.load(Path(winner['export'])/'embeddings.npy')
    assert v.shape[1]==(4096 if winner_family=='flip' else 2048)
    np.testing.assert_array_equal(v[:,:2048],e.source.load_vectors(multi_context,'validation'))
    again=e.run(multi_context,allow_outer=True)
    assert again['evaluations']==r['evaluations'] and all(x['status']=='cached' for x in again['events'])
    reports=deepcopy(r['calibration'])
    for item in reports.values(): item['ranking']['mAP@10']=.8
    assert e.search.selection(multi_context,reports)['selected']==e.BASELINE
    reports['V25_control']['split']='validation'
    with pytest.raises(e.old.IntegrityError): e.search.selection(multi_context,reports)
    denied=next(s for s in e.systems() if s not in r['selection']['evaluations'])
    with pytest.raises(e.old.IntegrityError):
        e.evaluate(multi_context,'validation',denied,e.source.load_vectors(multi_context,'validation'),multi_context['output']/'denied')


def test_isolated_error_continues_but_cannot_select_then_resume(multi_context,mock_flip,monkeypatch):
    original=e.evaluate
    def fail(c,split,spec,v,d):
        if spec['family']=='members': raise ValueError('Synthetic isolated experiment error')
        return original(c,split,spec,v,d)
    monkeypatch.setattr(e,'evaluate',fail)
    with pytest.raises(ValueError,match='Complete every calibration'): e.run(multi_context,allow_outer=True)
    r=e.old.read(multi_context['output']/'results.json')
    assert r['status']=='incomplete' and not r['evaluations']
    assert not (multi_context['output']/'frozen_selection.json').exists()
    assert all(v for v in r['calibration'].values() if v is not None)
    assert any(x['status']=='failed' for x in r['events'])
    assert r['calibration'][next(s['name'] for s in e.systems() if s['family']=='flip')]
    monkeypatch.setattr(e,'evaluate',original)
    assert e.run(multi_context,allow_outer=True)['status']=='complete'


def test_integrity_failure_stops_immediately(multi_context,mock_flip,monkeypatch):
    called=[]
    def fail(*a): called.append(1);raise e.old.IntegrityError('Synthetic source changed')
    monkeypatch.setattr(e,'evaluate',fail)
    with pytest.raises(e.old.IntegrityError): e.run(multi_context,allow_outer=True)
    assert called==[1] and not mock_flip
    assert not (multi_context['output']/'frozen_selection.json').exists()


def test_flip_chunk_resume_corruption_and_original_parity(multi_context,monkeypatch,tmp_path):
    original=e.source.load_vectors(multi_context,'calibration')
    q,g=e.old.review.old.protocol_rows(multi_context,'calibration')
    index={r['image_id']:i for i,r in enumerate(q+g)}
    calls=[]
    monkeypatch.setattr(e.dual,'DualRoleEncoder',lambda path:object())
    def encode(encoder,rows,dataset,*,flip,batch_size):
        calls.append((flip,len(rows)))
        return original[[index[r['image_id']] for r in rows]].copy()
    monkeypatch.setattr(inf,'encode_rows',encode)
    d=tmp_path/'flip'
    first=e.flip_features(multi_context,'calibration',d)
    assert first['fresh_flip_images_this_attempt']==len(q)+len(g)
    assert calls[0]==(False,2) and len(calls)==6
    calls.clear()
    again=e.flip_features(multi_context,'calibration',d)
    assert not calls and again['fresh_flip_images_this_attempt']==0
    p=d/'block_00000.npy';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(e.old.IntegrityError,match='cache block'): e.flip_features(multi_context,'calibration',d)
    monkeypatch.setattr(inf,'encode_rows',lambda *a,**kw:values(len(a[1])))
    with pytest.raises(e.old.IntegrityError,match='Original image'): e.flip_features(multi_context,'calibration',tmp_path/'bad')
    with pytest.raises(FileNotFoundError): e.flip_features(multi_context,'validation',tmp_path/'denied')


def test_notebook_clean_and_run_all():
    nb=nbformat.read(e.VARIANT/'run_multi_hypothesis.ipynb',as_version=4)
    nbformat.validate(nb)
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    compile(code,'v28','exec')
    assert 'ALLOW_OUTER_EVALUATION = True' in code and 'experiment.run(context' in code
    assert not any(s in code for s in ('WALL_HOURS','optimizer.step','calibrate_policy'))


def test_interrupted_flip_reuses_only_completed_blocks(multi_context,monkeypatch,tmp_path):
    original=e.source.load_vectors(multi_context,'calibration')
    q,g=e.old.review.old.protocol_rows(multi_context,'calibration')
    index={r['image_id']:i for i,r in enumerate(q+g)}
    batches=[]
    monkeypatch.setattr(e.dual,'DualRoleEncoder',lambda path:object())
    def encode(encoder,rows,dataset,*,flip,batch_size):
        positions=[index[r['image_id']] for r in rows]
        if flip:
            batches.append(positions)
            if positions[0]==8: raise KeyboardInterrupt('Synthetic interruption')
        return original[positions].copy()
    monkeypatch.setattr(inf,'encode_rows',encode)
    directory=tmp_path/'interrupted'
    with pytest.raises(KeyboardInterrupt): e.flip_features(multi_context,'calibration',directory)
    assert len(list(directory.glob('block_*.json')))==2
    batches.clear()
    def resume(encoder,rows,dataset,*,flip,batch_size):
        positions=[index[r['image_id']] for r in rows]
        if flip: batches.append(positions)
        return original[positions].copy()
    monkeypatch.setattr(inf,'encode_rows',resume)
    r=e.flip_features(multi_context,'calibration',directory)
    assert r['cached_flip_images_this_attempt']==8 and r['fresh_flip_images_this_attempt']==10
    assert [batch[0] for batch in batches]==[8,12,16]
    np.testing.assert_array_equal(np.load(r['path']),original)
