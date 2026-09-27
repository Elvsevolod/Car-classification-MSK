"""v31: unchanged model tensors, independent scale TTA, frozen selection and safe resume."""
from copy import deepcopy
from pathlib import Path
import csv
import os
import subprocess
import sys

import nbformat
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
import pytest

from backend.core import ROOT, normalize, sha256
from training import multiscale as e, multiscale_inference as inf
from training.audit import digest
from tests.test_dual_role_experiment import context
from tests.test_map_search import search_context
from tests.test_fusion_extension import extension_context
from tests.test_late_fusion import late_context, values
from tests.test_multi_hypothesis import multi_context, mock_flip
from tests.test_family_finalists import finalists_context, fake_validation_flip
from tests.test_pool_rerank import pool_context


def arrays(n=64):
    return np.concatenate([values(n, seed=41), normalize(np.random.default_rng(42).normal(size=(n,1536)).astype(np.float32))],axis=1)


def test_fixed_grid_and_fusion_formula_keep_original_candidates():
    assert e.systems()==[e.BASELINE]+[{'name':f'r1_s{s}_w{int(w*100)}','size':s,'high_weight':w}
                                    for s in (320,384) for w in (.25,.5)]
    v=arrays(); ref=inf.v25.rank(v[:8,:2048],v[8:,:2048]); before=v.copy()
    for spec in e.systems():
        actual=inf.rank(v[:,:2048] if spec==e.BASELINE else v,8,spec)
        for key in ('raw_order','confidence'): np.testing.assert_array_equal(actual[key],ref[key])
        if spec==e.BASELINE:
            np.testing.assert_array_equal(actual['order'],ref['order'])
        else:
            m,r=e.dual.unpack(v[:,:2048]); w=spec['high_weight']
            f=normalize(np.concatenate([normalize(m)*np.float32(np.sqrt(.5)),
                                       normalize(r)*np.float32(np.sqrt(.5*(1-w))),
                                       normalize(v[:,2048:])*np.float32(np.sqrt(.5*w))],axis=1))
            graph=inf.KReciprocalReranker(normalize(f[8:]),20,3)
            expected=np.argsort(np.stack([graph.distances(q,.5) for q in normalize(f[:8])]),axis=1,kind='stable')
            np.testing.assert_array_equal(actual['order'],expected)
    np.testing.assert_array_equal(v,before)


@pytest.mark.parametrize('spec', e.systems(), ids=lambda s:s['name'])
def test_queries_permutation_removal_batches(spec):
    v=arrays(96)
    if spec==e.BASELINE: v=v[:,:2048]
    q,g=v[:33],v[33:]; ref=inf.rank(v,33,spec)
    for indices in (np.arange(33)[::-1],np.array([4])):
        r=inf.rank(np.concatenate([q[indices],g]),len(indices),spec)
        for key in ('order','raw_order'): np.testing.assert_array_equal(r[key],ref[key][indices])
    for size in (1,8,16,32):
        parts=[inf.rank(np.concatenate([q[i:i+size],g]),len(q[i:i+size]),spec) for i in range(0,33,size)]
        for key in ('order','raw_order'): np.testing.assert_array_equal(np.concatenate([r[key] for r in parts]),ref[key])
        np.testing.assert_array_equal(np.concatenate([r['confidence'] for r in parts])>=.534365177154541,
                                      ref['confidence']>=.534365177154541)


def test_bad_shapes_specs_and_nonfinite_inputs():
    v=arrays()
    for data,n,spec in ((v,0,e.systems()[1]),(v,64,e.systems()[1]),(v,3,e.BASELINE),
                         (v[:,:2048],3,e.systems()[1]),(v,3,{'name':'unplanned'})):
        with pytest.raises(ValueError): inf.rank(data,n,spec)
    v[0,2050]=np.nan
    with pytest.raises(ValueError): inf.rank(v,3,e.systems()[1])


def tiny_model(seed=0):
    weights=np.random.default_rng(seed).normal(size=(3,512)).astype(np.float32)
    graph=helper.make_graph([
        helper.make_node('GlobalAveragePool',['image'],['pooled']),
        helper.make_node('Flatten',['pooled'],['flat'],axis=1),
        helper.make_node('MatMul',['flat','weights'],['embedding'])], 'scale-test',
        [helper.make_tensor_value_info('image',TensorProto.FLOAT,['batch',3,256,256])],
        [helper.make_tensor_value_info('embedding',TensorProto.FLOAT,['batch',512])],
        [numpy_helper.from_array(weights,'weights')])
    return helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=9)


def test_adaptation_changes_only_input_size_and_checks_corruption():
    model=tiny_model(); original=model.SerializeToString()
    for size in (256,320,384):
        payload=inf.resized_model_bytes(original,size)
        resized=onnx.load_model_from_string(payload)
        assert [d.dim_value for d in resized.graph.input[0].type.tensor_type.shape.dim[1:]]==[3,size,size]
        assert [t.SerializeToString() for t in resized.graph.initializer]==[t.SerializeToString() for t in model.graph.initializer]
        assert [n.SerializeToString() for n in resized.graph.node]==[n.SerializeToString() for n in model.graph.node]
        for d in resized.graph.input[0].type.tensor_type.shape.dim[2:]: d.dim_value=256
        assert resized.SerializeToString()==original
    with pytest.raises(ValueError): inf.resized_model_bytes(original,512)
    with pytest.raises(Exception): inf.resized_model_bytes(b'broken weights',320)
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value=208
    with pytest.raises(ValueError,match='256'): inf.resized_model_bytes(model.SerializeToString(),320)


def test_real_onnx_scale_encoder_same_bbox_png_jpeg_and_no_provider_fallback(tmp_path,monkeypatch):
    sources=[]
    for seed in range(3):
        path=tmp_path/f'model{seed}.onnx'; onnx.save_model(tiny_model(seed),path)
        sources.append({'model':str(path),'model_sha256':sha256(path)})
    monkeypatch.setattr(inf,'model_sources',lambda p:sources)
    folder=tmp_path/'images'; folder.mkdir()
    pixels=np.random.default_rng(7).integers(0,256,(13,19,3),dtype=np.uint8)
    for name in ('a.png','b.jpg'): Image.fromarray(pixels).save(folder/name)
    rows=[{'image_id':name,'x':1,'y':2,'w':17,'h':10,'vehicle_id':999,'camera_id':999} for name in ('a','b')]
    for size in (320,384):
        encoder=inf.R1ScaleEncoder('unused',size)
        v=encoder.encode_rows(rows,tmp_path,2)
        np.testing.assert_allclose(v,encoder.encode_rows(rows,tmp_path,1),rtol=0,atol=2e-5)
        assert v.shape==(2,1536)
        with Image.open(folder/'a.png') as image:
            expected=inf.resize_crop(inf.crop_image(image,(1,2,17,10)),'square',size)
            expected=(np.asarray(expected,dtype=np.float32)/255-inf.IMAGENET_MEAN)/inf.IMAGENET_STD
            np.testing.assert_array_equal(encoder.preprocess(image,(1,2,17,10)),expected.transpose(2,0,1))
        changed=[{k:v for k,v in row.items() if k not in ('vehicle_id','camera_id')} for row in rows]
        np.testing.assert_array_equal(encoder.encode_rows(changed,tmp_path,2),v)
        with pytest.raises(ValueError): encoder.encode_rows([{**rows[0],'w':100}],tmp_path,1)
    assert all(sha256(s['model'])==s['model_sha256'] for s in sources)
    with pytest.raises(RuntimeError,match='CPU'): inf.R1ScaleEncoder('unused',320,'CUDAExecutionProvider')
    monkeypatch.setattr(inf.frozen.ort,'get_available_providers',lambda:[])
    with pytest.raises(RuntimeError): inf.R1ScaleEncoder('unused',320)


@pytest.fixture
def scale_context(pool_context):
    c=pool_context; e.previous.run(c,allow_outer=True)
    sm,signature,protected=e.completed_source(c['output'])
    manifest={**sm,'version':31,'protected':protected,'scale_runtime':e.runtime(),
              'map_search_plan':{**sm['map_search_plan'],'systems':e.systems()},
              'multiscale_plan':{'source_directory':str(c['output']),'source_signature':signature,'systems':e.systems(),
                                'source_models':[],'batch_size':2,'chunk_size':4,'vector_atol':2e-5},
              'source_sha256':{**sm['source_sha256'], 'training/multiscale_inference.py':sha256(ROOT/'training/multiscale_inference.py')}}
    ctx={**c,'output':c['output'].parent/'v31','manifest':manifest,'signature':digest(manifest),'protected':protected}
    e.old.review.old.freeze_json(ctx['output']/'manifest.json',manifest)
    return ctx


@pytest.fixture
def mock_features(monkeypatch):
    calls=[]
    def fake(c,split,size,d):
        if split=='validation': assert e.search.read_selection(c)['selected']['size']==size
        q,g=e.old.review.old.protocol_rows(c,split)
        path=d/'features.npy'; np.save(path,arrays(len(q)+len(g))[:,2048:])
        calls.append((split,size))
        return {'path':str(path),'sha256':sha256(path),'split':split,'size':size,
                'ids':[r['image_id'] for r in q+g],'source_models':c['manifest']['multiscale_plan']['source_models']}
    monkeypatch.setattr(e,'features',fake)
    return calls


@pytest.mark.parametrize('winner',[0,1,4])
def test_frozen_selection_export_resume_without_training(scale_context,mock_features,monkeypatch,winner):
    c=scale_context; evaluate=e.evaluate
    def force(c,split,spec,v,d):
        r=evaluate(c,split,spec,v,d)
        if split=='calibration': r['ranking']['mAP@10']=.99 if spec==e.systems()[winner] else .1
        return r
    def denied(*a,**kw): raise AssertionError('No model training or threshold fitting')
    monkeypatch.setattr(e,'evaluate',force)
    monkeypatch.setattr(e.dual.policy,'calibrate_policy',denied)
    with pytest.raises(ValueError,match='allow_outer'): e.run(c)
    r=e.run(c,allow_outer=True)
    assert r['status']=='complete' and r['protected_unchanged']
    assert not r['threshold_fit'] and not r['promoted'] and r['optimizer_updates']==r['bn_updates']==0
    assert r['selection']['selected']==e.systems()[winner]
    expected=[('calibration',320),('calibration',384)]+([('validation',e.systems()[winner]['size'])] if winner else [])
    assert mock_features==expected
    assert len(r['evaluations'])==(2 if winner else 1)
    assert r['selected_vs_v25']['C_delta']==0
    control=Path(r['evaluations']['V25_control']['export'])
    source=Path(c['manifest']['multiscale_plan']['source_directory'])/'tasks/validation_V25_control/export'
    for name in ('submission.csv','candidates.csv','embeddings.npy'): assert sha256(control/name)==sha256(source/name)
    for name,report in r['evaluations'].items():
        exp=Path(report['export']); v=np.load(exp/'embeddings.npy')
        assert v.shape==(18,2048 if name=='V25_control' else 3584)
        np.testing.assert_array_equal(v[:,:2048],np.load(source/'embeddings.npy'))
        assert sha256(exp/'candidates.csv')==sha256(source/'candidates.csv')
        with (exp/'submission.csv').open() as f: rows=list(csv.reader(f))
        assert len(rows)==6 and all(len(row)==11 and len(set(row[1:]))==10 for row in rows)
    monkeypatch.setattr(e,'evaluate',denied)
    again=e.run(c,allow_outer=True)
    assert again['evaluations']==r['evaluations'] and all(x['status']=='cached' for x in again['events'])
    assert mock_features==expected
    reports=deepcopy(r['calibration'])
    for r in reports.values(): r['ranking']['mAP@10']=.8
    assert e.search.selection(c,reports)['selected']==e.BASELINE


def test_failed_scale_continues_other_scale_but_no_validation(scale_context,mock_features,monkeypatch):
    c=scale_context; features=e.features
    def fail(c,split,size,d):
        if size==320: raise ValueError('Synthetic extraction error')
        assert split=='calibration'
        return features(c,split,size,d)
    monkeypatch.setattr(e,'features',fail)
    with pytest.raises(ValueError,match='Complete every calibration'): e.run(c,allow_outer=True)
    result=e.old.read(c['output']/'results.json')
    assert result['status']=='incomplete' and result['evaluations']=={}
    assert result['calibration']['r1_s320_w25'] is None and result['calibration']['r1_s384_w50']
    assert not (c['output']/'frozen_selection.json').exists()
    monkeypatch.setattr(e,'features',features)
    assert e.run(c,allow_outer=True)['status']=='complete'


def test_validation_gates_and_changed_selection(scale_context,mock_features):
    c=scale_context; v=e.source.load_vectors(c,'validation')
    with pytest.raises(FileNotFoundError): e.evaluate(c,'validation',e.BASELINE,v,c['output']/'denied')
    r=e.run(c,allow_outer=True)
    excluded=next(s for s in e.systems() if s not in r['selection']['evaluations'])
    with pytest.raises(e.old.IntegrityError): e.evaluate(c,'validation',excluded,np.concatenate([v,v[:,512:]],axis=1),c['output']/'denied')
    frozen=deepcopy(r['selection']); frozen['selected']=excluded
    e.write_json(c['output']/'frozen_selection.json',frozen)
    with pytest.raises(ValueError): e.run(c,allow_outer=True)


def test_real_feature_blocks_resume_and_corruption(scale_context,monkeypatch):
    c=scale_context; rows=sum(e.old.review.old.protocol_rows(c,'calibration'),[])
    index={r['image_id']:i for i,r in enumerate(rows)}
    original=e.source.load_vectors(c,'calibration')[:,512:]; high=arrays(len(rows))[:,2048:]; calls=[]
    class Encoder:
        sources=[]; adapted_sha256=['synthetic']*3
        def __init__(self,path,size): self.size=size
        def encode_rows(self,entries,dataset,batch_size):
            ids=[index[r['image_id']] for r in entries]
            calls.append((self.size,ids))
            if self.size==320 and ids[0]==8 and interrupt[0]: raise KeyboardInterrupt('Synthetic interruption')
            return (original if self.size==256 else high)[ids].copy()
    interrupt=[True]; monkeypatch.setattr(inf,'R1ScaleEncoder',Encoder)
    directory=c['output']/'tasks/features_calibration_320'
    with pytest.raises(KeyboardInterrupt): e.features(c,'calibration',320,directory)
    assert (directory/'block_00000.json').exists() and (directory/'block_00004.json').exists()
    assert not (directory/'features.npy').exists()
    interrupt[0]=False; calls.clear()
    report=e.old.Queue(c).task('features_calibration_320',lambda d:e.features(c,'calibration',320,d))
    assert report['fresh_images_this_attempt']==len(rows)-8 and report['cached_images_this_attempt']==8
    np.testing.assert_array_equal(e.load_features(c,'calibration',320,report),high)
    assert (320,[4,5,6,7]) not in calls
    path=directory/'block_00000.npy'; path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(e.old.IntegrityError): e.features(c,'calibration',320,directory)
    with pytest.raises(e.old.IntegrityError): e.load_features(c,'calibration',320,report)


@pytest.mark.parametrize('fault',['runtime','threshold','grid','source','export'])
def test_changed_inputs_stop(scale_context,mock_features,monkeypatch,fault):
    c=scale_context
    if fault=='runtime': c['manifest']['scale_runtime']['onnx']='changed'
    elif fault=='threshold': monkeypatch.setattr(e.source,'frozen_profile',lambda c:{'threshold':.2})
    elif fault=='grid': c['manifest']['map_search_plan']['systems'].append({'name':'extra'})
    elif fault=='source':
        p=c['source_output']/'tasks/features_calibration_mvp/features.npz'; p.write_bytes(p.read_bytes()+b'changed')
    else:
        e.run(c,allow_outer=True)
        p=c['output']/'tasks/validation_V25_control/export/submission.csv'; p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(e.old.IntegrityError): e.run(c,allow_outer=True)


def test_original_image_parity_failure_is_not_ignored(scale_context,monkeypatch):
    c=scale_context
    class BadEncoder:
        sources=[]
        def __init__(self,path,size): pass
        def encode_rows(self,rows,dataset,batch_size): return arrays(len(rows))[:,2048:]
    monkeypatch.setattr(inf,'R1ScaleEncoder',BadEncoder)
    with pytest.raises(e.old.IntegrityError,match='256 image inference'):
        e.features(c,'calibration',320,c['output']/'failed')
    with pytest.raises(FileNotFoundError): e.features(c,'validation',320,c['output']/'denied')


def test_pure_inference_no_training_imports():
    code="from training import multiscale_inference; import sys; assert 'torch' not in sys.modules; assert 'training.multiscale' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True,env={**os.environ,'ORT_DISABLE_TELEMETRY':'1'},cwd=ROOT)


@pytest.mark.parametrize('executed',[False,True])
def test_notebook_run_all_accepts_saved_outputs_and_propagates_test_errors(monkeypatch,executed):
    nb=nbformat.read(e.VARIANT/'search_multiscale.ipynb',as_version=4)
    for c in nb.cells:
        if c.cell_type=='code':
            c.execution_count=1 if executed else None
            c.outputs=[nbformat.v4.new_output('stream',name='stdout',text='saved\n')] if executed else []
    nbformat.validate(nb)
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code'); compile(code,'v31','exec')
    assert 'ALLOW_OUTER_EVALUATION = True' in code and 'experiment.run(context' in code
    assert not any(s in code for s in ('WALL_HOURS','optimizer.step','calibrate_policy'))
    cell=next(c.source for c in nb.cells if c.id=='v31-tests')
    def fail(command,*,check,env):
        assert check is True and env['ORT_DISABLE_TELEMETRY']=='1' and 'tests/test_multiscale.py' in command
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError): exec(cell,{'subprocess':subprocess,'os':os,'sys':sys})


@pytest.mark.parametrize('loaded,disabled',[(False,None),(False,'0'),(True,None),(True,'1')])
def test_early_telemetry_guard(loaded,disabled):
    nb=nbformat.read(e.VARIANT/'search_multiscale.ipynb',as_version=4)
    code=next(c.source for c in nb.cells if c.id=='v31-config').split('candidates = ',1)[0]
    setup='import sys\n'+("sys.modules['onnxruntime']=object()\n" if loaded else '')
    env={k:v for k,v in os.environ.items() if k!='ORT_DISABLE_TELEMETRY'}
    if disabled is not None: env['ORT_DISABLE_TELEMETRY']=disabled
    p=subprocess.run([sys.executable,'-c',setup+code+"\nprint(os.environ['ORT_DISABLE_TELEMETRY'])"],env=env,capture_output=True,text=True)
    if loaded and disabled!='1': assert p.returncode!=0 and 'Restart Kernel' in p.stderr
    else: assert p.returncode==0 and p.stdout.strip()=='1'
