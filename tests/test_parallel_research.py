import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from training import research_io as io, research_models as models, research_scoring as scoring
from training import research_training as training, research_gpu as gpu, research_quick as quick
from training import map_inference
from training import research_inputs as inputs

torch.set_num_threads(2)


def test_grids_are_distinct_and_predeclared():
    trans=gpu.trans_grid(); nive=gpu.nive_grid({"encoder_lr":1e-5,"weight_decay":1e-4})
    assert len(trans)==12 and len(nive)==19 and len(gpu.frozen_grid())==10 and len(quick.fusion_grid())==17
    assert len({s["id"] for s in trans+nive+gpu.frozen_grid()})==41
    assert sum(s["aux"]=="nive" for s in nive)==9
    assert sum(s["aux"]=="target" for s in nive)==9
    assert all(s["p"]*s["k"] in (16,64) for s in trans)


@pytest.mark.parametrize("mode",["G1","G2"])
def test_zero_weight_exact_control_and_query_independence(mode):
    rng=np.random.default_rng(22)
    bank=np.concatenate([scoring.mix([rng.normal(size=(21,512))],[1]),
                         scoring.mix([rng.normal(size=(21,1536))],[1])],axis=1)
    expert=scoring.mix([rng.normal(size=(21,9))],[1])
    q,g=bank[:5],bank[5:]
    graph=scoring.FixedGraph(g); raw,jac=graph.components(q)
    d=scoring.fuse_distances(raw,jac,expert[:5],expert[5:],mode,0)
    assert np.array_equal(np.argsort(d,axis=1,kind="stable"),map_inference.rank(q,g)["order"])
    full=scoring.fuse_distances(raw,jac,expert[:5],expert[5:],mode,.1)
    for index in ([4,2,0],[3],[1,0,4,3,2]):
        r,j=graph.components(q[index])
        part=scoring.fuse_distances(r,j,expert[index],expert[5:],mode,.1)
        assert np.array_equal(np.argsort(part,axis=1),np.argsort(full[index],axis=1))


def test_fusion_modes_are_different_and_do_not_change_input():
    raw=np.array([[.2,.8]],dtype=np.float32); jac=np.array([[.7,.3]],dtype=np.float32)
    q=np.array([[1.,0.]]); g=np.eye(2)
    assert not np.array_equal(scoring.fuse_distances(raw,jac,q,g,"G1",.2),scoring.fuse_distances(raw,jac,q,g,"G2",.2))
    np.testing.assert_array_equal(raw,np.array([[.2,.8]],dtype=np.float32))


@pytest.mark.parametrize("path",["../escape","/tmp/escape","C:/escape","a\\b","."])
def test_portable_path_escape_rejected(tmp_path,path):
    with pytest.raises(ValueError): io.child(tmp_path,path)


def test_hash_guard_and_completion(tmp_path):
    io.write(tmp_path/"result.json",{"ok":True}); io.finish(tmp_path,"abc")
    assert io.completed(tmp_path,"abc")
    io.write(tmp_path/"result.json",{"ok":False})
    with pytest.raises(ValueError): io.completed(tmp_path,"abc")


def test_bn_modes_isolate_updates():
    model=nn.Sequential(nn.BatchNorm1d(3))
    before=model[0].running_mean.clone()
    models.set_domain(model,"aux","target_updates_only")
    model(torch.ones(4,3)*10)
    assert torch.equal(before,model[0].running_mean)
    models.set_domain(model,"main","target_updates_only")
    model(torch.ones(4,3)*3)
    assert not torch.equal(before,model[0].running_mean)
    models.split_bn(model)
    main=model[0].main.running_mean.clone()
    models.set_domain(model,"aux","domain_specific"); model(torch.ones(4,3)*50)
    assert torch.equal(main,model[0].main.running_mean)
    assert not torch.equal(main,model[0].aux.running_mean)
    models.set_domain(model,"main","domain_specific",False)
    assert model[0].domain=="main" and not model.training


def test_nonlocal_matches_official_expression():
    torch.manual_seed(1)
    model=models.NonLocal(4).eval(); x=torch.randn(2,4,5,5)
    g=model.g(x).flatten(2).transpose(1,2)
    theta=model.theta(x).flatten(2).transpose(1,2); phi=model.phi(x).flatten(2)
    y=((theta@phi)/25@g).transpose(1,2).reshape(2,1,5,5)
    torch.testing.assert_close(model(x),x+model.W(y),atol=1e-6,rtol=1e-6)


def test_gradient_ratio_and_direction():
    result=training.gradient_diagnostics([torch.tensor([1.,0.])],[torch.tensor([-.5,0.])])
    assert result["weighted_aux_ratio"]==.5 and result["gradient_cosine"]==-1


def test_activation_checkpoint_retains_gradient_and_state_keys():
    original=nn.Sequential(nn.Linear(3,5),nn.GELU(),nn.Linear(5,3))
    model=copy.deepcopy(original); keys=list(model.state_dict())
    models.checkpoint_blocks([model])
    x=torch.randn(4,3)
    original(x).sum().backward(); model(x).sum().backward()
    assert list(model.state_dict())==keys
    for left,right in zip(model.parameters(),original.parameters()): torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)


def tiny_rows(root):
    rows=[]
    for identity in range(3):
        for n in range(2):
            name=f"{identity}_{n}.png"
            Image.new("RGB",(20,20),(identity*60+n*20,50,110)).save(root/name)
            rows.append({"image_id":name,"vehicle_id":identity,"camera_id":n,"path":name,
                         "label":identity,"x":0,"y":0,"w":20,"h":20})
    return rows


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=nn.Sequential(nn.Flatten(),nn.Linear(3*16*16,8))
        self.head=nn.Linear(8,3)
    def forward(self,x):
        f=self.base(x)
        return [self.head(f)],[f]


@pytest.mark.parametrize("device",["cpu",pytest.param("mps",marks=pytest.mark.skipif(
    os.environ.get("RUN_REID_MPS_SMOKE")!="1",reason="opt-in real MPS resume smoke"))])
def test_resume_replays_exact_training(tmp_path,device):
    if device=="mps": gpu.research_mac.preflight(io.ROOT)
    rows=tiny_rows(tmp_path)
    c={"signature":"tiny","device":torch.device(device),"inputs":tmp_path,"train":rows,"external":[],"manifest":{}}
    spec={"id":"test","family":"transreid","seed":3,"lr":1e-4,"weight_decay":.01,"p":2,"k":2,
          "size":16,"loss":"supcon","clip":5.}
    training.seed_all(4); full=Tiny().to(device); interrupted=copy.deepcopy(full)
    training.train(full,c,spec,4,4,tmp_path/"full")
    training.train(interrupted,c,spec,2,4,tmp_path/"resume")
    resumed=Tiny().to(device); training.train(resumed,c,spec,4,4,tmp_path/"resume")
    for left,right in zip(full.parameters(),resumed.parameters()): torch.testing.assert_close(left,right,rtol=0,atol=0)
    # Allocator samples depend on live models, not training state; compare every scientific value exactly.
    histories=[[{k:v for k,v in row.items() if k not in {"mps_allocated_bytes","mps_driver_bytes"}}
                for row in io.read(tmp_path/name/"history.json")] for name in ("full","resume")]
    assert histories[0]==histories[1]


def test_archive_includes_reusable_weights_and_features(tmp_path):
    directory=tmp_path/"run"; directory.mkdir()
    io.write(directory/"results.json",{"ok":True})
    training.save_torch(directory/"step_000001.pt",{"x":torch.ones(1)})
    np.save(directory/"features.npy",np.ones((2,3),np.float32))
    io.archive(directory,tmp_path/"full.zip")
    with zipfile.ZipFile(tmp_path/"full.zip") as archive:
        assert {"results.json","step_000001.pt","features.npy","PACKAGE_MANIFEST.json"} <= set(archive.namelist())
        assert archive.testzip() is None
    io.archive(directory,tmp_path/"light.zip",light=True)
    with zipfile.ZipFile(tmp_path/"light.zip") as archive: assert "features.npy" not in archive.namelist()


def test_windows_import_tree_does_not_require_fcntl():
    command="import sys; sys.modules['fcntl']=None; from training import research_gpu, research_quick"
    subprocess.run([sys.executable,"-c",command],cwd=io.ROOT,check=True,timeout=60)


def test_accelerator_does_not_fallback():
    with pytest.raises(ValueError,match="requires MPS or CUDA"): gpu.run("missing",device="cpu")


def test_failed_trial_does_not_stop_other_queue(tmp_path,monkeypatch):
    def fail(*args): raise RuntimeError("simulated OOM")
    monkeypatch.setattr(gpu,"evaluate_trial",fail)
    monkeypatch.setattr(gpu.runtime,"cleanup",lambda:None)
    c={"output":tmp_path}; spec={"id":"failed","family":"transreid"}
    assert gpu.attempt(c,spec,10) is None
    assert io.read(tmp_path/"status/failed__10.json")["status"]=="failed"


def test_dino_pooling_excludes_registers():
    class Fake(nn.Module):
        def forward_features(self,x):
            return {"x_norm_clstoken":torch.ones(len(x),3),"x_norm_patchtokens":torch.ones(len(x),4,3)*2,
                    "x_norm_regtokens":torch.ones(len(x),4,3)*1000}
    model=models.ExternalModel(Fake(),3,2,"dino","patch").eval()
    torch.testing.assert_close(model.raw(torch.zeros(2,3,2,2)),torch.ones(2,3)*2)


def test_notebooks_are_valid_and_compilable():
    import nbformat
    for path in (quick.VARIANT/"quick_fusion_bn.ipynb",gpu.VARIANT/"train_windows_queues.ipynb",
                 gpu.VARIANT/"train_mac_m4_queues.ipynb"):
        notebook=nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        code="\n".join(c.source for c in notebook.cells if c.cell_type=="code")
        compile(code,path.name,"exec")
        assert code.index("ORT_DISABLE_TELEMETRY") < code.index("from training")
        assert "timeout=" not in code
        # Saved outputs (including an earlier interrupted run) must not prevent retrying Run All.


def test_unpack_rejects_changed_archive_before_creating_output(tmp_path):
    bad=tmp_path/"bad.zip"; bad.write_bytes(b"invalid")
    with pytest.raises(ValueError,match="Wrong/incomplete"):
        inputs.unpack(bad,tmp_path/"extracted")
    assert not (tmp_path/"extracted").exists()


def test_input_pin_is_shared_by_windows_and_quick_bn():
    pin=io.read(gpu.VARIANT/"INPUT_PACKAGE.json")
    config=io.read(gpu.VARIANT/"configs/rtx4060_v1.json")
    assert pin["inputs_manifest_sha256"] == config["input_sha256"]


def test_bn_recalibration_changes_only_running_statistics(tmp_path):
    class TinyBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn=nn.BatchNorm1d(3)
            self.dropout=nn.Dropout(.8)
        def embedding(self,x):
            assert not self.dropout.training
            return self.bn(self.dropout(x.mean((2,3))))
    rows=tiny_rows(tmp_path); model=TinyBN()
    params={k:v.detach().clone() for k,v in model.named_parameters()}
    models.recalibrate_bn(model,tmp_path,rows,torch.device("cpu"),size=16,batch_size=5)
    assert model.bn.num_batches_tracked.item()==2 and model.bn.momentum==.1
    assert not model.training
    assert all(torch.equal(params[k],v) for k,v in model.named_parameters())


def test_target_extra_never_reuses_current_main_images(tmp_path):
    rows=tiny_rows(tmp_path)
    blocked={r["image_id"] for r in rows[:2]}
    indices=training.auxiliary_indices(rows,blocked,2,2,12,0)
    assert all(rows[i]["image_id"] not in blocked for i in indices)


def test_official_fastreid_release_head_aliases_and_normalization():
    source={"pixel_mean":torch.tensor([123.675,116.28,103.53]).reshape(1,3,1,1),
            "pixel_std":torch.tensor([58.395,57.12,57.375]).reshape(1,3,1,1),
            "heads.classifier.weight":torch.zeros(575,2048),
            "heads.bnneck.num_batches_tracked":torch.tensor(100)}
    result=models.canonical_fastreid_state(source)
    assert set(result)=={"heads.weight","heads.bottleneck.0.num_batches_tracked"}
    assert "pixel_mean" in source  # The loaded checkpoint itself remains unchanged.
    with pytest.raises(ValueError,match="normalization"):
        models.canonical_fastreid_state({**source,"pixel_std":torch.ones(3)})
    with pytest.raises(ValueError,match="Ambiguous"):
        models.canonical_fastreid_state({**source,"heads.weight":torch.zeros(575,2048)})


def test_vehicle_bot_keeps_pretrained_gem_and_strict_shapes():
    from training.resnet_ibn import ResNet50IBNBackbone
    from training.osnet import GeM
    base=ResNet50IBNBackbone(pooling="gem").state_dict()
    power=base.pop("global_pool.p")
    state={"backbone."+k:v for k,v in base.items()}
    state.update({"heads.bottleneck.0."+k:v for k,v in nn.BatchNorm1d(2048).state_dict().items()})
    state["heads.classifier.weight"]=torch.zeros(2,2048)
    state["heads.pool_layer.p"]=power.reshape(1)
    model=models.VehicleIBN(state,sbs=False).eval()
    assert isinstance(model.backbone.global_pool,GeM)
    with torch.no_grad(): assert model(torch.zeros(2,3,64,64)).shape==(2,2048)
    state["heads.unexpected.weight"]=torch.zeros(1)
    with pytest.raises(ValueError,match="Unexpected FastReID"):
        models.VehicleIBN(state,sbs=False)
