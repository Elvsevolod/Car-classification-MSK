"""Synthetic search/resume contracts; real public weights only for tiny architecture smoke."""
import copy
import hashlib
import socket

import nbformat
import numpy as np
from PIL import Image
import pytest
import torch

from training import transreid_model as v, transreid_night as n


@pytest.fixture(autouse=True)
def threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


def settings():
    return n.base.old.load_json(v.VARIANT / "configs/night_v1.json")


def test_fixed_grid_protocol_and_budgets():
    s = settings(); trials = n.trial_grid(s)
    assert len(trials) == len({t["id"] for t in trials}) == 24
    assert {t["architecture"] for t in trials} == {"global", "jpm"}
    assert {t["metric_loss"] for t in trials} == {"soft_triplet", "supcon"}
    assert s["screen_steps"] == 400 and s["final_steps"] == 1800
    assert 24*400+4*(1800-400) == 15200
    assert s["identities_per_batch"]*s["images_per_identity"] == 16
    assert not any(s[k] for k in ("SIE_CAMERA","SIE_VIEW","external_train","threshold_fit","original_outer_evaluation","promoted"))
    assert s["wall_time_limit"] is None and s["new_encoder_weight"] == .1


def test_official_weights_and_runtime_need_no_network(monkeypatch):
    if not v.WEIGHT_PATH.exists(): pytest.skip("Download official public weights for this architecture smoke")
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("Offline inference used network"))
    assert v.download_weights() == v.WEIGHT_PATH
    for architecture in ("global", "jpm"):
        torch.manual_seed(8)
        model = v.ReIDModel(4, architecture)
        assert model.initialization["loaded_tensors"] == 150
        assert not any("sie_embed" in k for k in model.state_dict())
        assert model.base.cam_num == model.base.view_num == 0
        images = torch.randn(4,3,256,256); labels = torch.tensor([0,0,1,1])
        before = model.base.patch_embed.proj.weight.detach().clone()
        for metric in ("soft_triplet", "supcon"):
            model.train().zero_grad(set_to_none=True)
            losses = v.losses(model, images, labels, {"metric_loss":metric}, settings())
            losses["loss"].backward()
            assert all(torch.isfinite(z) for z in losses.values())
            assert model.base.patch_embed.proj.weight.grad.norm() > 0
            assert all(head.weight.grad.norm() > 0 for head in model.heads)
            if architecture == "jpm": assert model.local_block.attn.qkv.weight.grad.norm() > 0
        torch.testing.assert_close(model.base.patch_embed.proj.weight, before, atol=0, rtol=0)
        model.eval()
        with torch.no_grad():
            baseline = model(images)
            separate = torch.cat([model(image[None]) for image in images])
            torch.testing.assert_close(separate, baseline, atol=2e-5, rtol=0)
            assert baseline.shape == (4,384 if architecture=="global" else 1920)
            torch.testing.assert_close(baseline.norm(dim=1), torch.ones(4), atol=2e-6, rtol=0)


def test_strict_pretrain_missing_keys_not_ignored(monkeypatch):
    if not v.WEIGHT_PATH.exists(): pytest.skip("Requires public weights")
    payload = torch.load(v.WEIGHT_PATH, map_location="cpu", weights_only=True)
    payload["model"].pop("norm.bias")
    monkeypatch.setattr(v.torch, "load", lambda *a, **k: payload)
    with pytest.raises(RuntimeError, match="Missing key"):
        v.ReIDModel(4)


def test_corrupted_weights_fail_before_loading(tmp_path):
    path = tmp_path / "bad.pth"; path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Missing/changed"):
        v.ReIDModel(4, weight_path=path)


def test_shuffle_never_uses_cls_and_preserves_all_patches():
    tokens = torch.arange(257).reshape(1,257,1).float()
    values = v.shuffle_patches(tokens).flatten().tolist()
    assert sorted(values) == list(range(1,257))
    assert values[:4] == [8,136,9,137]


@pytest.mark.parametrize("labels", [[0,0,0,0],[0,1,2,3]])
def test_metric_requires_both_positive_and_negative(labels):
    with pytest.raises(ValueError, match="P>=2"):
        v.soft_triplet(torch.randn(4,384), torch.tensor(labels))


def test_soft_triplet_matches_manual_batch_hard():
    x = torch.tensor([[0.,0.],[1.,0.],[4.,0.],[6.,0.]])
    y = torch.tensor([0,0,1,1])
    expected = torch.nn.functional.softplus(torch.tensor([1-4,1-3,2-3,2-5],dtype=torch.float32)).mean()
    torch.testing.assert_close(v.soft_triplet(x,y), expected, atol=1e-7, rtol=0)


def test_global_and_local_loss_weighting():
    class Model:
        def __call__(self, images):
            torch.manual_seed(6)
            return [torch.randn(4,2) for _ in range(5)], [torch.randn(4,8) for _ in range(5)]
    model = Model(); y=torch.tensor([0,0,1,1])
    scores, features = model(None)
    ce = [torch.nn.functional.cross_entropy(z,y,label_smoothing=.1) for z in scores]
    triplet = [v.soft_triplet(z,y) for z in features]
    result=v.losses(model,None,y,{"metric_loss":"soft_triplet"},settings())
    torch.testing.assert_close(result["ce"],.5*ce[0]+.5*torch.stack(ce[1:]).mean())
    torch.testing.assert_close(result["metric"],.5*triplet[0]+.5*torch.stack(triplet[1:]).mean())


def test_image_paths_jpeg_png_bbox_and_metadata(tmp_path):
    directory=tmp_path/"images"; directory.mkdir()
    entries=[]
    for i,suffix in enumerate(("jpg","JPEG","png","PNG")):
        Image.fromarray(np.random.default_rng(i).integers(0,256,(9,12,3),dtype=np.uint8)).save(directory/f"i{i}.{suffix}")
        entries.append({"image_id":f"i{i}","x":0,"y":0,"w":12,"h":9,"label":i})
    paths=v.image_paths(tmp_path,entries)
    before=v.Images(entries,paths)[0][0]
    other=[{**r,"vehicle_id":999,"camera_id":456,"time":"ignored"} for r in entries]
    torch.testing.assert_close(v.Images(other,paths)[0][0],before,atol=0,rtol=0)
    with pytest.raises(ValueError,match="outside"):
        v.Images([{**entries[0],"w":13}],paths)[0]
    Image.new("RGB",(12,9)).save(directory/"i0.png")
    with pytest.raises(ValueError,match="found 2"): v.image_paths(tmp_path,entries)
    with pytest.raises(ValueError,match="found 0"): v.image_paths(tmp_path,[{"image_id":"missing"}])


def test_device_must_not_fallback(monkeypatch):
    monkeypatch.setattr(torch.backends.mps,"is_available",lambda:False)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    for name in ("mps","cuda","auto"):
        with pytest.raises(ValueError,match="fallback"): v.device_for(name)


def test_fixed_horizon_lr_not_restarted_on_promotion():
    optimizer=torch.optim.AdamW([{"params":[torch.nn.Parameter(torch.ones(1))],"base_lr":1e-4}])
    s=settings()
    assert n.set_lr(optimizer,s,0)[0] == pytest.approx(1e-6)
    assert n.set_lr(optimizer,s,99)[0] == pytest.approx(1e-4)
    at_promotion=n.set_lr(optimizer,s,400)[0]
    assert 2e-6 < at_promotion < 1e-4
    assert n.set_lr(optimizer,s,1799)[0] == pytest.approx(2e-6)


class TinyModel(torch.nn.Module):
    def __init__(self,classes,architecture):
        super().__init__()
        self.architecture=architecture
        self.base=torch.nn.Linear(4,384)
        count=1 if architecture=="global" else 5
        self.necks=torch.nn.ModuleList(torch.nn.BatchNorm1d(384) for _ in range(count))
        self.heads=torch.nn.ModuleList(torch.nn.Linear(384,classes,bias=False) for _ in range(count))
        self.dimension=384*count

    def forward(self,x):
        f=self.base(x); features=[f*(1+i*.1) for i in range(len(self.heads))]
        if self.training: return [h(b(z)) for h,b,z in zip(self.heads,self.necks,features)],features
        return torch.nn.functional.normalize(torch.cat(features,dim=1),dim=1)


class TinyImages:
    def __init__(self,rows,paths,train=False): self.rows,self.train=rows,train
    def __getitem__(self,index):
        row=self.rows[index]
        x=torch.tensor([row["label"],index/10,1.,.5])
        return x+(torch.rand(4)*.1 if self.train else 0), row["label"]


def random_features(rows,dimension):
    values=[]
    for row in rows:
        seed=int(hashlib.sha256(row["image_id"].encode()).hexdigest()[:8],16)
        values.append(np.random.default_rng(seed).normal(size=dimension))
    return n.base.normalize(np.asarray(values,dtype=np.float32))


@pytest.fixture
def context(tmp_path,monkeypatch):
    s=settings()
    s.update(screen_steps=2,final_steps=6,checkpoints=[2,4,6],save_interval=1,log_interval=1,warmup_steps=1)
    s["grid"]={"architectures":["global","jpm"],"encoder_lr":[1e-3],"weight_decay":[0.],"metric_loss":["soft_triplet","supcon"]}
    target=[{"image_id":f"t{i}","label":i//2,"vehicle_id":20+i//2,"camera_id":i%2} for i in range(4)]
    queries=[{"image_id":f"q{i}","vehicle_id":i,"camera_id":0} for i in range(4)]
    gallery=[{"image_id":f"g{i}_{j}","vehicle_id":i,"camera_id":1} for i in range(3) for j in range(5)]
    draw={"query_ids":[r["image_id"] for r in queries],"gallery_ids":[r["image_id"] for r in gallery]}
    score={"rows":queries+gallery,"manifest":{"inner":{"validation":[0,1,2,3]},"draws":{"regular_1":draw}}}
    rows=n.base.development_rows(score)
    reference=tmp_path/"reference"; reference.mkdir()
    r1=random_features(rows,512); np.save(reference/"features.npy",r1)
    n.base.write_json(reference/"metrics.json",n.base.score_features(score,r1,rows))
    c={"settings":s,"trials":n.trial_grid(s),"signature":"synthetic","output":tmp_path/"run",
       "device":torch.device("cpu"),"target":target,"rows":rows,"paths":{},"score_context":score,
       "schedule":[[0,1,2,3]]*6,"manifest":{"draws":{"regular_1":draw},"reference_directory":str(reference)}}
    def factory(c,trial):
        n.base.set_seed(8)
        return TinyModel(2,trial["architecture"])
    monkeypatch.setattr(n,"new_model",factory)
    monkeypatch.setattr(v,"Images",TinyImages)
    monkeypatch.setattr(n,"check_inputs",lambda c:None)
    return c


@pytest.mark.parametrize("architecture",["global","jpm"])
def test_exact_cpu_resume_and_saved_weights_not_deleted(context,architecture):
    trial=next(t for t in context["trials"] if t["architecture"]==architecture)
    full={**context,"output":context["output"]/"full"}
    resumed={**context,"output":context["output"]/"resumed"}
    n.train_until(full,trial,6)
    n.train_until(resumed,trial,2)
    n.train_until(resumed,trial,6)
    a,b=[torch.load(n.base.resume_path(c["output"]/"training"/trial["id"]),weights_only=True) for c in (full,resumed)]
    assert a["history"]==b["history"]
    for key in a["model"]: torch.testing.assert_close(a["model"][key],b["model"][key],atol=0,rtol=0)
    for key,state in a["optimizer"]["state"].items():
        for name,value in state.items():
            torch.testing.assert_close(value,b["optimizer"]["state"][key][name],atol=0,rtol=0)
    directory=resumed["output"]/"training"/trial["id"]
    assert len(list(directory.glob("resume_*.pt")))==1
    assert len(list(directory.glob("step_*.pt")))==3
    assert n.train_until(resumed,trial,6)["step"]==6


def test_corrupted_resume_fails(context):
    trial=context["trials"][0]; n.train_until(context,trial,2)
    path=n.base.resume_path(context["output"]/"training"/trial["id"]); path.write_bytes(b"changed")
    with pytest.raises(ValueError,match="Protected file changed"): n.train_until(context,trial,6)


def test_finalists_keep_single_and_complement_even_if_different():
    trials=n.trial_grid(settings()); reports={}
    for i,t in enumerate(trials):
        reports[t["id"]]={"single":{"mean_map":float(i)},"mixture":{"mean_map":float(-i)}}
    chosen=n.choose_finalists(trials,reports)
    assert chosen==[trials[11]["id"],trials[0]["id"],trials[23]["id"],trials[12]["id"]]
    for r in reports.values(): r["single"]["mean_map"]=r["mixture"]["mean_map"]=.8
    assert n.choose_finalists(trials,reports)==[trials[0]["id"],trials[1]["id"],trials[12]["id"],trials[13]["id"]]


def test_full_night_controller_and_cached_run(context,monkeypatch):
    calls=[]
    def encode(model,rows,*args,**kwargs):
        calls.append(len(rows))
        return random_features(rows,model.dimension)
    monkeypatch.setattr(v,"encode",encode)
    result=n.run(context)
    assert result["status"]=="complete"
    assert len(result["finalists"])==4 and len(result["leaderboard"])==14
    assert result["promoted"] is result["original_outer_evaluation"] is result["threshold_fit"] is False
    count=len(calls)
    assert n.run(context)==result and len(calls)==count
    path=context["output"]/"tasks"/"stock_global"/"features.npy"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError): n.run(context)


def test_trial_numerical_failure_does_not_stop_other_trials(context,monkeypatch):
    monkeypatch.setattr(v,"encode",lambda model,rows,*a,**k:random_features(rows,model.dimension))
    original=n.train_until
    def train(c,t,horizon):
        if t["id"]==c["trials"][0]["id"]: raise FloatingPointError("synthetic instability")
        return original(c,t,horizon)
    monkeypatch.setattr(n,"train_until",train)
    result=n.run(context)
    assert len(result["failures"])==1 and len(result["finalists"])==3


def test_unknown_runtime_error_is_not_silenced(context,monkeypatch):
    monkeypatch.setattr(v,"encode",lambda model,rows,*a,**k:random_features(rows,model.dimension))
    monkeypatch.setattr(n,"train_until",lambda *a: (_ for _ in ()).throw(RuntimeError("unexpected programmer bug")))
    with pytest.raises(RuntimeError,match="programmer bug"): n.run(context)


def test_final_failure_resume_does_not_erase_screen_finalists(context,monkeypatch):
    monkeypatch.setattr(v,"encode",lambda model,rows,*a,**k:random_features(rows,model.dimension))
    original=n.train_until
    def train(c,t,horizon):
        if t["id"]==c["trials"][0]["id"] and horizon==6:
            raise FloatingPointError("finalist-only instability")
        return original(c,t,horizon)
    monkeypatch.setattr(n,"train_until",train)
    probe=n.final_probe
    monkeypatch.setattr(n,"final_probe",lambda *a:(_ for _ in ()).throw(InterruptedError("interrupted before completion")))
    with pytest.raises(InterruptedError): n.run(context)
    selected=n.base.old.load_json(context["output"]/"finalists.json")["selected"]
    monkeypatch.setattr(n,"final_probe",probe)
    result=n.run(context)
    assert result["finalists"]==selected and len(selected)==4
    assert context["trials"][0]["id"] in result["failures"]


def test_requested_device_smoke_discards_models(context):
    result=n.runtime_smoke(context)
    assert result["status"]=="passed" and len(result["updates"])==4
    assert not (context["output"]/"training").exists()
    assert n.runtime_smoke(context)==result


def test_successful_final_checkpoint_survives_later_eval_failure_and_restart(context,monkeypatch):
    monkeypatch.setattr(v,"encode",lambda model,rows,*a,**k:random_features(rows,model.dimension))
    original=n.evaluate
    def evaluate(c,t,entry=None,step=0):
        if t["id"]==c["trials"][0]["id"] and step==6:
            raise FloatingPointError("last checkpoint invalid")
        return original(c,t,entry,step)
    monkeypatch.setattr(n,"evaluate",evaluate)
    probe=n.final_probe
    monkeypatch.setattr(n,"final_probe",lambda *a:(_ for _ in ()).throw(InterruptedError("stop")))
    with pytest.raises(InterruptedError): n.run(context)
    monkeypatch.setattr(n,"final_probe",probe)
    result=n.run(context)
    trial=context["trials"][0]["id"]
    assert {r["step"] for r in result["leaderboard"] if r["trial_id"]==trial}=={2,4}
    assert len(result["leaderboard"])==13 and trial in result["failures"]


def test_disk_guard_never_deletes_existing_outputs(tmp_path,monkeypatch):
    target=tmp_path/"runs"/"night"; target.mkdir(parents=True)
    keep=target/"old.txt"; keep.write_text("preserve")
    monkeypatch.setattr(n.shutil,"disk_usage",lambda p:type("Usage",(),{"free":1})())
    with pytest.raises(RuntimeError,match="Free space yourself"): n.disk_guard(target,settings())
    assert keep.read_text()=="preserve"


def test_mixture_cosine_and_query_independence():
    rows=[{"image_id":str(i)} for i in range(50)]
    a=random_features(rows,512); b=random_features(rows,1920)
    values=n.mix(a,b)
    x,y,z=(v.astype(np.float64) for v in (a,b,values))
    np.testing.assert_allclose(z@z.T,.9*(x@x.T)+.1*(y@y.T),atol=6e-7,rtol=0)
    original=n.base.policy.rank_vectors(values[:12],values[12:],"less_graph")
    for indices in (np.arange(12)[::-1],np.array([3])):
        result=n.base.policy.rank_vectors(values[indices],values[12:],"less_graph")
        for key in ("raw_order","order"): np.testing.assert_array_equal(result[key],original[key][indices])


def test_notebook_has_no_timeout_and_compiles():
    notebook=nbformat.read(v.VARIANT/"train_transreid_night.ipynb",as_version=4)
    nbformat.validate(notebook)
    code=[c.source for c in notebook.cells if c.cell_type=="code"]
    for i,text in enumerate(code): compile(text,f"cell{i}","exec")
    assert "ORT_DISABLE_TELEMETRY" in code[0] and "DEVICE = 'mps'" in code[0]
    assert "night.run(context)" in "\n".join(code)
    assert "timeout=" not in "\n".join(code)
