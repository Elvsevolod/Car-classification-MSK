"""Default checks are cheap; RUN_REID_MPS_SMOKE=1 opts into real Apple-GPU steps."""
import os
os.environ.setdefault("ORT_DISABLE_TELEMETRY","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","0")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH","0")
import inspect
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from training import research_mac as mac, research_gpu as gpu, research_io as io
from training import research_models as models, research_training as training, research_runtime as runtime


def test_mac_default_and_same_scientific_protocol():
    assert inspect.signature(gpu.run).parameters["device"].default=="mps"
    windows=io.read(gpu.VARIANT/"configs/rtx4060_v1.json")
    apple=io.read(gpu.VARIANT/"configs/mac_m4_v1.json")
    assert {k:v for k,v in windows.items() if k not in {"experiment","device"}} == {
        k:v for k,v in apple.items() if k not in {"experiment","device","mps_memory_fraction"}}
    assert 0 < apple["mps_memory_fraction"] <= 1


def test_reject_rosetta_and_unavailable_mps(monkeypatch,tmp_path):
    monkeypatch.setattr(mac.platform,"system",lambda:"Darwin")
    monkeypatch.setattr(mac.platform,"machine",lambda:"x86_64")
    with pytest.raises(ValueError,match="Rosetta"): mac.preflight(tmp_path)
    monkeypatch.setattr(mac.platform,"machine",lambda:"arm64")
    monkeypatch.setattr(torch.backends.mps,"is_available",lambda:False)
    with pytest.raises(ValueError,match="MPS unavailable"): mac.preflight(tmp_path)


def test_reject_fallback_and_unlimited_memory(monkeypatch,tmp_path):
    monkeypatch.setattr(mac.platform,"system",lambda:"Darwin")
    monkeypatch.setattr(mac.platform,"machine",lambda:"arm64")
    monkeypatch.setattr(torch.backends.mps,"is_available",lambda:True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK","1")
    with pytest.raises(ValueError,match="restart the kernel"): mac.preflight(tmp_path)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK","0")
    for fraction in (0,-1,1.5):
        with pytest.raises(ValueError,match="fraction"): mac.preflight(tmp_path,fraction)


def test_launch_scripts_are_valid_and_do_not_use_copied_venv():
    for name in ("setup_mac.sh","Start_v41_Mac.command"):
        path=gpu.VARIANT/name
        if shutil.which("bash"):
            subprocess.run(["bash","-n",str(path)],check=True)
        text=path.read_text()
        assert "/Users/" not in text and '"$PROJECT_ROOT/.venv-v41-m4' in text
        assert "rm -" not in text and "--clear" not in text
    setup=(gpu.VARIANT/"setup_mac.sh").read_text()
    assert ".v41-location" in setup and "--prefix" in setup


def test_mps_memory_is_a_sample_not_cuda_peak(monkeypatch):
    monkeypatch.setattr(torch.mps,"current_allocated_memory",lambda:10)
    monkeypatch.setattr(torch.mps,"driver_allocated_memory",lambda:20)
    assert mac.memory_sample()=={"mps_allocated_bytes":10,"mps_driver_bytes":20}


@pytest.mark.skipif(os.environ.get("RUN_REID_MPS_SMOKE")!="1",reason="opt-in real MPS/data smoke")
@pytest.mark.parametrize("kind",["trans_supcon","trans_soft_triplet","nive_shared",
    "nive_target_updates_only","nive_domain_specific","dino"])
def test_real_mps_forward_backward(kind):
    assert torch.backends.mps.is_available(), "This explicit smoke must run on the Apple GPU"
    mac.preflight(io.ROOT)
    root=io.ROOT/"research_transfer/v41_inputs"
    manifest=io.read(root/"inputs.json")
    c={"device":torch.device("mps"),"inputs":root,"manifest":manifest,
       "external":[{"label":0},{"label":1}],"train":[{"label":0},{"label":1}],
       "asset_cache":gpu.VARIANT/"weights"}
    spec={"seed":20260915,"lr":1e-4,"weight_decay":.01,"clip":5.,"loss":"supcon"}
    if kind.startswith("trans_"):
        spec.update(family="transreid",loss=kind.removeprefix("trans_"))
    elif kind.startswith("nive_"):
        spec.update(family="nive",aux="nive",bn_mode=kind.removeprefix("nive_"))
    else:
        assert (c["asset_cache"]/"dinov2_vits14.pth").is_file(), "Copy the existing official DINO-S cache"
        spec.update(family="external",model="dinov2_vits14",pool="cls",adapt=True,train_mode="last4",loss="soft_triplet")
    model=gpu.make_model(c,spec)
    try:
        size=280 if kind=="dino" else 256
        x=torch.randn(4,3,size,size,device="mps")
        labels=torch.tensor([0,0,1,1],device="mps")
        optimizer=training.optimizer_for(model,spec)
        if spec["family"]=="nive":
            auxiliary=training.domain_loss(model,[x,x,labels],manifest["config"],"aux",spec["bn_mode"])
            (.025*auxiliary["loss"]).backward()
            values=training.domain_loss(model,[x,x,labels],manifest["config"],"main",spec["bn_mode"])
        else:
            values=training.metric_loss(model,x,labels,spec)
        values["loss"].backward()
        assert torch.isfinite(values["loss"])
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step(); torch.mps.synchronize()
        model.eval()
        with torch.no_grad():
            features=model.embedding(x) if hasattr(model,"embedding") else model(x)
        assert features.shape[0]==4 and torch.isfinite(features).all()
        print(kind,mac.memory_sample())
        del optimizer,values,features,x,labels
    finally:
        del model
        runtime.cleanup()
