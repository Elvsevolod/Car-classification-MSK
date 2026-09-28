"""v41: resumable Windows/CUDA queues; one real experiment at a time on 8 GB."""
import argparse
from itertools import product, zip_longest
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn

from training import research_io as io, research_models as models, research_runtime as runtime
from training import research_scoring as scoring, research_training as training
from training.transreid_model import ReIDModel, WEIGHT_NAME

VARIANT = io.ROOT / "OSNet-AIN-x1.0/variant_41_windows_queues"


def trans_grid():
    return [{"id":f"T{i:02d}_p{p}k{k}_{loss}_lr{lr:g}","family":"transreid","lr":lr,
             "p":p,"k":k,"loss":loss,"size":256,"weight_decay":.01,"clip":5.,"seed":20260915}
            for i,(lr,(p,k),loss) in enumerate(product((3e-5,1e-4,3e-4),((8,2),(16,4)),
                                                       ("supcon","soft_triplet")),1)]


def nive_grid(config):
    common = {"family":"nive","lr":config["encoder_lr"],"p":16,"k":2,"loss":"supcon","size":256,
              "weight_decay":config["weight_decay"],"seed":20260915,"joint_steps":1600,"tail_steps":600}
    result = [{**common,"id":"N_target_only","aux":"none","bn_mode":"shared","alpha":0.}]
    for mode,alpha in product(("shared","target_updates_only","domain_specific"),(.005,.025,.1)):
        for aux in ("nive","target"):
            result.append({**common,"id":f"N_{aux}_{mode}_a{alpha:g}","aux":aux,"bn_mode":mode,"alpha":alpha})
    return result


def frozen_grid():
    return [{"id":f"F_{name}_{pool}","family":"external","model":name,"pool":pool,"adapt":False,
             "size":280 if name.startswith("dino") else 256,
             "interpolation":"bicubic" if name.startswith("dino") else "bilinear","seed":20260915}
            for name in models.ASSETS for pool in (("cls","patch","concat") if name.startswith("dino") else ("cls",))]


def adaptation_grid(frozen):
    """Pool/source choice uses only frozen PRIMARY scores; no outer scores are read."""
    result = []
    winners = []
    for name in ("dinov2_vits14","dinov2_vitb14"):
        available = [r for r in frozen if r["spec"]["model"] == name]
        if available: winners.append(max(available,key=lambda r:r["metrics"]["selection_score"]))
    vehicle = [r for r in frozen if r["spec"]["model"].startswith("r50_") and r["spec"]["model"] != "r50_imagenet"]
    winners += sorted(vehicle,key=lambda r:r["metrics"]["selection_score"],reverse=True)[:2]
    for winner in winners:
        original = winner["spec"]
        dino = original["model"].startswith("dino")
        for lr in ((3e-6,1e-5,3e-5) if dino else (1e-5,3e-5,1e-4)):
            result.append({**original,"id":f"A_{original['model']}_{original['pool']}_lr{lr:g}",
                           "adapt":True,"lr":lr,"p":8,"k":2,"loss":"soft_triplet", "clip":5.,
                           "weight_decay":.0005,"train_mode":"last4" if dino else "last_stage"})
        if dino:
            result.append({**original,"id":f"H_{original['model']}_{original['pool']}","adapt":True,
                           "lr":1e-5,"p":8,"k":2,"loss":"soft_triplet","clip":5.,
                           "weight_decay":.0005,"train_mode":"head_only"})
    return result


def make_model(c,spec):
    training.seed_all(spec["seed"])
    classes = len(c["manifest"]["train_ids"])
    if spec["family"] == "transreid":
        model = ReIDModel(classes,"global",weight_path=c["inputs"] / "weights" / WEIGHT_NAME)
        models.checkpoint_blocks(model.base.blocks)
    elif spec["family"] == "nive":
        model = models.load_osnet(c["inputs"],c["manifest"])
        classes_aux = len({r["label"] for r in (c["external"] if spec["aux"] == "nive" else c["train"])})
        training.seed_all(spec["seed"]+77)
        model.aux_head = nn.Linear(512,classes_aux,bias=False)
        nn.init.normal_(model.aux_head.weight,std=.01)
        if spec["bn_mode"] == "domain_specific": models.split_bn(model)
    else:
        model = models.external_model(spec["model"],spec["pool"],classes,c["asset_cache"],spec["adapt"])
        model.base.requires_grad_(False)
        trainable = []
        if spec["adapt"] and spec["train_mode"] == "last4":
            trainable = list(model.base.blocks[-4:])+[model.base.norm]
            models.checkpoint_blocks(model.base.blocks[-4:])
        elif spec["adapt"] and spec["train_mode"] == "last_stage":
            trainable = [model.base.backbone.layer4,model.base.neck]
        for block in trainable: block.requires_grad_(True)
        # Plain list, not registered again: state_dict has no duplicate tensor paths.
        model.trainable_blocks = trainable
    return model.to(c["device"])


def evaluate_trial(c,spec,rung):
    directory = c["output"] / "trials" / spec["id"]
    stage = directory / f"rung_{rung}"
    signature = io.digest({"run":c["signature"],"spec":spec,"rung":rung})
    if io.completed(stage,signature):
        print(f"RESUME: verified {spec['id']} / {rung}",flush=True)
        saved = io.read(stage / "result.json")
        if saved["training"]:
            io.verify(c["output"], {saved["training"]["path_from_run"]:saved["training"]["sha256"]})
        return saved
    model = make_model(c,spec)
    start = time.perf_counter()
    trained = None
    try:
        if spec["family"] == "nive":
            steps = spec["joint_steps"]+spec["tail_steps"]
            trained = training.train(model,c,spec,steps,steps,directory)
        elif spec["family"] == "transreid" or spec.get("adapt"):
            horizon_passes = 90 if spec["family"] == "transreid" else 30
            steps = math.ceil(rung*len(c["train"])/(spec["p"]*spec["k"]))
            horizon = math.ceil(horizon_passes*len(c["train"])/(spec["p"]*spec["k"]))
            trained = training.train(model,c,spec,steps,horizon,directory)
        stage.mkdir(parents=True,exist_ok=True)
        features = models.encode(model,c["inputs"],c["rows"],c["device"],spec["size"],8,
                                 spec.get("interpolation","bilinear"))
        np.save(stage / "features.npy",features)
        metrics = scoring.score_features(features,c["rows"],c["manifest"]["draws"],c["control"])
        report = {"spec":spec,"rung":rung,"metrics":metrics,"training":trained,
                  "elapsed_seconds":time.perf_counter()-start,"signature":signature,"status":"complete"}
        if trained:
            report["training"]["path_from_run"] = (directory / trained["checkpoint"]).relative_to(c["output"]).as_posix()
        io.write(stage / "result.json",report)
        io.write(stage / "order.json",[r["image_id"] for r in c["rows"]])
        io.finish(stage,signature)
        print(f"DONE {spec['id']} / {rung}: {metrics['means']}",flush=True)
        return report
    finally:
        del model; runtime.cleanup()


def attempt(c,spec,rung):
    name = f"{spec['id']}__{rung}"
    entry = {"trial":spec["id"],"family":spec["family"],"rung":rung}
    try:
        value = evaluate_trial(c,spec,rung)
        entry.update(status="complete",score=value["metrics"]["selection_score"],delta=value["metrics"]["delta_system"])
    except Exception as error:
        # Includes OOM: record exact failed configuration; NEVER shrink P/K or loosen guards silently.
        value = None
        entry.update(status="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        print(f"FAILED {name}: {type(error).__name__}: {error}. Other queues continue.",flush=True)
        runtime.cleanup()
    io.write(c["output"] / "status" / f"{name}.json",entry)
    return value


def summarize(c, interrupted=False):
    statuses = [io.read(p) for p in sorted((c["output"] / "status").glob("*.json"))]
    reports = [io.read(p) for p in sorted((c["output"] / "trials").glob("*/rung_*/result.json"))
               if (p.parent / "complete.json").exists()]
    leaders = sorted(reports,key=lambda r:r["metrics"]["selection_score"],reverse=True)
    summary = {"status":"interrupted" if interrupted else "completed_with_failures" if any(
        s["status"] != "complete" for s in statuses) else "complete", "signature":c["signature"],
        "baseline":c["manifest"]["baseline"],"primary_only":True,"promoted":False,"statuses":statuses,
        "leaderboard":[{"trial":r["spec"]["id"],"rung":r["rung"],"means":r["metrics"]["means"],
                        "delta":r["metrics"]["delta_system"],"training":r["training"]} for r in leaders]}
    io.write(c["output"] / "results.json",summary)
    lines = ["# v41 Windows / RTX 4060 research", "", f"Status: {summary['status']}", "",
             c["manifest"]["baseline"],"", "Primary development only. No outer evaluation, threshold fit or promotion.",
             "F1/TNR not optimized; final release requires a separately frozen full-system comparison.","",
             "| Trial | Rung | Single graph | System G2 | Delta |","|---|---:|---:|---:|---:|"]
    lines += [f"| {r['trial']} | {r['rung']} | {r['means']['graph']:.6f} | {r['means']['G2']:.6f} | {r['delta']:+.6f} |"
              for r in summary["leaderboard"]]
    lines += ["", "## Failed trials", ""]+[f"- {s['trial']} / {s['rung']}: {s['error']}" for s in statuses if s["status"]!="complete"]
    (c["output"] / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


def run(inputs,run_name="rtx4060_v1",device="cuda"):
    if not run_name.replace("_","").replace("-","").isalnum(): raise ValueError("Simple RUN_NAME required")
    if device != "cuda": raise ValueError("v41 requires CUDA; use v40 for Mac. No CPU fallback.")
    settings = io.read(VARIANT / "configs/rtx4060_v1.json")
    output = VARIANT / "runs" / run_name
    with io.lock(output):
        c = runtime.prepare(inputs,output,device,settings)
        c["asset_cache"] = VARIANT / "weights"
        c["control"] = runtime.baselines(c)  # Fresh control on THIS runtime, not copied Mac measurements.
        trans, nive, frozen = trans_grid(),nive_grid(c["manifest"]["config"]),frozen_grid()
        io.freeze(output / "trial_plan.json", {"transreid":trans,"nive":nive,"frozen":frozen,
                  "transreid_rungs":[10,30,90],"finalists":4,"adaptation":"best pool per DINO, best 2 vehicle IBN; LR grid"})
        completed_frozen, first_trans = [],[]
        try:
            # Round-robin FIRST wave gives all three hypotheses a chance before long continuations.
            for group in zip_longest(trans,nive,frozen):
                for spec in group:
                    if spec is None: continue
                    rung = 10 if spec["family"]=="transreid" else 2200 if spec["family"]=="nive" else 0
                    value = attempt(c,spec,rung)
                    if value and spec["family"]=="external": completed_frozen.append(value)
                    if value and spec["family"]=="transreid": first_trans.append(value)
                    summarize(c)
            selection_path = output / "adaptation_selection.json"
            if selection_path.exists():
                adapt = io.read(selection_path)["specs"]
            else:
                adapt = adaptation_grid(completed_frozen)
                io.freeze(selection_path, {"specs":adapt,"frozen_inputs":{
                    r["spec"]["id"]:r["metrics"]["selection_score"] for r in completed_frozen}})
            adapted, longer = [],[]
            for group in zip_longest([r["spec"] for r in first_trans],adapt):
                for spec in group:
                    if spec is None: continue
                    rung = 30 if spec["family"]=="transreid" else 10
                    value = attempt(c,spec,rung)
                    if value: (longer if spec["family"]=="transreid" else adapted).append(value)
                    summarize(c)
            finalists = sorted(longer,key=lambda r:r["metrics"]["selection_score"],reverse=True)[:4]
            external_best = []
            for prefix in ("dinov2","r50"):
                group = [r for r in adapted if r["spec"]["model"].startswith(prefix)]
                external_best += sorted(group,key=lambda r:r["metrics"]["selection_score"],reverse=True)[:2]
            final_path = output / "continuation_selection.json"
            if final_path.exists():
                saved = io.read(final_path)
                finalists = [r for r in longer if r["spec"]["id"] in saved["transreid"]]
                external_best = [r for r in adapted if r["spec"]["id"] in saved["external"]]
            else:
                io.freeze(final_path, {"transreid":[r["spec"]["id"] for r in finalists],
                                       "external":[r["spec"]["id"] for r in external_best]})
            for r in external_best+finalists:
                attempt(c,r["spec"],90 if r["spec"]["family"]=="transreid" else 30)
                summarize(c)
        except KeyboardInterrupt:
            summarize(c,interrupted=True)
            raise
        finally:
            if Path(c["asset_cache"]).exists():
                io.write(output / "external_sources.json",{p.name:io.read(p) for p in Path(c["asset_cache"]).glob("*.json")})
        summary = summarize(c)
    io.archive(output,VARIANT / f"{run_name}_analysis.zip",light=True)
    io.archive(output,VARIANT / f"{run_name}_full_results.zip")
    return summary


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True); p.add_argument("--run-name",default="rtx4060_v1")
    a=p.parse_args(); run(a.inputs,a.run_name)
