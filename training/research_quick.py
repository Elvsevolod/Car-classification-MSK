"""v40: cache-only G1/G2 and one-pass BN recalibration, no optimizer updates."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from training import research_io as io, research_models as models, research_runtime as runtime
from training import research_scoring as scoring, retrieval_policy as policy, map_inference
from training.research_training import save_torch
from backend.core import read_rows

VARIANT = io.ROOT / "OSNet-AIN-x1.0/variant_40_quick_diagnostics"


def fusion_grid():
    return [{"name":"V25_control", "expert":"T12", "mode":"G1", "weight":0.}] + [
        {"name":f"{expert}_{mode}_w{round(weight*100):02d}", "expert":expert, "mode":mode, "weight":weight}
        for expert in ("N1", "T12") for mode in ("G1", "G2") for weight in (.02,.05,.10,.20)]


def cache_input(plan, split, expert):
    io.verify(io.ROOT, {"dataset/train.csv":plan["train_csv_sha256"]})
    source = plan["cache_sources"][split][expert]
    io.verify(io.ROOT, source["files"])
    bank = np.load(io.child(io.ROOT, source["features"]), allow_pickle=False)
    order = io.read(io.child(io.ROOT, source["order"]))
    if order["sha256"] != io.sha(io.child(io.ROOT, source["features"])) or len(order["ids"]) != len(bank):
        raise ValueError("Cache ID order/hash mismatch")
    n = order["query_count"]
    by_id = {r["image_id"]:r for r in read_rows(io.ROOT / "dataset/train.csv")}
    query, gallery = ([by_id[i] for i in part] for part in (order["ids"][:n], order["ids"][n:]))
    dimension, start = (3072,2560) if expert == "N1" else (2432,2048)
    if bank.shape != (len(query)+len(gallery), dimension): raise ValueError("Wrong expert bank layout")
    return query, gallery, bank[:,:2048], bank[:,start:]


def fusion_stage(plan, output, signature, split, spec):
    folder = output / "fusion" / f"{split}_{spec['name']}"
    if io.completed(folder, signature): return io.read(folder / "result.json")
    query, gallery, baseline, expert = cache_input(plan, split, spec["expert"])
    n = len(query)
    raw, jac = scoring.FixedGraph(baseline[n:]).components(baseline[:n])
    zero = np.argsort(scoring.fuse_distances(raw,jac,expert[:n],expert[n:],spec["mode"],0),axis=1,kind="stable")
    original = map_inference.rank(baseline[:n], baseline[n:])
    if not np.array_equal(zero, original["order"]): raise ValueError("Zero expert weight changed baseline order")
    distances = scoring.fuse_distances(raw,jac,expert[:n],expert[n:],spec["mode"],spec["weight"])
    ranked = {**original, "order":np.argsort(distances,axis=1,kind="stable")}
    # Explicit permutation/subset independence, including exact top-10.
    indices = np.arange(n-1,-1,-1)[:min(17,n)]
    r2,j2 = scoring.FixedGraph(baseline[n:]).components(baseline[indices])
    d2 = scoring.fuse_distances(r2,j2,expert[indices],expert[n:],spec["mode"],spec["weight"])
    if not np.array_equal(np.argsort(d2,axis=1,kind="stable")[:,:10],ranked["order"][indices,:10]):
        raise ValueError("Query permutation/subset changed decisions")
    before = policy.predictions(query,gallery,original,plan["threshold"],"raw_top1")[1]
    after = policy.predictions(query,gallery,ranked,plan["threshold"],"raw_top1")[1]
    if before != after: raise ValueError("Candidate policy changed")
    report = {"spec":spec, "split":split, **scoring.ranking_report(query,gallery,ranked["order"]),
              "candidates":policy.evaluate(query,gallery,ranked,plan["threshold"],"raw_top1")["candidates"],
              "candidate_unchanged":True, "query_independence":True}
    folder.mkdir(parents=True,exist_ok=True)
    policy.export_csv(folder / "export", query,gallery,ranked,plan["threshold"],"raw_top1")
    np.save(folder / "export/embeddings.npy",np.concatenate([baseline,expert],axis=1))
    io.write(folder / "export/embedding_order.json", {"ids":[r["image_id"] for r in query+gallery],
        "query_count":n,"layout":{"v25":[0,2048],"expert":[2048,2048+expert.shape[1]]},
        "spec":spec,"threshold":plan["threshold"],"note":"real unit blocks, G1/G2 distances; not one joint cosine"})
    io.write(folder / "result.json",report); io.finish(folder,signature)
    print(f"{split} {spec['name']}: mAP={report['metrics']['mAP@10']:.6f}",flush=True)
    return report


def run_fusion(plan, output):
    output = Path(output)
    frozen = {"plan":plan,"grid":fusion_grid(),"sources":io.source_hashes(),
              "scope":"previously observed calibration/validation; development, NOT independent hidden test"}
    io.freeze(output / "fusion_manifest.json",frozen)
    signature = io.digest(frozen)
    scores = [fusion_stage(plan,output,signature,"calibration",spec) for spec in fusion_grid()]
    winner = max(scores,key=lambda r:r["metrics"]["mAP@10"])["spec"]
    io.freeze(output / "fusion_selection.json", {"winner":winner,"calibration_scores":{
        r["spec"]["name"]:r["metrics"]["mAP@10"] for r in scores},"signature":signature})
    selected = [fusion_grid()[0]] + ([winner] if winner["weight"] else [])
    validation = [fusion_stage(plan,output,signature,"validation",spec) for spec in selected]
    io.write(output / "fusion_results.json", {"calibration":scores,"validation":validation,"promoted":False})
    return validation


def run_bn(inputs, output, device="mps"):
    pin = io.read(io.ROOT / "OSNet-AIN-x1.0/variant_41_windows_queues/INPUT_PACKAGE.json")
    c = runtime.prepare(inputs, Path(output)/"bn",device,{"experiment":"v40_BN","passes":1,"batch":32,
                        "input_sha256":pin["inputs_manifest_sha256"],
                        "reset_stats":True,"momentum":"cumulative","data":"primary TRAIN only"})
    control = runtime.baselines(c)
    reports = {}
    for name in ("R1_20260915","N1"):
        for recalibrate in (False,True):
            tag = name + ("_recalibrated" if recalibrate else "_unchanged")
            folder = c["output"] / tag
            if io.completed(folder,c["signature"]):
                reports[tag] = io.read(folder / "metrics.json"); continue
            model = models.load_osnet(c["inputs"],c["manifest"],name).to(c["device"])
            parameters = {k:v.detach().cpu().clone() for k,v in model.named_parameters()}
            if recalibrate: models.recalibrate_bn(model,c["inputs"],c["train"],c["device"])
            if any(not torch.equal(v.detach().cpu(),parameters[k]) for k,v in model.named_parameters()):
                raise ValueError("BN recalibration modified a learned parameter")
            values = models.encode(model,c["inputs"],c["rows"],c["device"])
            folder.mkdir(parents=True,exist_ok=True)
            if tag == "R1_20260915_unchanged" and not np.allclose(values,np.load(c["output"] / "baseline/R1_20260915.npy"),atol=1e-6,rtol=0):
                raise ValueError("Unchanged R1 failed control replay")
            np.save(folder / "features.npy",values)
            reports[tag] = scoring.score_features(values,c["rows"],c["manifest"]["draws"],control)
            io.write(folder / "metrics.json",reports[tag])
            save_torch(folder / "model.pt",{"model":model.state_dict(),"source":c["manifest"]["models"][name],
                                            "bn_recalibrated":recalibrate,"train_ids":c["manifest"]["train_ids"]})
            io.finish(folder,c["signature"])
            print(f"BN {tag}: {reports[tag]['means']}",flush=True)
            del model; runtime.cleanup()
    io.write(c["output"] / "results.json",reports)
    return reports


def run(inputs,run_name="quick_v1",device="mps"):
    if not run_name.replace("_","").replace("-","").isalnum(): raise ValueError("Simple RUN_NAME required")
    output = VARIANT / "runs" / run_name
    started = time.perf_counter()
    with io.lock(output):
        plan = io.read(VARIANT / "configs/quick_v1.json")
        validation = run_fusion(plan,output)
        bn = run_bn(inputs,output,device)
        report = {"status":"complete","elapsed_seconds":time.perf_counter()-started,
                  "fusion_validation":{r["spec"]["name"]:r["metrics"] for r in validation},
                  "bn_primary":{k:v["means"] for k,v in bn.items()},"optimizer_updates":0,"promoted":False}
        io.write(output / "results.json",report)
        (output / "REPORT.md").write_text("# v40 — быстрые проверки\n\n"+
            "Рабочий v25 не изменён. Fusion: наблюдавшиеся development splits. BN: только primary train.\n\n"+
            "\n".join(f"- {name}: mAP@10 {m['mAP@10']:.6f}" for name,m in report["fusion_validation"].items())+
            "\n\nBN standalone/system:\n"+"\n".join(f"- {n}: {m}" for n,m in report["bn_primary"].items())+"\n",encoding="utf-8")
    io.archive(output,VARIANT / f"{run_name}_analysis.zip",light=True)
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True); p.add_argument("--run-name",default="quick_v1")
    p.add_argument("--device",choices=["cpu","mps","cuda"],default="mps")
    a=p.parse_args(); run(a.inputs,a.run_name,a.device)
