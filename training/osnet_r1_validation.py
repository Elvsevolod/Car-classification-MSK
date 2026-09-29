"""v43: one frozen v42 checkpoint vs released v25; inference only, no promotion."""
import os
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import argparse
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch

from backend.core import normalize, read_rows
from training import research_io as io, research_models as models
from training import dual_role_inference as dual, map_inference as v25

VARIANT = io.ROOT / "OSNet-AIN-x1.0/variant_43_r03_validation"
CONTROL, CANDIDATE = "MVP_fusion_v25", "R03_replace_ranking_only"
SYSTEMS = (CONTROL, CANDIDATE)
THRESHOLD = 0.534365177154541
LAYOUT = {"dimension": 2560, "mvp": [0, 512], "original_R1_candidates": [512, 2048],
          "R03": [2048, 2560], "normalization": "real unit blocks; not a globally normalized vector"}


def pack(original, member):
    dual.unpack(original)
    dual.validate_block(member, 512)
    if len(original) != len(member):
        raise ValueError("Original and R03 image orders/counts must match")
    return np.concatenate([original, member], axis=1)


def rank(values, n_query, system):
    """Labels are never accepted here. Gallery is static; queries are independent."""
    if system not in SYSTEMS or not 0 < n_query < len(values):
        raise ValueError("Unknown system or invalid query count")
    if values.ndim != 2 or values.shape[1] != (2048 if system == CONTROL else 2560):
        raise ValueError("Wrong real-feature layout")
    original = values[:, :2048]
    mvp, r1 = dual.unpack(original)
    baseline = v25.rank(original[:n_query], original[n_query:])
    if system == CONTROL:
        return baseline
    member = values[:, 2048:]
    dual.validate_block(member, 512)
    # The original equal3 blocks are scaled by 1/sqrt(3); re-normalize each member.
    replaced = dual.policy.combine_members([member, r1[:, 512:1024], r1[:, 1024:]])
    ranking_bank = dual.pack(mvp, replaced)
    ranking = v25.rank(ranking_bank[:n_query], ranking_bank[n_query:])
    return {"order": ranking["order"], "raw_order": baseline["raw_order"],
            "confidence": baseline["confidence"]}


def export_arrays(directory, query, gallery, values, system):
    directory = Path(directory)
    if directory.exists() or len(values) != len(query) + len(gallery) or len(gallery) < 10:
        raise ValueError("Need new output, matching rows, and at least ten gallery images")
    ranked = rank(values, len(query), system)
    metrics = dual.policy.export_csv(directory, query, gallery, ranked, THRESHOLD, "raw_top1")
    np.save(directory / "embeddings.npy", values)
    io.write(directory / "embedding_order.json", {
        "ids": [r["image_id"] for r in query + gallery], "query_count": len(query),
        "gallery_count": len(gallery), "layout": dual.LAYOUT if system == CONTROL else LAYOUT,
        "system": system, "ranking": v25.SPEC, "threshold": THRESHOLD,
        "candidate": "unchanged original full-train R1 equal3 raw_top1",
        "replacement": None if system == CONTROL else "seed_20260915 in ranking only",
        "sha256": io.sha(directory / "embeddings.npy")})
    replay = rank(np.load(directory / "embeddings.npy", allow_pickle=False), len(query), system)
    if dual.policy.predictions(query, gallery, ranked, THRESHOLD, "raw_top1") != dual.policy.predictions(
            query, gallery, replay, THRESHOLD, "raw_top1"):
        raise ValueError("NPY replay changed decisions")
    diagnostics = dual.policy.query_diagnostics(query, gallery, ranked)["per_query"]
    for i, row in enumerate(query):
        diagnostics[row["image_id"]]["top10"] = [gallery[j]["image_id"] for j in ranked["order"][i, :10]]
    return {"system": system, **metrics, "per_query": diagnostics}


def paired_comparison(control, candidate):
    if control["candidates"] != candidate["candidates"]:
        raise ValueError("Candidate metrics changed")
    left, right = control["per_query"], candidate["per_query"]
    if set(left) != set(right):
        raise ValueError("Unpaired query IDs")
    deltas = []
    for qid, a in left.items():
        b = right[qid]
        if any(a[k] != b[k] for k in ("vehicle_id", "known", "confidence", "raw")):
            raise ValueError("Original candidate decisions or query identity changed")
        if a["known"]:
            deltas.append({"query_id": qid, "vehicle_id": a["vehicle_id"],
                           "delta_ap": b["ranking"]["ap"] - a["ranking"]["ap"]})
    if not deltas or len({x["vehicle_id"] for x in deltas}) != len(deltas):
        raise ValueError("Expected one scored query per validation vehicle ID")
    values = np.array([r["delta_ap"] for r in deltas])
    rng = np.random.default_rng(20260929)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {"delta_map": candidate["ranking"]["mAP@10"] - control["ranking"]["mAP@10"],
            "better": int((values > 0).sum()), "worse": int((values < 0).sum()),
            "equal": int((values == 0).sum()), "per_query": deltas,
            "bootstrap_95": np.quantile(means, [.025, .975]).tolist(),
            "bootstrap_scope": "paired vehicle resampling; descriptive, no correction for earlier research selection",
            "top10_changed": sum(left[q]["top10"] != right[q]["top10"] for q in left),
            "candidates_unchanged": True}


def verify_files(files):
    for name, expected in files.items():
        if not Path(name).is_file() or io.sha(name) != expected:
            raise ValueError(f"Missing/changed protected file: {name}")


def validate_split(rows, protocol, identities, trained, inner_holdout):
    groups = {key: set(ids) for key, ids in identities.items()}
    if (set(trained) & set(inner_holdout) or set(trained) | set(inner_holdout) != groups["train"]
            or any(groups[a] & groups[b] for a, b in (("train", "calibration"), ("train", "validation"),
                                                     ("calibration", "validation")))):
        raise ValueError("Training/selection/validation identity leakage")
    lookup = {r["image_id"]: r for r in rows}
    ids = protocol["query_ids"] + protocol["gallery_ids"]
    if len(lookup) != len(rows) or len(set(ids)) != len(ids) or not protocol["query_ids"] or len(protocol["gallery_ids"]) < 10:
        raise ValueError("Invalid or overlapping query/gallery IDs")
    if not {lookup[i]["vehicle_id"] for i in ids} <= groups["validation"]:
        raise ValueError("Only the original validation protocol is allowed")
    return tuple([lookup[i] for i in protocol[key]] for key in ("query_ids", "gallery_ids"))


def device_for(name):
    if name not in {"cpu", "mps", "cuda"}:
        raise ValueError("Choose an explicit device")
    if name == "mps" and (not torch.backends.mps.is_available() or any(os.environ.get(k) != "0" for k in
            ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"))):
        raise ValueError("Native MPS required, with fallback/fast math disabled before torch import")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no CPU fallback")
    if name == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    return torch.device(name)


def prepare(run_name="validation_v1", device="mps"):
    if not run_name or not run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Simple RUN_NAME required")
    settings = io.read(VARIANT / "config.json")
    if settings["threshold"] != THRESHOLD or settings["vector_atol"] != 2e-5 or settings["batch_size"] != 16:
        raise ValueError("Frozen threshold/precision/batch settings changed")
    device = device_for(device)
    protected = {str(io.child(io.ROOT, p)): h for p, h in settings["files"].items()}
    verify_files(protected)
    source = io.ROOT / settings["source_run"]
    stage = io.ROOT / settings["source_stage"]
    source_meta, selection, result = (io.read(source / f) for f in ("manifest.json", "selected_candidate.json", "results.json"))
    best = selection["best_system"]
    if (result["status"] != "complete" or result["completed_trials"] != 18 or result["planned_trials"] != 18
            or selection["signature"] != io.digest(source_meta) or best != result["best_system"]
            or (best["trial"], best["presentations"], best["policy"]) !=
               ("R03_p16k2_supcon_lr1", 12800, "replace_R1_20260915")
            or best["checkpoint"]["sha256"] != settings["checkpoint_sha256"]
            or (source / best["checkpoint"]["path"]).resolve() != (stage / "checkpoint.pt").resolve()):
        raise ValueError("Expected the frozen concrete v42 winner, not a refit or last checkpoint")
    io.verify(io.ROOT, source_meta["source_sha256"])
    inputs = io.read(io.ROOT / settings["inputs"] / "inputs.json")
    v25_directory = io.ROOT / settings["v25_run"]
    reference = io.ROOT / settings["v25_stage"]
    old = io.read(v25_directory / "manifest.json")
    if io.read(v25_directory / "frozen_selection.json")["selected"] != v25.SPEC:
        raise ValueError("Approved v25 ranking changed")
    split = io.read(io.ROOT / "artifacts/splits.json")
    dataset = io.ROOT / "dataset"
    rows = read_rows(dataset / "train.csv")
    protocol = old["protocols"]["validation"]
    query, gallery = validate_split(rows, protocol, split["identities"], inputs["train_ids"], inputs["holdout_ids"])
    profile_path = io.ROOT / settings["v24_profile"]
    profile, paths, bundle = dual.load_profile(profile_path)
    if profile["threshold"] != THRESHOLD or "seed_20260915" not in bundle["members"][0]["path"]:
        raise ValueError("Wrong threshold or R1 member order")
    # Match the deployed product's assets, not merely a similarly named research profile.
    product = io.ROOT.parent / "Car-classification-MSK-main"
    active = io.read(product / "release_decision.json")
    spec = io.read(product / "models/profiles.json")[CONTROL]
    if (active["active_profile"] != CONTROL or spec["bundle_sha256"] != io.sha(paths["r1_bundle"])
            or spec["r1_weight"] != .5 or spec["lambda"] != .5 or spec["candidate_policy"] != "raw_top1"):
        raise ValueError("Active MVP no longer matches this experiment's control")
    deployed_bundle = product / spec["bundle"]
    if io.sha(deployed_bundle) != spec["bundle_sha256"] or io.sha(product / "models" / paths["mvp"].name) != profile["mvp"]["sha256"]:
        raise ValueError("Deployed MVP weights differ")
    for root_bundle in (paths["r1_bundle"], deployed_bundle):
        protected[str(root_bundle)] = io.sha(root_bundle)
        for item in bundle["members"]:
            member = (root_bundle.parent / item["path"]).resolve()
            if io.sha(member) != item["sha256"]:
                raise ValueError("Changed full-train R1 member")
            model = io.read(member)["model"]
            model_path = (member.parent / model["path"]).resolve()
            if io.sha(model_path) != model["sha256"]:
                raise ValueError("Changed full-train R1 weights")
            protected.update({str(member): item["sha256"], str(model_path): model["sha256"]})
    protected[str(paths["mvp"])] = profile["mvp"]["sha256"]
    for p in [product / "release_decision.json", product / "models/profiles.json", *sorted((product / "models").rglob("*"))]:
        if p.is_file(): protected[str(p)] = io.sha(p)
    lookup = {r["image_id"]: r for r in rows}
    probe = [lookup[i] for i in io.read(stage / "order.json")[:32]]
    for row in query + gallery + probe:
        path = dual.image_path(dataset, row["image_id"])
        protected[str(path)] = split["frame_sha256"][row["image_id"]]
        row["path"] = path.relative_to(dataset).as_posix()
    cache = reference / "export"
    if io.read(cache / "embedding_order.json")["ids"] != [r["image_id"] for r in query + gallery]:
        raise ValueError("Validation cache row order changed")
    original = np.load(cache / "embeddings.npy", allow_pickle=False)
    dual.unpack(original)
    if len(original) != len(query) + len(gallery):
        raise ValueError("Validation cache shape changed")
    verify_files(protected)
    manifest = {"version": 43, "settings": settings, "systems": list(SYSTEMS), "graph": v25.SPEC,
                "source_signature": selection["signature"], "winner": best, "protocol": protocol,
                "trained_ids": inputs["train_ids"], "parent": inputs["models"]["R1_20260915"],
                "spec": io.read(stage / "result.json")["spec"], "threshold": THRESHOLD,
                "protected": protected, "source_sha256": io.source_hashes(), "runtime": io.runtime(device),
                "optimizer_updates": 0, "bn_updates": 0, "threshold_fit": False, "promoted": False,
                "scope": "one primary-selected checkpoint; original previously observed validation; not hidden test",
                "deployment_cost": "4 encoders for v25; 5 for ranking-only replacement with unchanged refusals"}
    c = {"output": VARIANT / "runs" / run_name, "manifest": manifest, "signature": io.digest(manifest),
         "device": device, "dataset": dataset, "query": query, "gallery": gallery,
         "probe": probe, "original": original, "stage": stage, "reference": reference, "profile": profile_path}
    print(f"PREFLIGHT OK: {len(query)} query / {len(gallery)} gallery; R03 step400; {device}; no training", flush=True)
    return c


def load_model(c):
    path = c["stage"] / "checkpoint.pt"
    if io.sha(path) != c["manifest"]["settings"]["checkpoint_sha256"]:
        raise ValueError("R03 checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    expected_signature = io.digest({"run": c["manifest"]["source_signature"], "spec": c["manifest"]["spec"]})
    if (saved["signature"] != expected_signature or saved["spec"] != c["manifest"]["spec"]
            or saved["step"] != 400 or saved["presentations"] != 12800 or saved["parent"] != c["manifest"]["parent"]):
        raise ValueError("R03 checkpoint provenance changed")
    model = models.ReIDExperimentModel(len(c["manifest"]["trained_ids"]), use_bnneck=True)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(c["device"]).eval().requires_grad_(False)


def task(c, name, action):
    target = c["output"] / name
    if io.completed(target, c["signature"]):
        print(f"REUSE {name}: verified cache, not a new measurement", flush=True)
        return io.read(target / "result.json")
    if target.exists():
        raise ValueError(f"Uncommitted task directory: {target}; use a new RUN_NAME")
    began = time.perf_counter()
    print(f"STAGE {name}: start", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".pending_{name}_", dir=c["output"]) as tmp:
        directory = Path(tmp) / "stage"
        directory.mkdir()
        result = action(directory)
        io.write(directory / "result.json", result)
        io.finish(directory, c["signature"])
        directory.rename(target)
    print(f"STAGE {name}: done in {time.perf_counter()-began:.1f}s", flush=True)
    return result


def probe(c, model):
    expected = np.load(c["stage"] / "features.npy", allow_pickle=False)[:32]
    actual = models.encode(model, c["dataset"], c["probe"], c["device"], batch_size=16)
    if not np.allclose(actual, expected, rtol=0, atol=2e-5):
        raise ValueError("R03 fresh features differ from v42; do not increase tolerance")
    reference_encoder = dual.DualRoleEncoder(c["profile"])
    original = reference_encoder.encode_rows(c["query"][:8], c["dataset"], 8)
    if not np.allclose(original, c["original"][:8], rtol=0, atol=2e-5):
        raise ValueError("Fresh v25 images differ from the protected reference cache")
    return {"status": "passed", "r03_max_error": float(abs(actual-expected).max()),
            "v25_max_error": float(abs(original-c["original"][:8]).max()), "training_updates": 0}


def check_unchanged(c):
    verify_files(c["manifest"]["protected"])
    io.verify(io.ROOT, c["manifest"]["source_sha256"])


def run(run_name="validation_v1", device="mps"):
    c = prepare(run_name, device)
    if shutil.disk_usage(VARIANT).free < 1024**3:
        raise OSError("Need 1 GiB free; historical files will not be deleted")
    with io.lock(c["output"]):
        io.freeze(c["output"] / "manifest.json", c["manifest"])
        io.freeze(c["output"] / "frozen_candidate.json", {"winner": c["manifest"]["winner"],
                  "systems": list(SYSTEMS), "threshold": THRESHOLD, "selection": "v42 primary only; no validation tuning"})
        model = load_model(c)
        buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        task(c, "runtime_probe", lambda _: probe(c, model))
        def extract(directory):
            features = models.encode(model, c["dataset"], c["query"] + c["gallery"], c["device"], batch_size=16)
            dual.validate_block(features, 512)
            np.save(directory / "features.npy", features)
            return {"ids": [r["image_id"] for r in c["query"] + c["gallery"]], "dimension": 512}
        extraction = task(c, "features", extract)
        if extraction["ids"] != [r["image_id"] for r in c["query"] + c["gallery"]]:
            raise ValueError("Extracted row order changed")
        if any(not torch.equal(v.cpu(), buffers[k]) for k, v in model.named_buffers()):
            raise ValueError("Inference changed a model buffer")
        del model
        if c["device"].type == "mps": torch.mps.empty_cache()
        features = np.load(c["output"] / "features/features.npy", allow_pickle=False)
        reports = {}
        for system in SYSTEMS:
            values = c["original"] if system == CONTROL else pack(c["original"], features)
            reports[system] = task(c, system, lambda directory: export_arrays(
                directory / "export", c["query"], c["gallery"], values, system))
        control = c["output"] / CONTROL / "export"
        candidate = c["output"] / CANDIDATE / "export"
        for name in ("submission.csv", "candidates.csv"):
            if (control / name).read_bytes() != (c["reference"] / "export" / name).read_bytes():
                raise ValueError(f"v25 reference replay changed {name}")
        if (candidate / "candidates.csv").read_bytes() != (control / "candidates.csv").read_bytes():
            raise ValueError("Original refusals/candidates changed")
        comparison = paired_comparison(reports[CONTROL], reports[CANDIDATE])
        check_unchanged(c)
        result = {"status": "complete", "systems": reports, "comparison": comparison,
                  "signature": c["signature"], "protected_unchanged": True, "optimizer_updates": 0,
                  "bn_updates": 0, "threshold_fit": False, "promoted": False,
                  "validation_map_higher": comparison["delta_map"] > 0,
                  "note": "Not automatic promotion and not a claim about unseen test quality"}
        io.write(c["output"] / "results.json", result)
        lines = ["# v43 — сохранённый R03 против v25", "", "Завершено; нового обучения нет.", "",
                 "| Система | mAP@10 | Rank-1 | F1 | TNR |", "|---|---:|---:|---:|---:|"]
        for name, r in reports.items():
            lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} | {r['ranking']['Rank-1']:.6f} | "
                         f"{r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |")
        lines += ["", f"Δ mAP: {comparison['delta_map']:+.6f}.",
                  f"Запросов лучше/хуже/без изменения AP: {comparison['better']}/{comparison['worse']}/{comparison['equal']}.",
                  f"Парный bootstrap 95%: {comparison['bootstrap_95']} (описательная оценка).", "",
                  "R03 step400 / 12800 предъявлений; другие checkpoints на validation не проверялись.",
                  "Кандидаты/отказы побайтно совпали с v25; threshold = 0.534365177154541.",
                  "Ranking: MVP=.5; R03/R1_seed16/R1_seed17 по 1/6; k1=20, k2=3, lambda=.5.",
                  "Сохранение исходной кандидатской ветки требует 5 энкодеров против 4 у v25.",
                  "Вектор кандидата: реальные 2560 float32 координат с отдельным старым R1-блоком.",
                  "Исходная validation ранее уже использовалась; это не новый независимый test.",
                  "ONNX-export R03, скорость и внедрение в приложение в этот этап не входят.",
                  "MVP, v41, v42, bbox, исходные данные и веса не изменены. Автопродвижения нет."]
        (c["output"] / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        io.archive(c["output"], VARIANT / f"{run_name}_analysis.zip", light=True)
        print(f"DONE: delta mAP={comparison['delta_map']:+.6f}; MVP unchanged", flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="validation_v1")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = parser.parse_args()
    run(args.run_name, args.device)
