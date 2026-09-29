"""v44: frozen transferred T01 + v25, inference only; no validation search/promotion."""
from training import osnet_r1_validation as checks  # Explicit device/environment and shared audit helpers.

import argparse
from pathlib import Path
import shutil

import numpy as np
import torch

from training import research_io as io, research_models as models, research_scoring as scoring
from training import dual_role_inference as dual, map_inference as v25
from training.transreid_model import ReIDModel
from backend.core import read_rows

VARIANT = io.ROOT / "OSNet-AIN-x1.0/variant_44_t01_validation"
CONTROL, CANDIDATE = "MVP_fusion_v25", "V25_T01_pre_graph_w10"
SYSTEMS = (CONTROL, CANDIDATE)
TRIAL = "T01_p8k2_supcon_lr3e-05"
THRESHOLD = checks.THRESHOLD
CHECKPOINT_SHA256 = "6f282d73bb2e9293e7cdde9b52ec066d784abd1eaa397bda2ac13f0593aa58be"
POLICY = {"control_weight": .9, "expert_weight": .1, "k1": 20, "k2": 3, "lambda": .5}
LAYOUT = {"dimension": 2432, "mvp": [0, 512], "original_R1_candidates": [512, 2048],
          "T01": [2048, 2432], "normalization": "real unit blocks, not global normalization"}


def pack(original, member):
    dual.unpack(original)
    dual.validate_block(member, 384)
    if len(original) != len(member):
        raise ValueError("Original and T01 row counts/order must match")
    return np.concatenate([original, member], axis=1)


def rank(values, n_query, system):
    if system not in SYSTEMS or not 0 < n_query < len(values):
        raise ValueError("Unknown system or query count")
    if values.ndim != 2 or values.shape[1] != (2048 if system == CONTROL else 2432):
        raise ValueError("Wrong real feature-bank layout")
    original = np.ascontiguousarray(values[:, :2048])
    baseline = v25.rank(original[:n_query], original[n_query:])
    if system == CONTROL:
        return baseline
    extra = values[:, 2048:]
    dual.validate_block(extra, 384)
    # Exactly the v41 pre_graph_10 arithmetic, now with the real deployed v25 blocks.
    mixed = scoring.mix([scoring.control_vectors(original), extra], [.9, .1])
    ranking = dual.policy.rank_vectors(mixed[:n_query], mixed[n_query:], "legacy")
    return {"order": ranking["order"], "raw_order": baseline["raw_order"],
            "confidence": baseline["confidence"]}


def export_arrays(directory, query, gallery, values, system):
    directory = Path(directory)
    if directory.exists() or len(values) != len(query)+len(gallery) or len(gallery) < 10:
        raise ValueError("Need new output, matching rows and at least ten gallery images")
    ranked = rank(values, len(query), system)
    metrics = dual.policy.export_csv(directory, query, gallery, ranked, THRESHOLD, "raw_top1")
    np.save(directory / "embeddings.npy", values)
    io.write(directory / "embedding_order.json", {
        "ids": [r["image_id"] for r in query+gallery], "query_count": len(query), "gallery_count": len(gallery),
        "system": system, "layout": dual.LAYOUT if system == CONTROL else LAYOUT,
        "ranking": v25.SPEC if system == CONTROL else POLICY, "threshold": THRESHOLD,
        "candidate": "unchanged original v25 R1 equal3 raw_top1", "checkpoint": None if system == CONTROL else CHECKPOINT_SHA256,
        "sha256": io.sha(directory / "embeddings.npy")})
    replay = rank(np.load(directory / "embeddings.npy", allow_pickle=False), len(query), system)
    if dual.policy.predictions(query, gallery, ranked, THRESHOLD, "raw_top1") != dual.policy.predictions(
            query, gallery, replay, THRESHOLD, "raw_top1"):
        raise ValueError("NPY replay changed decisions")
    diagnostics = dual.policy.query_diagnostics(query, gallery, ranked)["per_query"]
    for i, row in enumerate(query):
        diagnostics[row["image_id"]]["top10"] = [gallery[j]["image_id"] for j in ranked["order"][i, :10]]
    return {"system": system, **metrics, "per_query": diagnostics}


def prepare(run_name="validation_v1", device="mps"):
    if not run_name or not run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Simple RUN_NAME required")
    settings = io.read(VARIANT / "config.json")
    if (settings["checkpoint_sha256"] != CHECKPOINT_SHA256 or settings["policy"] != POLICY
            or settings["threshold"] != THRESHOLD or settings["batch_size"] != 8 or settings["vector_atol"] != 2e-5):
        raise ValueError("Frozen checkpoint, policy, threshold or precision changed")
    device = checks.device_for(device)
    source = io.child(io.ROOT.parent, settings["source_run"])
    protected = {str(io.child(io.ROOT, p)): h for p, h in settings["files"].items()}
    protected.update({str(io.child(source, p)): h for p, h in settings["source_files"].items()})
    protected[str(VARIANT / "config.json")] = io.sha(VARIANT / "config.json")
    checks.verify_files(protected)
    source_meta = io.read(source / "manifest.json")
    io.verify(io.ROOT, source_meta["source_sha256"])
    source_signature = io.digest(source_meta)
    stage = io.child(source, settings["source_stage"])
    report = io.read(stage / "result.json")
    trial = io.read(stage.parent / "trial.json")
    expected = io.digest({"run": source_signature, "spec": report["spec"], "rung": 10})
    if (report["spec"]["id"] != TRIAL or report["rung"] != 10 or report["status"] != "complete"
            or report["signature"] != expected or not io.completed(stage, expected)
            or report["training"]["step"] != 2883 or report["training"]["sha256"] != CHECKPOINT_SHA256
            or trial["spec"] != report["spec"] or trial["horizon"] != 25943
            or trial["signature"] != io.digest({"run": source_signature, "spec": trial["spec"], "horizon": trial["horizon"]})):
        raise ValueError("Expected completed T01 rung10, not a mutable resume or later checkpoint")
    checkpoint = io.child(source, report["training"]["path_from_run"])
    if str(checkpoint) not in protected or io.sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("T01 checkpoint is not protected")
    inputs = io.read(io.ROOT / settings["inputs"] / "inputs.json")
    if source_meta["input_sha256"] != io.sha(io.ROOT / settings["inputs"] / "inputs.json"):
        raise ValueError("Transferred primary split/input definition changed")
    old = io.read(io.ROOT / settings["v25_run"] / "manifest.json")
    if io.read(io.ROOT / settings["v25_run"] / "frozen_selection.json")["selected"] != v25.SPEC:
        raise ValueError("Approved v25 ranking changed")
    split = io.read(io.ROOT / "artifacts/splits.json")
    dataset = io.ROOT / "dataset"
    rows = read_rows(dataset / "train.csv")
    query, gallery = checks.validate_split(rows, old["protocols"]["validation"], split["identities"],
                                          inputs["train_ids"], inputs["holdout_ids"])
    profile_path = io.ROOT / settings["v24_profile"]
    profile, paths, bundle = dual.load_profile(profile_path)
    product = io.ROOT.parent / "Car-classification-MSK-main"
    spec = io.read(product / "models/profiles.json")[CONTROL]
    if (io.read(product / "release_decision.json")["active_profile"] != CONTROL
            or profile["threshold"] != THRESHOLD
            or spec["bundle_sha256"] != io.sha(paths["r1_bundle"]) or spec["r1_weight"] != .5
            or spec["ranking"] != "legacy" or spec["lambda"] != .5 or spec["candidate_policy"] != "raw_top1"):
        raise ValueError("Deployed v25 no longer matches this comparison")
    deployed_bundle = product / spec["bundle"]
    if io.sha(deployed_bundle) != spec["bundle_sha256"] or io.sha(product / "models" / paths["mvp"].name) != profile["mvp"]["sha256"]:
        raise ValueError("Deployed control weights differ")
    for root_bundle in (paths["r1_bundle"], deployed_bundle):
        protected[str(root_bundle)] = io.sha(root_bundle)
        for item in bundle["members"]:
            member = (root_bundle.parent / item["path"]).resolve()
            if io.sha(member) != item["sha256"]:
                raise ValueError("Changed R1 member bundle")
            model = io.read(member)["model"]
            model_path = (member.parent / model["path"]).resolve()
            if io.sha(model_path) != model["sha256"]:
                raise ValueError("Changed full-train R1 weights")
            protected.update({str(member): item["sha256"], str(model_path): model["sha256"]})
    protected[str(paths["mvp"])] = profile["mvp"]["sha256"]
    for path in [product / "release_decision.json", product / "models/profiles.json", *sorted((product / "models").rglob("*"))]:
        if path.is_file(): protected[str(path)] = io.sha(path)
    lookup = {r["image_id"]: r for r in rows}
    probe_rows = [lookup[i] for i in io.read(stage / "order.json")[:32]]
    for row in query+gallery+probe_rows:
        path = dual.image_path(dataset, row["image_id"])
        protected[str(path)] = split["frame_sha256"][row["image_id"]]
        row["path"] = path.relative_to(dataset).as_posix()
    reference = io.ROOT / settings["v25_stage"]
    ids = [r["image_id"] for r in query+gallery]
    if io.read(reference / "export/embedding_order.json")["ids"] != ids:
        raise ValueError("Original validation row order changed")
    original = np.load(reference / "export/embeddings.npy", allow_pickle=False)
    dual.unpack(original)
    if len(original) != len(ids):
        raise ValueError("Original validation feature count changed")
    checks.verify_files(protected)
    manifest = {"version": 44, "settings": settings, "systems": list(SYSTEMS), "ranking": POLICY,
                "source_signature": source_signature, "checkpoint_signature": trial["signature"],
                "spec": report["spec"], "trained_ids": inputs["train_ids"], "threshold": THRESHOLD,
                "protocol": old["protocols"]["validation"], "protected": protected,
                "source_sha256": io.source_hashes(), "runtime": io.runtime(device),
                "optimizer_updates": 0, "bn_updates": 0, "threshold_fit": False, "promoted": False,
                "scope": "T01/pre_graph10 fixed from primary only; previously observed original validation, not hidden test",
                "deployment_cost": "five encoders, 2432 real float32 coordinates; no deployment/speed acceptance"}
    print(f"PREFLIGHT OK: {len(query)} query / {len(gallery)} gallery; T01 step2883; {device}; no training", flush=True)
    return {"output": VARIANT / "runs" / run_name, "manifest": manifest, "signature": io.digest(manifest),
            "device": device, "dataset": dataset, "query": query, "gallery": gallery, "probe": probe_rows,
            "original": original, "stage": stage, "checkpoint": checkpoint, "reference": reference, "profile": profile_path}


def load_model(c):
    if io.sha(c["checkpoint"]) != CHECKPOINT_SHA256:
        raise ValueError("T01 checkpoint changed")
    saved = torch.load(c["checkpoint"], map_location="cpu", weights_only=True)
    if (saved["signature"] != c["manifest"]["checkpoint_signature"] or saved["spec"] != c["manifest"]["spec"]
            or saved["step"] != 2883):
        raise ValueError("Wrong T01 checkpoint provenance")
    # All tensors are strictly restored; no pretrained download or partial/random fallback.
    model = ReIDModel(len(c["manifest"]["trained_ids"]), "global", pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(c["device"]).eval().requires_grad_(False)


def probe(c, model):
    expected = np.load(c["stage"] / "features.npy", allow_pickle=False)[:32]
    actual = models.encode(model, c["dataset"], c["probe"], c["device"], batch_size=8)
    if not np.allclose(actual, expected, rtol=0, atol=2e-5):
        raise ValueError("Fresh T01 features differ from Mac mini; do not increase tolerance")
    return {"status": "passed", "max_error": float(abs(actual-expected).max()), "images": len(actual)}


def run(run_name="validation_v1", device="mps"):
    c = prepare(run_name, device)
    if shutil.disk_usage(VARIANT).free < 1024**3:
        raise OSError("Need 1 GiB free; historical files will not be removed")
    with io.lock(c["output"]):
        io.freeze(c["output"] / "manifest.json", c["manifest"])
        io.freeze(c["output"] / "frozen_candidate.json", {"trial": TRIAL, "step": 2883, "sha256": CHECKPOINT_SHA256,
                  "systems": list(SYSTEMS), "policy": POLICY, "threshold": THRESHOLD, "selection": "primary only"})
        model = load_model(c)
        buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        checks.task(c, "runtime_probe", lambda _: probe(c, model))
        ids = [r["image_id"] for r in c["query"]+c["gallery"]]
        def extract(directory):
            values = models.encode(model, c["dataset"], c["query"]+c["gallery"], c["device"], batch_size=8)
            dual.validate_block(values, 384)
            np.save(directory / "features.npy", values)
            return {"ids": ids, "dimension": 384}
        if checks.task(c, "T01_features", extract)["ids"] != ids:
            raise ValueError("Extracted T01 row order changed")
        if any(not torch.equal(v.cpu(), buffers[k]) for k, v in model.named_buffers()):
            raise ValueError("Inference changed model buffers")
        del model
        if c["device"].type == "mps": torch.mps.empty_cache()
        def fresh_control(directory):
            encoder = dual.DualRoleEncoder(c["profile"])
            values = encoder.encode_rows(c["query"]+c["gallery"], c["dataset"], 16)
            if not np.allclose(values, c["original"], rtol=0, atol=2e-5):
                raise ValueError("Fresh v25 features differ from protected reference")
            np.save(directory / "features.npy", values)
            return {"ids": ids, "max_error": float(abs(values-c["original"]).max())}
        if checks.task(c, "v25_features", fresh_control)["ids"] != ids:
            raise ValueError("Fresh v25 row order changed")
        original = np.load(c["output"] / "v25_features/features.npy", allow_pickle=False)
        member = np.load(c["output"] / "T01_features/features.npy", allow_pickle=False)
        reports = {}
        for system in SYSTEMS:
            values = original if system == CONTROL else pack(original, member)
            reports[system] = checks.task(c, system, lambda directory: export_arrays(
                directory / "export", c["query"], c["gallery"], values, system))
        control, candidate = (c["output"] / name / "export" for name in SYSTEMS)
        for name in ("submission.csv", "candidates.csv"):
            if (control / name).read_bytes() != (c["reference"] / "export" / name).read_bytes():
                raise ValueError(f"v25 reference replay changed {name}")
        if (candidate / "candidates.csv").read_bytes() != (control / "candidates.csv").read_bytes():
            raise ValueError("Original candidates/refusals changed")
        comparison = checks.paired_comparison(reports[CONTROL], reports[CANDIDATE])
        checks.check_unchanged(c)
        result = {"status": "complete", "signature": c["signature"], "systems": reports, "comparison": comparison,
                  "protected_unchanged": True, "optimizer_updates": 0, "bn_updates": 0, "threshold_fit": False,
                  "promoted": False, "validation_map_higher": comparison["delta_map"] > 0}
        io.write(c["output"] / "results.json", result)
        lines = ["# v44 — готовый T01 + рабочий v25", "", "Завершено; нового обучения и подбора параметров нет.", "",
                 "| Система | mAP@10 | Rank-1 | F1 | TNR |", "|---|---:|---:|---:|---:|"]
        for name, r in reports.items():
            lines.append(f"| {name} | {r['ranking']['mAP@10']:.6f} | {r['ranking']['Rank-1']:.6f} | "
                         f"{r['candidates']['F1']:.6f} | {r['candidates']['TNR']:.6f} |")
        lines += ["", f"Δ mAP: {comparison['delta_map']:+.6f}.",
                  f"Запросов лучше/хуже/без изменения AP: {comparison['better']}/{comparison['worse']}/{comparison['equal']}.",
                  f"Парный bootstrap 95%: {comparison['bootstrap_95']} (описательный).", "",
                  "T01 step2883 / 10 проходов; политика выбрана на primary до validation.",
                  "Ranking: 90% v25 + 10% T01 перед графом; k1=20, k2=3, lambda=.5.",
                  "Исходная ветка кандидатов/отказов побайтно сохранена; threshold=0.534365177154541.",
                  "Обе системы использовали свежие признаки v25; все решения контроля совпали с историческим экспортом.",
                  "Пять энкодеров вместо четырёх; реальные 2432 float32 координаты.",
                  "Исходная validation уже наблюдалась; это не новый независимый test.",
                  "ONNX, GPU/скорость и интеграция в приложение не проверены. Автопродвижения нет.",
                  "Рабочий MVP, переданный v41, bbox и исходные данные не изменены."]
        (c["output"] / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        io.archive(c["output"], VARIANT / f"{run_name}_analysis.zip", light=True)
        print(f"DONE: delta mAP={comparison['delta_map']:+.6f}; MVP unchanged", flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="validation_v1")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = parser.parse_args()
    run(args.run_name, args.device)
