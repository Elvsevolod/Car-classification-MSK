"""v22: compare concrete saved weights; no training, search, or automatic release."""
import gc
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.core import ARTIFACTS, MODEL, ROOT, Encoder, encode_rows, normalize, sha256
from training import model_search as search
from training.audit import digest, fixed_reference
from training.frozen_inference import _encode_rows
from training.osnet_ablations import AblationDataset
from training.policy_inference import PolicyEncoder, validate_bundle
from training.stage6 import write_json

old = search.old
VARIANT = ROOT / "OSNet-AIN-x1.0/variant_22_final_comparison"
FULL_CASE = "R1_equal3/less_graph/raw_top1"


def check_runtime(context):
    if old.seeds.runtime(context["device"]) != context["manifest"]["runtime"]:
        raise old.IntegrityError("Use the original kernel, device, threads and determinism")
    device = context["device"].type
    if ((device == "mps" and not torch.backends.mps.is_available())
            or (device == "cuda" and not torch.cuda.is_available())):
        raise old.IntegrityError(f"{device} unavailable; no silent CPU fallback")


def validate_protocols(context):
    outer = context["manifest"]["outer"]
    train, cal, val = (set(outer[k]) for k in ("train", "calibration", "validation"))
    primary = set(context["manifest"]["inner"]["primary"]["train"])
    if not primary or not primary <= train or train & (cal | val) or cal & val:
        raise old.IntegrityError("Training/calibration/validation identity overlap")
    all_ids = [r["image_id"] for r in context["rows"]]
    if len(all_ids) != len(set(all_ids)):
        raise old.IntegrityError("Duplicate source image IDs")
    seen = set()
    for split in ("calibration", "validation"):
        query, gallery = old.review.old.protocol_rows(context, split)
        ids = [r["image_id"] for r in query + gallery]
        if (not query or len(gallery) < 10 or len(ids) != len(set(ids)) or seen & set(ids)
                or not {r["vehicle_id"] for r in query + gallery} <= set(outer[split])):
            raise old.IntegrityError(f"Invalid original {split} protocol")
        seen.update(ids)


def prepare(run_name="comparison_v1", *, source_run="review_v1", search_run="search_v1", policy_run="policy_v1"):
    source, _, protected = old.seeds.load_source(source_run)
    check_runtime(source)
    validate_protocols(source)
    source["confirmation"] = old.read(source["output"] / "confirm_inner.json")
    search_output = old.seeds.run_directory(search.VARIANT, search_run)
    sm = old.read(search_output / "manifest.json")
    sr = old.read(search_output / "results.json")
    selected = old.read(search_output / "selected_candidate.json")
    policy_output = old.seeds.run_directory(old.previous.VARIANT, policy_run)
    pm = old.read(policy_output / "manifest.json")
    if (sr["status"] != "complete" or not sr["protected_unchanged"] or sr["selection"] != selected
            or selected["source_signature"] != digest(sm)
            or pm["retrieval_plan"]["source_signature"] != source["signature"]
            or sm["policy_source_signature"] != digest(pm)
            or any(m[k] != source["manifest"][k] for m in (sm, pm)
                   for k in ("outer", "inner", "protocols", "frames_sha256", "runtime"))):
        raise old.IntegrityError("Completed v16/v18/v21 sources disagree")
    protected.update(sm["protected"])
    protected.update(pm["protected"])
    for path in search_output.glob("tasks/*/complete.json"):
        protected.update(search.verify_task_source(search_output, digest(sm), path))
    protected.update(search.verify_task_source(policy_output, digest(pm), policy_output / "final_complete.json"))
    for path in (search_output / "manifest.json", search_output / "results.json",
                 search_output / "selected_candidate.json", policy_output / "manifest.json"):
        protected[str(path)] = sha256(path)

    seeds = source["seeds"]
    controls = [f"R1_control_{s}" for s in seeds]
    average = f"R1_full1700_avg_{seeds[0]}"
    expected = search.system(f"replace_{seeds[0]}", [average, *controls[1:]])
    if (selected["system"] != expected or selected["ranking"] != {"k1": 20, "k2": 3, "lambda": .65}
            or selected["candidate_policy"] != "raw_top1" or selected["full_train"]
            or selected["threshold"] is not None or selected["promoted"]):
        raise old.IntegrityError("This comparison is for the saved v21 replace-seed15/lambda65 finalist only")
    components = {}
    for name, seed in zip(controls, seeds):
        summary = old.source_summary(source, "primary", seed)
        entry = summary["checkpoints"]["800"]
        components[name] = {"kind": "torch", "seed": seed, "path": str(source["output"] / entry["path"]),
                            "sha256": entry["sha256"], "dimension": 512}
    components[average] = {"kind": "average", "seed": seeds[0], "dimension": 512,
                           **{k: selected["members"][average][k] for k in ("path", "sha256")}}
    for name in selected["system"]["members"]:
        if any(components[name][k] != selected["members"][name][k] for k in ("path", "sha256")):
            raise old.IntegrityError("Selected component differs from the original checkpoint")
    for component in components.values():
        if (component["path"] not in protected or protected[component["path"]] != component["sha256"]
                or sha256(component["path"]) != component["sha256"]):
            raise old.IntegrityError("Missing or changed verified primary weights")
    payload = torch.load(components[average]["path"], map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if (metadata["case"] != "R1_full1700_avg" or metadata["context_signature"] != sm["quality_source_signature"]
            or metadata["bn_identity_digest"] != digest(sorted(source["manifest"]["inner"]["primary"]["train"]))):
        raise old.IntegrityError("Saved average has different train-only BN provenance")
    del payload

    bundle_path = policy_output / "final" / FULL_CASE / "bundle.json"
    bundle = old.read(bundle_path)
    validate_bundle(bundle)
    reference = next(c for c in old.read(policy_output / "final.json")["cases"] if c["case"] == FULL_CASE)
    if (sha256(bundle_path) != reference["bundle_sha256"] or bundle["ranking"] != "less_graph"
            or bundle["candidate_policy"] != "raw_top1" or len(bundle["members"]) != 3
            or bundle["calibration"]["protocol_sha256"] != digest(source["manifest"]["protocols"]["calibration"])):
        raise old.IntegrityError("Full-train v18 profile changed")
    for member in bundle["members"]:
        path = (bundle_path.parent / member["path"]).resolve()
        value = old.read(path)
        model = (path.parent / value["model"]["path"]).resolve()
        for p, expected_hash in ((path, member["sha256"]), (model, value["model"]["sha256"])):
            if protected.get(str(p)) != expected_hash or sha256(p) != expected_hash:
                raise old.IntegrityError("Full-train encoder provenance changed")
    components["full_v18"] = {"kind": "policy", "path": str(bundle_path), "sha256": sha256(bundle_path), "dimension": 1536}
    threshold = fixed_reference()
    components["mvp"] = {"kind": "mvp", "path": str(MODEL), "sha256": sha256(MODEL), "dimension": 512}
    for path in (MODEL, ARTIFACTS / "baseline_metrics.json"):
        protected[str(path)] = sha256(path)
    specs = []
    for name, members, lam, candidate, train_count in (
            ("V21_selected_primary", expected["members"], .65, "raw_top1", len(source["manifest"]["inner"]["primary"]["train"])),
            ("R1_equal3_primary", controls, .75, "raw_top1", len(source["manifest"]["inner"]["primary"]["train"])),
            ("R1_equal3_full_v18", ["full_v18"], .75, "raw_top1", len(source["manifest"]["outer"]["train"])),
            ("MVP_recalibrated", ["mvp"], .50, "ranking_top1", len(source["manifest"]["outer"]["train"]))):
        specs.append({**search.system(name, members), "lambda": lam, "candidate_policy": candidate,
                      "train_identities": train_count, "dimension": sum(components[n]["dimension"] for n in members)})
    plan = {"systems": specs, "components": components, "active_mvp_threshold": threshold,
            "ranking_k1": 20, "ranking_k2": 3, "batch_torch": 32, "batch_onnx": 16,
            "threshold_rule": "calibration only; maximize C=.7F1+.3TNR, ties F1 then higher threshold",
            "runtime": "primary weights: original torch device; saved ONNX: explicit CPUExecutionProvider",
            "scope": "already observed outer development data; not an independent or hidden-test estimate",
            "optimizer_updates": 0, "bn_updates": 0, "wall_time_limit": None, "promoted": False,
            "source_selection": selected}
    manifest = {**source["manifest"], "version": 22, "comparison_plan": plan,
                "search_source_signature": digest(sm), "policy_source_signature": digest(pm),
                "protected": {**source["protected"], **protected},
                "source_sha256": {**sm["source_sha256"], "training/final_model_comparison.py": sha256(Path(__file__))}}
    context = {**source, "source": source, "output": old.seeds.run_directory(VARIANT, run_name),
               "manifest": manifest, "signature": digest(manifest), "protected": manifest["protected"]}
    old.review.check_inputs(context)
    old.review.check_other_runs(context)
    with old.review.old.run_lock(context["output"]):
        old.review.old.freeze_json(context["output"] / "manifest.json", manifest)
    return context


def validate_vectors(values, count, dimension):
    if (values.shape != (count, dimension) or values.dtype != np.float32 or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1., rtol=0, atol=2e-5)):
        raise old.IntegrityError("Invalid real embeddings; never replace missing/zero vectors")


def load_primary(context, spec):
    if sha256(spec["path"]) != spec["sha256"]:
        raise old.IntegrityError("Saved weights changed")
    summary = old.source_summary(context["source"], "primary", spec["seed"])
    model, variant = old.review.load_model(context["source"], {**summary, "stop_step": 800})
    if spec["kind"] == "average":
        saved = torch.load(spec["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False)
    return model.eval(), variant


def feature_task(context, split, name, spec, directory):
    if split == "validation":
        read_frozen(context)  # No validation inference before ALL calibration choices are fixed.
    query, gallery = old.review.old.protocol_rows(context, split)
    rows = query + gallery
    if spec["kind"] in ("torch", "average"):
        model, variant = load_primary(context, spec)
        try:
            loader = DataLoader(AblationDataset(rows, variant, context["dataset"]),
                                batch_size=32, shuffle=False, num_workers=0)
            batches = []
            with torch.no_grad():
                for i, (batch, _, _) in enumerate(loader):
                    batches.append(normalize(model.embedding(batch.to(context["device"])).cpu().numpy()))
                    if i % 10 == 0 or i + 1 == len(loader):
                        print(f"  {split}/{name}: {min((i+1)*32,len(rows))}/{len(rows)} images", flush=True)
            vectors = np.concatenate(batches)
        finally:
            del model
            gc.collect()
    elif spec["kind"] == "mvp":
        vectors = encode_rows(Encoder(spec["path"]), rows, context["dataset"], batch_size=16)
    elif spec["kind"] == "policy":
        encoder = PolicyEncoder(spec["path"], "CPUExecutionProvider")
        batches = []
        # Keep query+gallery order and the historical batch16 boundaries.
        for start in range(0, len(rows), 160):
            batches.append(_encode_rows(encoder, rows[start:start+160], context["dataset"], 16))
            print(f"  {split}/{name}: {min(start+160,len(rows))}/{len(rows)} images", flush=True)
        vectors = np.concatenate(batches)
    else:
        raise ValueError("Unknown fixed component")
    validate_vectors(vectors, len(rows), spec["dimension"])
    path = directory / "features.npz"
    np.savez_compressed(path, vectors=vectors, ids=np.array([r["image_id"] for r in rows]))
    return {"path": str(path), "sha256": sha256(path), "model": spec, "split": split}


def split_features(context, queue, split):
    query, gallery = old.review.old.protocol_rows(context, split)
    ids = [r["image_id"] for r in query + gallery]
    features = {}
    for name, spec in context["manifest"]["comparison_plan"]["components"].items():
        item = queue.task(f"features_{split}_{name}", lambda d: feature_task(context, split, name, spec, d))
        if item is not None:
            with np.load(item["path"], allow_pickle=False) as arrays:
                if arrays["ids"].tolist() != ids:
                    raise old.IntegrityError("Cached embedding order differs from original protocol")
                features[name] = arrays["vectors"]
            validate_vectors(features[name], len(ids), spec["dimension"])
    if len(features) != len(context["manifest"]["comparison_plan"]["components"]):
        raise ValueError(f"Incomplete {split} features; resume the same RUN_NAME")
    return features


def system_vectors(spec, features):
    # Do not add a redundant normalization to the stored MVP/full-ensemble vectors.
    return (features[spec["members"][0]] if len(spec["members"]) == 1 else
            search.combine([features[n] for n in spec["members"]], spec["weights"]))


def calibrate(context, spec, features, *, split):
    if split != "calibration":
        raise ValueError("Threshold selection is calibration-only")
    query, gallery = old.review.old.protocol_rows(context, split)
    vectors = system_vectors(spec, features)
    ranking, _ = search.rank(vectors[:len(query)], vectors[len(query):], spec["lambda"])
    result = old.policy.calibrate_policy(query, gallery, ranking, spec["candidate_policy"], split=split)
    return {**result, "system": spec, "protocol_sha256": digest(context["manifest"]["protocols"][split])}


def read_frozen(context):
    value = old.read(context["output"] / "frozen_comparison.json")
    specs = context["manifest"]["comparison_plan"]["systems"]
    if (value["signature"] != context["signature"] or value["systems"] != specs
            or set(value["thresholds"]) != {s["name"] for s in specs}
            or value["calibration_protocol_sha256"] != digest(context["manifest"]["protocols"]["calibration"])):
        raise old.IntegrityError("Frozen comparison changed")
    for spec in specs:
        directory = context["output"] / "tasks" / f"calibrate_{spec['name']}"
        search.verify_task_source(context["output"], context["signature"], directory / "complete.json")
        cal = old.read(directory / "result.json")
        if (cal["split"] != "calibration" or cal["system"] != spec
                or cal["candidate_policy"] != spec["candidate_policy"]
                or cal["protocol_sha256"] != value["calibration_protocol_sha256"]
                or not np.isfinite(value["thresholds"][spec["name"]])
                or value["thresholds"][spec["name"]] != cal["selected"]["threshold"]):
            raise old.IntegrityError("Threshold differs from its calibration receipt")
    return value


def evaluate(context, spec, features, threshold, directory):
    frozen = read_frozen(context)
    origin = "MVP_recalibrated" if spec["name"] == "MVP_active" else spec["name"]
    if {**spec, "name": origin} not in frozen["systems"]:
        raise old.IntegrityError("Validation system differs from the frozen configuration")
    expected = (context["manifest"]["comparison_plan"]["active_mvp_threshold"]
                if spec["name"] == "MVP_active" else frozen["thresholds"][origin])
    if threshold != expected:
        raise old.IntegrityError("Validation threshold was not frozen on calibration")
    query, gallery = old.review.old.protocol_rows(context, "validation")
    vectors = system_vectors(spec, features)
    validate_vectors(vectors, len(query)+len(gallery), spec["dimension"])
    ranking, _ = search.rank(vectors[:len(query)], vectors[len(query):], spec["lambda"])
    output = directory / "export"
    metrics = old.previous.verify_or_export_csv(output, query, gallery, ranking, threshold, spec["candidate_policy"])
    if metrics != old.policy.evaluate(query, gallery, ranking, threshold, spec["candidate_policy"]):
        raise old.IntegrityError("Official CSV metrics differ from in-memory predictions")
    path = output / "embeddings.npy"
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), vectors):
            raise old.IntegrityError("Existing export embeddings differ")
    else:
        pending = output / "embeddings.pending.npy"
        np.save(pending, vectors)
        pending.replace(path)
    old.review.old.freeze_json(output / "embedding_order.json", {
        "ids": [r["image_id"] for r in query + gallery], "query_count": len(query),
        "gallery_count": len(gallery), "dimension": spec["dimension"], "sha256": sha256(path)})
    _, accepted = old.policy.predictions(query, gallery, ranking, threshold, spec["candidate_policy"])
    return {"system": spec, "threshold": threshold, **metrics,
            "accepted": len(accepted), "refused": len(query)-len(accepted),
            "raw_map": old.policy.evaluate(query, gallery, {**ranking, "order": ranking["raw_order"]},
                                           threshold, spec["candidate_policy"])["ranking"]["mAP@10"],
            **old.policy.query_diagnostics(query, gallery, ranking), "export": str(output)}


def paired_delta(before, after):
    if set(before["per_query"]) != set(after["per_query"]):
        raise old.IntegrityError("Unpaired validation queries")
    delta = [after["per_query"][qid]["ranking"]["ap"] - row["ranking"]["ap"]
             for qid, row in before["per_query"].items() if row["ranking"]["ap"] is not None]
    return {"mAP_delta": after["ranking"]["mAP@10"]-before["ranking"]["mAP@10"],
            "C_delta": after["candidates"]["C"]-before["candidates"]["C"],
            "improved": sum(d > 1e-12 for d in delta), "worsened": sum(d < -1e-12 for d in delta),
            "unchanged": sum(abs(d) <= 1e-12 for d in delta)}


def write_report(context, result):
    lines = ["# v22 — сравнение конкретных сохранённых моделей", "", f"Статус: **{result['status']}**.",
             "Это уже наблюдавшиеся outer development данные, не независимый тест и не оценка hidden test.",
             "Нового обучения, пересчёта BatchNorm, поиска λ и смены MVP нет. Пороги выбраны только на calibration.",
             "", "| Система | train ID | λ | Порог | mAP@10 | Rank-1 | Rank-5 | F1 | TNR | C | Принято / отказ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, report in result["evaluations"].items():
        if report is None:
            lines.append(f"| {name} | — | — | — | FAILED | — | — | — | — | — | — |")
            continue
        spec, r, c = report["system"], report["ranking"], report["candidates"]
        lines.append(f"| {name} | {spec['train_identities']} | {spec['lambda']:.2f} | {report['threshold']:.8f} | "
                     f"{r['mAP@10']:.6f} | {r['Rank-1']:.6f} | {r['Rank-5']:.6f} | "
                     f"{c['F1']:.6f} | {c['TNR']:.6f} | {c['C']:.6f} | {report['accepted']} / {report['refused']} |")
    for name, change in result.get("selected_vs", {}).items():
        lines += ["", f"Новый v21 против {name}: ΔmAP {100*change['mAP_delta']:+.3f} п.п.; "
                  f"ΔC {100*change['C_delta']:+.3f} п.п.; запросов лучше/хуже/без изменений: "
                  f"{change['improved']}/{change['worsened']}/{change['unchanged']}."]
    lines += ["", "## Как читать результат", "",
              "- V21_selected_primary и R1_equal3_primary обучались на 740 ID; full_v18 и MVP — на 925. Это разные конкретные веса, не чистая абляция рецепта.",
              "- MVP_active использует действующий исторический порог. MVP_recalibrated — отдельная оценка того же encoder с общей calibration-only процедурой; рабочий порог не меняется.",
              "- C = 0.7 F1 + 0.3 TNR. Ранжирование и выбор принятого кандидата оцениваются отдельно; максимум mAP не обязательно максимум C.",
              "- raw_map и попарные AP сохранены в results.json как диагностика, без дополнительного отбора конфигураций.",
              "- Метрики получены из новых CSV официальным evaluator: десять ID даже при отказе. Junk удаляется evaluator уже после подачи top-10; одиннадцатое место не восстанавливается.",
              "- Исторические числа не подставляются вместо нового расчёта. Сравнение не является GPU benchmark или приёмкой релиза.",
              "- Экспорты — для исходного validation-протокола, не файлы для отправки на неизвестный test. Порядок реальных float32-векторов: query, затем gallery (embedding_order.json).",
              "- Готовые задачи возобновляются по checksum. Для изменения кода нужен новый RUN_NAME. Общего ограничения времени нет.",
              f"- Исходные файлы сохранены: {result.get('protected_unchanged', False)}. Автоматическое продвижение: False.",
              f"- Время текущего запуска: {result.get('elapsed_seconds', 0)/60:.1f} мин; cached-задачи не являются новым замером."]
    (context["output"] / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True is required for this development comparison")
    check_runtime(context)
    queue = old.Queue(context, wall_hours=None)
    result = {"status": "running", "evaluations": {}, "optimizer_updates": 0, "bn_updates": 0,
              "outer_evaluated": False, "promoted": False, "signature": context["signature"]}
    with old.review.old.run_lock(context["output"]):
        try:
            old.review.check_inputs(context, rehash=True)
            specs = context["manifest"]["comparison_plan"]["systems"]
            print("\nSTAGE 1/3: calibration features and thresholds", flush=True)
            features = split_features(context, queue, "calibration")
            calibration = {s["name"]: queue.task(f"calibrate_{s['name']}", lambda d: calibrate(
                context, s, features, split="calibration")) for s in specs}
            if any(v is None for v in calibration.values()):
                raise ValueError("Complete every calibration before validation; resume this RUN_NAME")
            frozen = {"signature": context["signature"], "systems": specs,
                      "calibration_protocol_sha256": digest(context["manifest"]["protocols"]["calibration"]),
                      "thresholds": {n: v["selected"]["threshold"] for n, v in calibration.items()}}
            old.review.old.freeze_json(context["output"] / "frozen_comparison.json", frozen)
            read_frozen(context)
            del features
            print("\nSTAGE 2/3: validation, all configurations and thresholds frozen", flush=True)
            features = split_features(context, queue, "validation")
            active = {**next(s for s in specs if s["name"] == "MVP_recalibrated"), "name": "MVP_active"}
            thresholds = {**frozen["thresholds"], "MVP_active": context["manifest"]["comparison_plan"]["active_mvp_threshold"]}
            for spec in [*specs, active]:
                name = spec["name"]
                result["evaluations"][name] = queue.task(f"evaluate_{name}", lambda d: evaluate(
                    context, spec, features, thresholds[name], d))
            if any(v is None for v in result["evaluations"].values()):
                raise ValueError("Incomplete validation export; resume this RUN_NAME")
            print("\nSTAGE 3/3: paired report; MVP unchanged", flush=True)
            selected = result["evaluations"]["V21_selected_primary"]
            result["selected_vs"] = {n: paired_delta(r, selected) for n, r in result["evaluations"].items()
                                     if n != "V21_selected_primary"}
            result.update(status="complete", outer_evaluated=True, frozen=frozen)
        except BaseException:
            result["status"] = "incomplete"
            raise
        finally:
            result.update(events=queue.events, elapsed_seconds=time.monotonic()-queue.started)
            try:
                old.review.check_inputs(context, rehash=True)
                result["protected_unchanged"] = True
            except BaseException:
                result.update(status="integrity_check_failed", protected_unchanged=False)
                raise
            finally:
                write_json(context["output"] / "results.json", result)
                write_report(context, result)
    return result
