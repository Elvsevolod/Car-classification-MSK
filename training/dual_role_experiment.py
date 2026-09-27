"""v24: fixed MVP ranking + full-train R1 candidate; verify, never fit/select."""
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import PIL

from backend.core import DATASET, ROOT, read_rows, sha256
from training import dual_role_inference as inference
from training import final_model_comparison as previous
from training.audit import digest
from training.stage6 import write_json

old, search = previous.old, previous.search
VARIANT = ROOT / "OSNet-AIN-x1.0/variant_24_dual_role"
COMPONENTS = ("mvp", "full_v18")


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "onnxruntime": ort.__version__,
            "pillow": PIL.__version__, "provider": "CPUExecutionProvider"}


def source_task(context, name):
    directory = context["source_output"] / "tasks" / name
    search.verify_task_source(context["source_output"], context["manifest"]["dual_role_plan"]["source_signature"],
                              directory / "complete.json")
    return old.read(directory / "result.json")


def prepare(run_name="dual_role_v1", *, source_run="comparison_v1", dataset=DATASET):
    source_output = old.seeds.run_directory(previous.VARIANT, source_run)
    sm, sr = (old.read(source_output / n) for n in ("manifest.json", "results.json"))
    signature = digest(sm)
    if (sm["version"] != 22 or sr["status"] != "complete" or not sr["protected_unchanged"]
            or sr["signature"] != signature or sr["optimizer_updates"] != 0 or sr["bn_updates"] != 0
            or sr["promoted"]):
        raise old.IntegrityError("Need the complete unchanged v22 comparison")
    frozen = previous.read_frozen({"output": source_output, "manifest": sm, "signature": signature})
    if sr["frozen"] != frozen:
        raise old.IntegrityError("v22 report/frozen comparison mismatch")
    for name, expected in sm["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise old.IntegrityError(f"Original v22 source changed: {name}")
    protected = dict(sm["protected"])
    for receipt in source_output.glob("tasks/*/complete.json"):
        protected.update(search.verify_task_source(source_output, signature, receipt))
    for name in ("manifest.json", "results.json", "frozen_comparison.json"):
        protected[str(source_output / name)] = sha256(source_output / name)
    output = old.seeds.run_directory(VARIANT, run_name)
    components = sm["comparison_plan"]["components"]
    profile_path = output / "dual_role_profile.json"
    profile = inference.profile_value(profile_path, components["mvp"]["path"], components["full_v18"]["path"])
    if (profile["threshold"] != frozen["thresholds"]["R1_equal3_full_v18"]
            or profile["calibration"]["protocol_sha256"] != frozen["calibration_protocol_sha256"]
            or any(profile[key]["sha256"] != components[name]["sha256"]
                   for key, name in (("mvp", "mvp"), ("r1_bundle", "full_v18")))):
        raise old.IntegrityError("Dual-role profile differs from completed source weights/threshold")
    plan = {"source_run": source_run, "source_directory": str(source_output), "source_signature": signature,
            "profile": profile, "batch_size": 16, "vector_atol": 2e-5, "decision_parity": "exact CSV bytes",
            "scope": "fixed post-hoc development system; both splits already observed, not independent test",
            "threshold_fit": False, "selection": False, "optimizer_updates": 0, "bn_updates": 0,
            "wall_time_limit": None, "promoted": False}
    modules = ("training/dual_role_experiment.py", "training/dual_role_inference.py", "training/frozen_inference.py",
               "training/retrieval_policy.py", "training/preprocessing.py", "backend/core.py",
               "backend/evaluate.py", "backend/rerank.py", "evaluate.py")
    manifest = {**sm, "version": 24, "dual_role_plan": plan, "analysis_runtime": runtime(), "protected": protected,
                "source_sha256": {**sm["source_sha256"], **{p: sha256(ROOT / p) for p in modules}}}
    context = {"output": output, "source_output": source_output, "source_manifest": sm,
               "profile_path": profile_path, "dataset": Path(dataset), "rows": read_rows(Path(dataset)/"train.csv"),
               "manifest": manifest, "signature": digest(manifest), "protected": protected}
    previous.validate_protocols(context)
    old.review.check_inputs(context, rehash=True)
    for split in ("calibration", "validation"):
        for name in COMPONENTS:
            source_task(context, f"features_{split}_{name}")
    for name in ("evaluate_MVP_active", "evaluate_R1_equal3_full_v18"):
        source_task(context, name)
    old.review.check_other_runs(context)
    with old.review.old.run_lock(output):
        old.review.old.freeze_json(output / "manifest.json", manifest)
        old.review.old.freeze_json(profile_path, profile)
        frozen_profile(context)
    print(f"Preflight OK. Frozen threshold={profile['threshold']:.9f}; no fitting, no model selection.", flush=True)
    return context


def frozen_profile(context):
    profile, _, _ = inference.load_profile(context["profile_path"])
    if profile != context["manifest"]["dual_role_plan"]["profile"]:
        raise old.IntegrityError("Frozen dual-role profile changed")
    return profile


def load_vectors(context, split):
    if split not in ("calibration", "validation"):
        raise ValueError("Only original calibration/validation protocols")
    frozen_profile(context)
    query, gallery = old.review.old.protocol_rows(context, split)
    ids = [r["image_id"] for r in query+gallery]
    vectors = []
    for name in COMPONENTS:
        item = source_task(context, f"features_{split}_{name}")
        path = context["source_output"] / "tasks" / f"features_{split}_{name}" / "features.npz"
        spec = context["source_manifest"]["comparison_plan"]["components"][name]
        if (Path(item["path"]).resolve() != path.resolve() or sha256(path) != item["sha256"]
                or item["split"] != split or item["model"] != spec):
            raise old.IntegrityError("Source cache/model provenance changed")
        with np.load(path, allow_pickle=False) as data:
            if data["ids"].tolist() != ids:
                raise old.IntegrityError("Source feature IDs/order changed")
            values = data["vectors"]
        previous.validate_vectors(values, len(ids), spec["dimension"])
        vectors.append(values)
    return inference.pack(*vectors)


def verify_reference(context, split, query, gallery, values, metrics, export):
    """Compare concrete decisions, not just aggregate scores; do not loosen failures."""
    mvp, r1 = inference.unpack(values)
    n = len(query)
    left = inference.policy.rank_vectors(mvp[:n], mvp[n:], "legacy")
    right = inference.policy.rank_vectors(r1[:n], r1[n:], "less_graph")
    threshold = frozen_profile(context)["threshold"]
    actual = inference.policy.predictions(query, gallery, inference.rank(values[:n], values[n:]), threshold, "raw_top1")
    expected_ranking = inference.policy.predictions(query, gallery, left, threshold, "ranking_top1")[0]
    expected_candidates = inference.policy.predictions(query, gallery, right, threshold, "raw_top1")[1]
    if actual != (expected_ranking, expected_candidates):
        raise old.IntegrityError("Dual-role decisions differ from their original component")
    if split == "validation":
        for name, filename, key in (("MVP_active", "submission.csv", "ranking"),
                                    ("R1_equal3_full_v18", "candidates.csv", "candidates")):
            source = source_task(context, f"evaluate_{name}")
            path = context["source_output"] / "tasks" / f"evaluate_{name}" / "export" / filename
            if metrics[key] != source[key] or sha256(export / filename) != sha256(path):
                raise old.IntegrityError(f"{filename} no longer exactly reproduces v22; stop without retuning")
    else:
        selected = source_task(context, "calibrate_R1_equal3_full_v18")["selected"]
        if {"threshold": threshold, **metrics["candidates"]} != selected:
            raise old.IntegrityError("R1 calibration decisions changed at the original threshold")
    return {"MVP_top10_exact": True, "R1_candidate_confidence_refusal_exact": True,
            "v22_csv_byte_parity": split == "validation"}


def evaluate(context, split, values, directory, *, fresh=False, reference=None):
    profile = frozen_profile(context)
    query, gallery = old.review.old.protocol_rows(context, split)
    export = directory / "export"
    metrics = inference.export_arrays(profile, query, gallery, values, export)
    parity = verify_reference(context, split, query, gallery, values, metrics, export)
    if fresh:
        atol = context["manifest"]["dual_role_plan"]["vector_atol"]
        error = float(np.max(np.abs(values-reference)))
        if values.shape != reference.shape or not np.allclose(values, reference, rtol=0, atol=atol):
            raise old.IntegrityError(f"Fresh image embeddings differ from source; max error={error}; atol={atol}")
        parity.update(vector_atol=atol, max_abs_error=error, bit_exact_vectors=bool(np.array_equal(values, reference)))
    ranking = inference.rank(values[:len(query)], values[len(query):])
    ordered, accepted = inference.policy.predictions(query, gallery, ranking, profile["threshold"], "raw_top1")
    return {"split": split, "threshold": profile["threshold"], **metrics, "parity": parity, "export": str(export),
            "accepted": len(accepted), "refused": len(query)-len(accepted), "fresh_image_inference": fresh,
            "candidate_differs_from_top1": sum(c[0][0] != ordered[q][0] for q, c in accepted.items()),
            "candidate_outside_top10": sum(c[0][0] not in ordered[q] for q, c in accepted.items()),
            "quality_points_out_of_55": 45*metrics["ranking"]["mAP@10"]+10*metrics["candidates"]["C"]}


def fresh_validation(context, reference, directory):
    query, gallery = old.review.old.protocol_rows(context, "validation")
    encoder = inference.DualRoleEncoder(context["profile_path"])
    values = encoder.encode_rows(query+gallery, context["dataset"], context["manifest"]["dual_role_plan"]["batch_size"])
    return evaluate(context, "validation", values, directory, fresh=True, reference=reference)


def write_report(context, result):
    lines = ["# v24 — MVP ranking + full-train R1 candidate", "", f"Статус: **{result['status']}**.",
             "Профиль фиксирован заранее; нет обучения, подбора порога и выбора по текущим метрикам.",
             "Это post-hoc development-проверка на уже наблюдавшихся данных, не независимый тест.", "",
             "| Этап | mAP@10 | F1 | TNR | C | 45×mAP+10×C | Принято / отказ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in result["evaluations"].items():
        if item:
            a, b = item["ranking"], item["candidates"]
            lines.append(f"| {name} | {a['mAP@10']:.6f} | {b['F1']:.6f} | {b['TNR']:.6f} | {b['C']:.6f} | "
                         f"{item['quality_points_out_of_55']:.4f} | {item['accepted']} / {item['refused']} |")
        else:
            lines.append(f"| {name} | FAILED | — | — | — | — | — |")
    if result.get("comparison"):
        delta = result["comparison"]
        lines += ["", f"Против MVP: ΔmAP {delta['mAP_delta']:+.6f}; ΔC {delta['C_delta']:+.6f}; "
                  f"Δдвух компонентов {delta['quality_points_delta']:+.4f} балла из 55.",
                  "Рост относится к выбору кандидата, а не к ranking. Это не полный конкурсный балл."]
    item = result["evaluations"].get("fresh_validation")
    if item:
        lines += ["", f"Новый проход по изображениям: {item['parity']}.",
                  f"Принятый R1-кандидат отличается от top-1 MVP у {item['candidate_differs_from_top1']} запросов; "
                  f"вне top-10 MVP у {item['candidate_outside_top10']}. Confidence относится только к кандидату R1."]
    lines += ["", "## Артефакты и ограничения", "",
              "- В каждом tasks/*/export: submission.csv (10 ID при любом отказе), candidates.csv, embeddings.npy, embedding_order.json.",
              "- Настоящие float32 признаки: столбцы [0:512] MVP, [512:2048] R1. Блоки единичные, общая норма sqrt(2). Глобальная L2 не обязательна по Q&A №8.",
              "- Сохранённый NPY воспроизводит решения через общий dual-role scorer. Референсный cosine всего вектора не равен финальному ranking: роли описаны отдельно.",
              "- Порог и веса заморожены; выбор кандидата независим от submission (Q&A №13–14, №31). Номерная зона не обрабатывается, bbox/validation не исправляются.",
              "- R1-реранкер не нужен для выбора raw-кандидата. Ranking использует неизменённый MVP legacy20/3/0.50.",
              "- Cached этапы не являются новыми измерениями скорости. Fresh validation использует изображения; при возобновлении готовый этап отмечен CACHED.",
              "- Junk/top-10: исходный evaluator не меняется. Десять записанных ID не позволяют восстановить 11-е место после удаления junk.",
              "- Только CPU-прототип качества. Нет проверки официальной GPU-скорости, Linux, номерного сигнала или полной готовности к сдаче.",
              "- dual_role_profile.json ссылается на веса в исследовательском репозитории: это не переносимая поставка. MVP и приложение не переключаются.",
              f"- Защищённые файлы неизменны: {result.get('protected_unchanged', False)}; promoted=False. Время текущего вызова: {result.get('elapsed_seconds', 0)/60:.1f} мин."]
    (context["output"] / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(context, *, allow_outer=False):
    if not allow_outer:
        raise ValueError("Explicit allow_outer=True is required for this development check")
    if runtime() != context["manifest"]["analysis_runtime"]:
        raise old.IntegrityError("Use the frozen runtime; do not mix library versions")
    queue = old.Queue(context, wall_hours=None)
    result = {"status": "running", "signature": context["signature"], "evaluations": {}, "optimizer_updates": 0,
              "bn_updates": 0, "threshold_fit": False, "promoted": False}
    with old.review.old.run_lock(context["output"]):
        try:
            old.review.check_inputs(context, rehash=True)
            frozen_profile(context)
            for stage, split in enumerate(("calibration", "validation"), 1):
                print(f"\nSTAGE {stage}/4: {split}, cached source blocks -> shared export", flush=True)
                values = load_vectors(context, split)
                name = f"cached_{split}"
                result["evaluations"][name] = queue.task(name, lambda d: evaluate(context, split, values, d))
                if result["evaluations"][name] is None:
                    raise ValueError(f"Incomplete {name}; inspect task error before resuming")
            print("\nSTAGE 3/4: fresh validation images, four original ONNX encoders, batch16", flush=True)
            result["evaluations"]["fresh_validation"] = queue.task("fresh_validation", lambda d: fresh_validation(context, values, d))
            if result["evaluations"]["fresh_validation"] is None:
                raise ValueError("Incomplete fresh validation; inspect task error before resuming")
            current = result["evaluations"]["fresh_validation"]
            mvp = source_task(context, "evaluate_MVP_active")
            result["comparison"] = {"mAP_delta": current["ranking"]["mAP@10"]-mvp["ranking"]["mAP@10"],
                                    "C_delta": current["candidates"]["C"]-mvp["candidates"]["C"],
                                    "quality_points_delta": current["quality_points_out_of_55"]-(
                                        45*mvp["ranking"]["mAP@10"]+10*mvp["candidates"]["C"])}
            print("\nSTAGE 4/4: exact decision parity verified; report; MVP unchanged", flush=True)
            result["status"] = "complete"
        except BaseException:
            result["status"] = "incomplete"
            raise
        finally:
            result.update(events=queue.events, elapsed_seconds=time.monotonic()-queue.started)
            try:
                old.review.check_inputs(context, rehash=True)
                frozen_profile(context)
                result["protected_unchanged"] = True
            except BaseException:
                result.update(status="integrity_check_failed", protected_unchanged=False)
                raise
            finally:
                write_json(context["output"] / "results.json", result)
                write_report(context, result)
    return result
