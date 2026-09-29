"""Read-only encoders, primary-only policy selection, explicit final evaluation."""
import gc
import json
import tempfile
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from backend.core import ARTIFACTS, ROOT, sha256
from training.audit import digest
from training import final_seed_confirmation as seeds
from training import frozen_inference as frozen
from training import policy_inference as inference
from training import retrieval_policy as policy
from training.stage6 import write_json


review = seeds.review
VARIANT = ROOT / "OSNet-AIN-x1.0/variant_18_retrieval_policy"
SYSTEMS = (*seeds.NAMES, "R1_equal3")
SOURCES = ("retrieval_policy", "policy_inference", "retrieval_policy_experiment")


def prepare(run_name="policy_v1", source_run="review_v1", seed_run="final_seeds_v1"):
    source, _, protected = seeds.load_source(source_run)
    previous = seeds.run_directory(seeds.VARIANT, seed_run)
    manifest17 = review.old.load_json(previous / "manifest.json")
    receipt = seeds.verify_receipt({"output": previous, "signature": digest(manifest17)}, previous / "complete.json")
    result17 = review.old.load_json(previous / "results.json")
    if (manifest17["extension"]["source_signature"] != source["signature"]
            or result17["signature"] != digest(manifest17) or not result17["complete"]):
        raise ValueError("Source runs do not describe the same completed experiment")
    review.check_inputs({"manifest": manifest17, "protected": manifest17["protected"]})
    if seeds.runtime(source["device"]) != source["manifest"]["runtime"]:
        raise ValueError("Preserve source runtime, device, threads and determinism")
    protected.update({str(previous / p): h for p, h in receipt["artifacts"].items()})
    protected.update({str(previous / p): sha256(previous / p) for p in ("manifest.json", "complete.json")})
    manifest = {**source["manifest"], "version": 18,
                "source_sha256": {**manifest17["source_sha256"],
                    **{f"training/{n}.py": sha256(ROOT / "training" / f"{n}.py") for n in SOURCES}},
                "protected": {**source["protected"], **protected},
                "retrieval_plan": {"source_signature": source["signature"], "seed_signature": digest(manifest17),
                    "policies": policy.POLICIES, "systems": list(SYSTEMS), "step": 800,
                    "selection": "primary regular draws mean within seed, then all three seeds; policy order breaks ties",
                    "alternate": "frozen selected policy plus raw/legacy controls; no selection",
                    "candidate_policies": list(policy.CANDIDATES), "ensemble": "equal three R1 seeds; no member selection",
                    "final_provider": "CPUExecutionProvider", "training_updates": 0,
                    "checkpoint_phase": "primary diagnostic only; never replaces step800 final rule",
                    "outer_status": "already observed development data, never used for selection"}}
    output = seeds.run_directory(VARIANT, run_name)
    context = {**source, "source": source, "seed_output": previous, "seed_results": result17,
               "output": output, "manifest": manifest, "signature": digest(manifest),
               "protected": manifest["protected"],
               "confirmation": review.old.load_json(source["output"] / "confirm_inner.json")}
    review.check_other_runs(context)
    with review.old.run_lock(output):
        review.old.freeze_json(output / "manifest.json", manifest)
    return context


def cached_report(context, relative, compute):
    path = context["output"] / relative
    if path.exists():
        saved = review.old.load_json(path)
        if saved["signature"] != context["signature"] or saved["digest"] != digest(saved["result"]):
            raise ValueError(f"Changed cached report: {relative}")
        return saved["result"]
    result = compute()
    write_json(path, {"signature": context["signature"], "digest": digest(result), "result": result})
    return result


def cached_vectors(context, relative, rows, compute):
    path = context["output"] / "cache" / f"{relative}.npz"
    receipt = path.with_suffix(".json")
    ids = [r["image_id"] for r in rows]
    if receipt.exists():
        saved = review.old.load_json(receipt)
        if (saved["signature"] != context["signature"] or saved["ids"] != ids
                or sha256(path) != saved["sha256"]):
            raise ValueError(f"Changed embedding cache: {relative}")
        with np.load(path, allow_pickle=False) as data:
            values = data["vectors"]
    else:
        print(f"Encoding {relative}: {len(rows)} images", flush=True)
        values = np.asarray(compute(), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(rows) or not np.isfinite(values).all():
            raise ValueError("Invalid cached embeddings")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pending.npz")
        np.savez_compressed(temporary, vectors=values)
        temporary.replace(path)
        write_json(receipt, {"signature": context["signature"], "ids": ids, "sha256": sha256(path)})
    return dict(zip(ids, values))


def inner_vectors(context, fold, name, seed, step=800):
    protocols = context["manifest"]["draws"][fold]
    ids = {i for p in protocols.values() for k in ("query_ids", "gallery_ids") for i in p[k]}
    rows = [r for r in context["rows"] if r["image_id"] in ids]
    summary = next(s for s in context["confirmation"][fold] if s["variant"] == name and s["seed"] == seed)
    if str(step) not in summary["checkpoints"]:
        raise ValueError("Requested checkpoint was not saved; new training needs a separate experiment")

    def encode():
        model, variant = review.load_model(context["source"], {**summary, "stop_step": step})
        try:
            result = review.old.encode(model, rows, context["source"], variant)
            return np.stack([result[r["image_id"]] for r in rows])
        finally:
            del model
            gc.collect()

    return cached_vectors(context, f"{fold}/{name}/{seed}/step_{step}", rows, encode)


def combine_encoded(encoded):
    ids = list(encoded[0])
    if any(set(e) != set(ids) for e in encoded):
        raise ValueError("Ensemble has different image IDs")
    values = policy.combine_members([np.stack([e[i] for i in ids]) for e in encoded])
    return dict(zip(ids, values))


def rows_for(context, protocol):
    by_id = {r["image_id"]: r for r in context["rows"]}
    return [[by_id[i] for i in protocol[k]] for k in ("query_ids", "gallery_ids")]


def measure(queries, gallery, encoded, setting):
    qv, gv = (np.stack([encoded[r["image_id"]] for r in rows]) for rows in (queries, gallery))
    ranking = policy.rank_vectors(qv, gv, setting)
    q, g = policy.frames(queries, gallery)
    predictions, _ = policy.predictions(queries, gallery, ranking, 2., "raw_top1")
    return {"ranking": policy.official.ranking_metrics(q, g, predictions),
            **policy.query_diagnostics(queries, gallery, ranking),
            "gallery": policy.gallery_diagnostics(gallery, ranking["graph"])}


def evaluate_inner(context, fold, system, seed, encoded, settings, step=800):
    reports = {}
    for draw, protocol in context["manifest"]["draws"][fold].items():
        if fold == "primary" and not protocol["selection_eligible"]:
            continue
        queries, gallery = rows_for(context, protocol)
        reports[draw] = {}
        for setting in settings:
            relative = f"reports/{fold}/{system}/{seed}/step_{step}/{draw}/{setting}.json"
            reports[draw][setting] = cached_report(context, relative,
                lambda: measure(queries, gallery, encoded, setting))
        print(f"{fold}/{system}/{seed}/step{step}/{draw}: complete", flush=True)
    return reports


def summarize_matrix(matrix, setting):
    by_seed = {str(seed): mean(report[setting]["ranking"]["mAP@10"]
                             for draw, report in draws.items() if draw.startswith("regular_"))
               for seed, draws in matrix.items()}
    return {"mean": mean(by_seed.values()), "seed_values": by_seed,
            "std": stdev(by_seed.values()) if len(by_seed) > 1 else None,
            "unit": "training seed" if len(by_seed) > 1 else "one fixed three-encoder ensemble"}


def select_policies(primary):
    selected = {}
    for system in SYSTEMS:
        scores = {p: summarize_matrix(primary[system], p) for p in policy.POLICIES}
        selected[system] = {"policy": max(scores, key=lambda p: scores[p]["mean"]), "scores": scores, "step": 800}
    return selected


def run_inner(context):
    primary, alternate = {}, {}
    for fold, destination in (("primary", primary), ("alternate", alternate)):
        if fold == "alternate":
            selection = review.old.load_json(context["output"] / "selection.json")
        r1 = []
        for system in seeds.NAMES:
            destination[system] = {}
            settings = list(policy.POLICIES) if fold == "primary" else list(dict.fromkeys(
                ["legacy", "raw", selection[system]["policy"]]))
            for seed in context["seeds"]:
                encoded = inner_vectors(context, fold, system, seed)
                destination[system][str(seed)] = evaluate_inner(context, fold, system, seed, encoded, settings)
                if system == "R1_resolution256":
                    r1.append(encoded)
        settings = list(policy.POLICIES) if fold == "primary" else list(dict.fromkeys(
            ["legacy", "raw", selection["R1_equal3"]["policy"]]))
        destination["R1_equal3"] = {"equal3": evaluate_inner(context, fold, "R1_equal3", "equal3", combine_encoded(r1), settings)}
        if fold == "primary":
            # Commit before accessing ANY alternate report or alternate embeddings.
            review.old.freeze_json(context["output"] / "selection.json", select_policies(primary))
    selection = review.old.load_json(context["output"] / "selection.json")
    summary = {}
    for fold, matrix in (("primary", primary), ("alternate", alternate)):
        summary[fold] = {}
        for system in SYSTEMS:
            first_draw = next(iter(next(iter(matrix[system].values())).values()))
            summary[fold][system] = {p: summarize_matrix(matrix[system], p) for p in first_draw}
    overlaps = {fold: {draw: policy.error_overlap({seed: reports[draw]["raw"] for seed, reports in matrix["R1_resolution256"].items()})
                       for draw in next(iter(matrix["R1_resolution256"].values()))}
                for fold, matrix in (("primary", primary), ("alternate", alternate))}
    return {"selection": selection, "summary": summary, "error_overlap": overlaps,
            "source_inner_selection": context["confirmation"]["selection"],
            "source_alternate": context["confirmation"]["alternate_summary"],
            "training_updates": 0, "outer_evaluated": False, "promoted": False}


def run_checkpoints(context):
    scores = {}
    for name in seeds.NAMES:
        summaries = [s for s in context["confirmation"]["primary"] if s["variant"] == name]
        steps = sorted(set.intersection(*(set(s["checkpoints"]) for s in summaries)), key=int)
        scores[name] = {}
        for step in steps:
            matrix = {str(seed): evaluate_inner(context, "primary", name, seed,
                        inner_vectors(context, "primary", name, seed, int(step)), list(policy.POLICIES), int(step))
                      for seed in context["seeds"]}
            scores[name][step] = {p: summarize_matrix(matrix, p) for p in policy.POLICIES}
    proposals = {name: min(((step, p) for step in steps for p in policy.POLICIES),
                           key=lambda x: (-steps[x[0]][x[1]]["mean"], int(x[0]), list(policy.POLICIES).index(x[1])))
                 for name, steps in scores.items()}
    return {"scores": scores, "primary_proposals_only": proposals, "changes_final_recipe": False,
            "note": "Alternate/final saved only step800. Other steps need separately authorized refit/confirmation.",
            "training_updates": 0, "outer_evaluated": False, "promoted": False}


def source_bundle(context, name, seed):
    root = context["source"]["output"] if seed == context["seeds"][0] else context["seed_output"]
    return root / "final" / name / f"seed_{seed}" / "bundle.json"


def final_vectors(context, name, seed, split):
    query, gallery = review.old.protocol_rows(context["source"], split)
    rows = query + gallery
    def encode():
        encoder = frozen.FrozenEncoder(source_bundle(context, name, seed), "CPUExecutionProvider")
        return frozen._encode_rows(encoder, rows, context["dataset"], 16)
    return cached_vectors(context, f"onnx/{split}/{name}/{seed}", rows, encode)


def final_systems(context, split):
    encoded, paths = {}, {}
    for name in seeds.NAMES:
        for seed in context["seeds"]:
            key = f"{name}_{seed}"
            encoded[key] = final_vectors(context, name, seed, split)
            paths[key] = [source_bundle(context, name, seed)]
    keys = [f"R1_resolution256_{s}" for s in context["seeds"]]
    encoded["R1_equal3"] = combine_encoded([encoded[k] for k in keys])
    paths["R1_equal3"] = [paths[k][0] for k in keys]
    return encoded, paths


def rank_encoded(query, gallery, encoded, setting):
    return policy.rank_vectors(*(np.stack([encoded[r["image_id"]] for r in rows]) for rows in (query, gallery)), setting)


def verify_or_export_csv(output, query, gallery, ranking, threshold, candidate_policy):
    if output.exists():
        ordered, accepted = policy.predictions(query, gallery, ranking, threshold, candidate_policy)
        loaded = policy.official.load_submission(output / "submission.csv", {r["image_id"] for r in gallery})
        candidates = policy.official.load_candidates(output / "candidates.csv")
        if loaded != ordered or set(candidates) != set(accepted) or any(
            candidates[q][0][0] != accepted[q][0][0] or not np.isclose(candidates[q][0][1], accepted[q][0][1], atol=1e-15, rtol=0)
            for q in accepted):
            raise ValueError("Existing official CSV differs from frozen policy")
        q, g = policy.frames(query, gallery)
        return {"ranking": policy.official.ranking_metrics(q, g, loaded), "candidates": policy.candidate_metrics(q, g, candidates)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="csv_pending_", dir=output.parent) as temporary:
        source = Path(temporary) / "csv"
        result = policy.export_csv(source, query, gallery, ranking, threshold, candidate_policy)
        source.rename(output)
    return result


def run_final(context):
    seeds.verify_receipt(context, context["output"] / "inner_complete.json")
    selection = review.old.load_json(context["output"] / "selection.json")
    query, gallery = review.old.protocol_rows(context["source"], "calibration")
    calibration, members = final_systems(context, "calibration")
    frozen_cases = []
    for key, encoded in calibration.items():
        system = next(n for n in SYSTEMS if key == n or key.startswith(n + "_"))
        for setting in dict.fromkeys(["legacy", selection[system]["policy"]]):
            ranking = rank_encoded(query, gallery, encoded, setting)
            for candidate in policy.CANDIDATES:
                case = f"{key}/{setting}/{candidate}"
                cal = cached_report(context, f"calibration/{case}.json", lambda:
                    policy.calibrate_policy(query, gallery, ranking, candidate, split="calibration"))
                threshold = cal["selected"]["threshold"]
                bundle_path = context["output"] / "final" / case / "bundle.json"
                inference.write_bundle(bundle_path, members[key], ranking=setting, candidate_policy=candidate,
                    threshold=threshold, calibration={"split": "calibration", "candidate_policy": candidate,
                    "method": "maximize C=0.7F1+0.3TNR; ties F1 then higher threshold",
                    "protocol_sha256": digest(context["manifest"]["protocols"]["calibration"])})
                frozen_cases.append({"case": case, "key": key, "system": system, "ranking_policy": setting,
                                     "candidate": candidate, "threshold": threshold,
                                     "bundle_sha256": sha256(bundle_path)})
    # No validation encoding/metrics before every calibration decision is frozen.
    review.old.freeze_json(context["output"] / "final_selection.json", frozen_cases)
    validation, _ = final_systems(context, "validation")
    query, gallery = review.old.protocol_rows(context["source"], "validation")
    results = []
    for case in frozen_cases:
        encoded = validation[case["key"]]
        def evaluate_case():
            ranking = rank_encoded(query, gallery, encoded, case["ranking_policy"])
            result = verify_or_export_csv(context["output"] / "final" / case["case"] / "csv", query, gallery,
                                          ranking, case["threshold"], case["candidate"])
            expected = policy.evaluate(query, gallery, ranking, case["threshold"], case["candidate"])
            if result != expected:
                raise ValueError("Official CSV metrics differ from in-memory policy")
            return {**result, **policy.query_diagnostics(query, gallery, ranking)}
        result = cached_report(context, f"final/{case['case']}/evaluation.json", evaluate_case)
        results.append({**case, **result})
        print(f"Official CSV checked: {case['case']}", flush=True)
    return {"cases": results, "training_updates": 0, "outer_evaluated": True, "promoted": False,
            "inference": "ONNX CPU, batch16; no GPU or speed verification", "baseline": review.old.load_json(ARTIFACTS / "baseline_metrics.json"),
            "note": "No best-seed/system selected on outer. Ensemble is one system, not three independent trials."}


def report_text(phase, result):
    lines = [f"# Retrieval policy: {phase}", "", "Обучение encoder: 0 updates. MVP не изменён.", ""]
    if phase == "inner":
        lines += ["Выбор по primary; alternate не выбирает параметры. Шаг 800 фиксирован.", "",
                  "| Система | Политика | Primary legacy → selected | Alternate legacy → selected |", "|---|---|---:|---:|"]
        for name in SYSTEMS:
            chosen = result["selection"][name]["policy"]
            cells = [" → ".join(f"{100*result['summary'][fold][name][p]['mean']:.3f}%" for p in ("legacy", chosen)) for fold in ("primary", "alternate")]
            lines.append(f"| {name} | {chosen} | {' | '.join(cells)} |")
        lines += ["", "Одиночные модели: сначала среднее трёх draws внутри seed, затем среднее трёх seed.",
                  "Equal3 — один ансамбль, не среднее метрик. Дисперсия по draws не объявляется дисперсией по seed.",
                  "Per-query AP, top-1 transitions, confidence, gallery-neighbor diagnostics и overlap находятся в JSON reports.",
                  "Следующий final запускается только после отдельного разбора этого отчёта."]
    elif phase == "final":
        lines += ["Метрики прочитаны официальным evaluator из CSV. Значения — проценты; mean ± sample SD по трём seed.",
                  "Для equal3 показан один ансамбль, без фиктивного n=3.", "",
                  "| Система | Ranking | Candidate | mAP@10 | F1 | TNR | C |", "|---|---|---|---:|---:|---:|---:|"]
        groups = {}
        for item in result["cases"]:
            groups.setdefault((item["system"], item["ranking_policy"], item["candidate"]), []).append(item)
        for (system, setting, candidate), items in groups.items():
            values = [mean(x["ranking"]["mAP@10"] for x in items)]
            values += [mean(x["candidates"][m] for x in items) for m in ("F1", "TNR", "C")]
            individual = [[x["ranking"]["mAP@10"], *(x["candidates"][m] for m in ("F1", "TNR", "C"))] for x in items]
            cells = [f"{100*v:.3f}" + (f" ± {100*stdev(row[i] for row in individual):.3f}" if len(items) > 1 else "")
                     for i, v in enumerate(values)]
            lines.append(f"| {system} | {setting} | {candidate} | " + " | ".join(cells) + " |")
        if "validation" in result["baseline"]:
            b = result["baseline"]["validation"]
            lines.append("| MVP, historical | legacy | ranking_top1 | " + " | ".join(f"{100*b[k]:.3f}" for k in
                ("mAP_at_10", "candidate_F1", "TNR", "candidate_score")) + " |")
        lines += ["", "C = 0.7 F1 + 0.3 TNR. Кандидатский блок весит 10%, не полный балл соревнования.",
                  "Outer уже использовалась: это development, не независимый тест. Ни seed, ни победитель по ней не выбираются.",
                  "Три encoder требуют трёх forward; официальный GPU benchmark не выполнен."]
    else:
        lines += [result["note"], "", json.dumps(result["primary_proposals_only"], ensure_ascii=False, indent=2)]
    return "\n".join(lines) + "\n"


def run(context, phase="inner", allow_outer=False):
    if phase not in {"inner", "checkpoints", "final"}:
        raise ValueError("Unknown experiment phase")
    if phase == "final" and not allow_outer:
        raise ValueError("Explicit ALLOW_OUTER_EVALUATION=True required")
    if seeds.runtime(context["device"]) != context["manifest"]["runtime"]:
        raise ValueError("Runtime changed since preflight")
    output = context["output"]
    with review.old.run_lock(VARIANT / "runs"), review.old.run_lock(output):
        review.check_other_runs(context)
        review.check_inputs(context, rehash=True)
        receipt = output / f"{phase}_complete.json"
        if receipt.exists():
            seeds.verify_receipt(context, receipt)
            return review.old.load_json(output / f"{phase}.json")
        result = {"signature": context["signature"], **{"inner": run_inner, "checkpoints": run_checkpoints, "final": run_final}[phase](context)}
        review.check_inputs(context, rehash=True)
        write_json(output / f"{phase}.json", result)
        (output / f"{phase.upper()}_RESULTS.md").write_text(report_text(phase, result), encoding="utf-8")
        artifacts = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                     if p.is_file() and p.name != ".lock" and ".pending" not in p.name}
        write_json(receipt, {"signature": context["signature"], "artifacts": artifacts})
        return result
