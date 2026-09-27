"""v20 Run All: isolated quality experiments, never outer/final/deployment work."""
import base64
import gc
import html
import io
import time
from dataclasses import asdict
from statistics import mean

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from backend.core import ROOT, bbox, crop_image, normalize, sha256
from training import overnight_system as old
from training import overnight_evaluation as evaluation
from training import overnight_training as training
from training import quality_clock as clock
from training import quality_verifier as verifier
from training.audit import digest
from training.local_verification import extract_local_tokens, mix_topk_scores, topk_oracle_diagnostics
from training.osnet_ablations import AblationDataset
from training.stage6 import write_json


VARIANT = ROOT / "OSNet-AIN-x1.0/variant_20_quality"
HEAD_WEIGHTS = {"baseline": 0., "head_w05": .05, "head_w10": .10}
MODULES = ("quality_experiment", "quality_verifier", "quality_clock", "overnight_system",
           "overnight_evaluation", "overnight_training", "local_verification")


def prepare(run_name="quality_v1", *, source_run="review_v1", policy_run="policy_v1",
            run_head=True, run_clock=True, auto_confirm=True):
    """Reconstruct existing evidence read-only; write ONLY into a new v20 run."""
    source, _, protected = old.seeds.load_source(source_run)
    if old.seeds.runtime(source["device"]) != source["manifest"]["runtime"]:
        raise old.IntegrityError("Use the research kernel and original device/versions/threads/determinism")
    if source["device"].type == "mps" and not torch.backends.mps.is_available():
        raise old.IntegrityError("Source MPS device unavailable; CPU fallback would change the controlled experiment")
    previous = old.seeds.run_directory(old.previous.VARIANT, policy_run)
    policy_manifest = old.read(previous / "manifest.json")
    policy_signature = digest(policy_manifest)
    if policy_manifest["retrieval_plan"]["source_signature"] != source["signature"]:
        raise old.IntegrityError("v18 and v16 source experiments differ")
    receipt = old.seeds.verify_receipt({"output": previous, "signature": policy_signature}, previous / "checkpoints_complete.json")
    protected.update(policy_manifest["protected"])
    protected.update({str(previous / name): value for name, value in receipt["artifacts"].items()})
    for name in ("manifest.json", "checkpoints_complete.json"):
        protected[str(previous / name)] = sha256(previous / name)
    plan = {"run_head": run_head, "run_clock": run_clock, "auto_confirm": auto_confirm,
            "head": asdict(verifier.HeadConfig()), "head_weights": HEAD_WEIGHTS,
            "clock_jobs": {name: asdict(job) for name, job in clock.jobs().items()},
            "clock_cases": list(clock.CASES), "head_gate": .002, "clock_pilot_gate": .003,
            "confirmation_gate": .002, "outer_evaluation": False, "promotion": False,
            "external_data": False, "annotation_edits": 0, "wall_time_limit": None,
            "pair_training": "same-fold encoder train identities only; cross-camera positive, real top50 negatives",
            "averaging": "full1700 last three fixed checkpoints; train-only BN, matched BN-only control",
            "selection": "primary only; alternate only confirms the frozen choice; all three seeds equally",
            "candidate": "pair head changes ranking only; clock C is separately calibrated inner diagnostic"}
    manifest = {**source["manifest"], "version": 20, "quality_plan": plan,
                "policy_source_signature": policy_signature,
                "protected": {**source["protected"], **protected},
                "source_sha256": {**policy_manifest["source_sha256"],
                    **{f"training/{n}.py": sha256(ROOT / "training" / f"{n}.py") for n in MODULES}}}
    output = old.seeds.run_directory(VARIANT, run_name)
    context = {**source, "source": source, "output": output, "old_output": previous,
               "manifest": manifest, "signature": digest(manifest), "protected": manifest["protected"],
               "confirmation": old.read(source["output"] / "confirm_inner.json")}
    old.review.check_inputs(context)
    old.review.check_other_runs(context)
    with old.review.old.run_lock(output):
        old.review.old.freeze_json(output / "manifest.json", manifest)
    return context


def cached_features(context, fold, seed, role):
    if role not in {"train", "evaluation"}:
        raise ValueError("Invalid feature role")
    allowed = training.fold_training_ids(context, fold)
    rows = (verifier.training_rows(context["rows"], allowed, seed=seed,
                                  per_identity=context["manifest"]["quality_plan"]["head"]["images_per_identity"])
            if role == "train" else old.fold_rows(context, fold))
    identities = {r["vehicle_id"] for r in rows}
    if (role == "train" and not identities <= allowed) or (role == "evaluation" and identities & allowed):
        raise old.IntegrityError("Pair train/evaluation identity leakage")
    summary = old.source_summary(context, fold, seed)
    specification = {"signature": context["signature"], "checkpoint": summary["checkpoints"]["800"],
                     "rows": rows, "grid": [4, 4], "role": role}
    path = context["output"] / "cache" / fold / str(seed) / f"{role}.npz"
    receipt = path.with_suffix(".json")
    if receipt.exists():
        saved = old.read(receipt)
        if saved["specification"] != specification or sha256(path) != saved["sha256"]:
            raise old.IntegrityError("Feature cache changed")
        with np.load(path, allow_pickle=False) as values:
            vectors, tokens = values["vectors"], values["tokens"]
    else:
        model, variant = old.review.load_model(context["source"], {**summary, "stop_step": 800})
        vectors, tokens = [], []
        loader = DataLoader(AblationDataset(rows, variant, context["dataset"]), batch_size=32, shuffle=False, num_workers=0)
        try:
            with torch.no_grad():
                for index, (batch, _, _) in enumerate(loader):
                    batch = batch.to(context["device"])
                    tokens.append(extract_local_tokens(model.backbone, batch))
                    vectors.append(normalize(model.embedding(batch).cpu().numpy()))
                    if index % 10 == 0 or (index + 1) * 32 >= len(rows):
                        print(f"  features {fold}/{seed}/{role}: {min((index+1)*32,len(rows))}/{len(rows)}", flush=True)
            vectors, tokens = np.concatenate(vectors), np.concatenate(tokens)
        finally:
            del model
            gc.collect()
        if role == "evaluation":
            reference = old.source_vectors(context, fold, seed)
            if not np.allclose(vectors, np.stack([reference[r["image_id"]] for r in rows]), rtol=0, atol=2e-5):
                raise old.IntegrityError("Frozen R1 no longer reproduces v18 features")
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".pending.npz")
        np.savez_compressed(pending, vectors=vectors, tokens=tokens)
        pending.replace(path)
        write_json(receipt, {"specification": specification, "sha256": sha256(path)})
    if vectors.shape != (len(rows), 512) or tokens.shape != (len(rows), 16, 512) or not all(np.isfinite(x).all() for x in (vectors, tokens)):
        raise old.IntegrityError("Invalid local feature arrays")
    return rows, vectors, tokens


def train_head(context, fold, seed, directory):
    rows, vectors, tokens = cached_features(context, fold, seed, "train")
    config = verifier.HeadConfig(**context["manifest"]["quality_plan"]["head"])
    pairs, labels = verifier.mine_pairs(rows, vectors, config)
    features = np.empty((len(pairs), 289), dtype=np.float32)
    for i in np.unique(pairs[:, 0]):
        positions = np.flatnonzero(pairs[:, 0] == i)
        js = pairs[positions, 1]
        features[positions] = verifier.pair_features(tokens[i], tokens[js], vectors[js] @ vectors[i])
    # Train and evaluate identities are disjoint for encoder AND head; no OOF claim.
    result = verifier.fit_head(features, labels, directory, seed=seed,
                               signature=digest([context["signature"], fold]), config=config)
    return {**result, "fold": fold, "seed": seed, "train_images": len(rows),
            "train_identities": sorted({r["vehicle_id"] for r in rows}),
            "encoder_train_features": True, "scope": "supervised train pairs; evaluation identities never seen"}


def ranking_report(query, gallery, ranking):
    result = old.policy.evaluate(query, gallery, ranking, 2., "raw_top1")["ranking"]
    diagnostics = old.policy.query_diagnostics(query, gallery, ranking)["per_query"]
    return {"ranking": result, "per_query": {qid: {"vehicle_id": value["vehicle_id"], "ap": value["ranking"]["ap"]}
                                              for qid, value in diagnostics.items()}}


def visual_errors(query, gallery, order, diagnostic, dataset, path, limit=16):
    """Small inspectable HTML of real organizer crops; never changes input files."""
    def thumbnail(row):
        with Image.open(dataset / "images" / f"{row['image_id']}.jpg") as image:
            crop = crop_image(image, bbox(row)).convert("RGB")
            crop.thumbnail((150, 110))
            buffer = io.BytesIO()
            crop.save(buffer, format="JPEG")
        return '<img src="data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode() + '">'
    cases = [(i, diagnostic["per_query"][r["image_id"]]) for i, r in enumerate(query)]
    cases = sorted((x for x in cases if x[1]["known"] and x[1]["baseline"]["AP10"] < 1),
                   key=lambda x: x[1]["baseline"]["AP10"])[:limit]
    lines = ['<!doctype html><meta charset="utf-8"><title>v20 error review</title>',
             '<h1>Внутренние ошибки: query и первые 5 результатов</h1>',
             '<p>Только диагностика. Исходные bbox, изображения и разметка не изменены. Oracle — не достигнутая метрика.</p>']
    for i, entry in cases:
        row = query[i]
        lines.append(f'<h3>{html.escape(row["image_id"])}: AP={entry["baseline"]["AP10"]:.3f}; '
                     f'oracle@50={entry["pools"]["50"]["AP10_oracle"]:.3f}</h3><div>{thumbnail(row)}')
        for j in order[i, :5]:
            item = gallery[int(j)]
            label = "совпадение" if item["vehicle_id"] == row["vehicle_id"] else "другая машина"
            lines.append(f'<span style="display:inline-block;text-align:center">{thumbnail(item)}<br>{label}</span>')
        lines.append('</div>')
    path.write_text("\n".join(lines), encoding="utf-8")


def head_trial(context, fold, seed, head_summary, directory, names):
    rows, vectors, tokens = cached_features(context, fold, seed, "evaluation")
    if {r["vehicle_id"] for r in rows} & set(head_summary["train_identities"]):
        raise old.IntegrityError("Head has seen evaluation identities")
    head = verifier.load_head(head_summary)
    reports = {name: {"draws": {}} for name in names}
    for draw, query, gallery, qv, gv, (qt, gt) in old.draw_arrays(context, fold, rows, vectors, tokens):
        baseline, scores = evaluation.rank_with_scores(qv, gv)
        count = min(50, len(gallery))
        local = np.stack([verifier.score_pairs(head, qt[i], gt[pool], gv[pool] @ qv[i])
                          for i, pool in enumerate(baseline["order"][:, :count])])
        for name in names:
            mixed = mix_topk_scores(scores, baseline["order"], local, top_k=50, weight=HEAD_WEIGHTS[name])
            ranking = {**baseline, "order": mixed["order"]}
            report = ranking_report(query, gallery, ranking)
            oracle = topk_oracle_diagnostics(query, gallery, baseline["order"], ks=(50,), comparison_order=ranking["order"])
            report["oracle"] = oracle
            # The candidate branch is raw_top1 and confidence is unchanged for ALL thresholds.
            assert np.array_equal(ranking["raw_order"], baseline["raw_order"])
            assert np.array_equal(ranking["confidence"], baseline["confidence"])
            report["candidate_unchanged_for_any_threshold"] = True
            reports[name]["draws"][draw] = report
            if name == "baseline":
                visual_errors(query, gallery, baseline["order"], oracle, context["dataset"], directory / f"errors_{draw}.html")
        print(f"  {fold}/{seed}/{draw}: " + ", ".join(f"{n}={reports[n]['draws'][draw]['ranking']['mAP@10']:.5f}" for n in names), flush=True)
    for report in reports.values():
        report["mean_map"] = evaluation.summarize_draws(report["draws"])
    return reports


def diagnose(context, directory):
    """Read existing inner R1 weights before any new optimization."""
    seed = context["seeds"][0]
    rows, vectors, tokens = cached_features(context, "primary", seed, "evaluation")
    reports = {}
    for draw, query, gallery, qv, gv, _ in old.draw_arrays(context, "primary", rows, vectors):
        ranking = old.policy.rank_vectors(qv, gv, "less_graph")
        report = topk_oracle_diagnostics(query, gallery, ranking["order"], ks=(50,),
                                         comparison_order=ranking["raw_order"])
        known = [x for x in report["per_query"].values() if x["known"]]
        report["error_counts"] = {
            "known_queries": len(known),
            "no_valid_positive_in_top50": sum(not x["pools"]["50"]["any_positive"] for x in known),
            "reordering_headroom": sum(x["pools"]["50"]["AP10_oracle"] > x["baseline"]["AP10"] + 1e-12 for x in known),
            "graph_hurt_vs_raw": sum(x["baseline"]["AP10"] < x["comparison"]["AP10"] - 1e-12 for x in known),
            "graph_helped_vs_raw": sum(x["baseline"]["AP10"] > x["comparison"]["AP10"] + 1e-12 for x in known)}
        report["note"] = "baseline=less_graph; comparison=raw; flags can overlap; repeated draws not new identities"
        reports[draw] = report
        visual_errors(query, gallery, ranking["order"], report, context["dataset"], directory / f"errors_{draw}.html")
        print(f"  diagnostics {draw}: {report['error_counts']}", flush=True)
    return {"draws": reports, "scope": "primary held-out train identities only; no new fitting"}


def paired_comparison(matrix, candidate, control="baseline", minimum=.002):
    seeds = list(matrix.get(control, {}))
    if len(seeds) != 3 or set(matrix.get(candidate, {})) != set(seeds) or any(
            matrix[n][s] is None for n in (control, candidate) for s in seeds):
        return {"complete": False, "passed": False, "reason": "Need all three paired seeds"}
    gains = {s: matrix[candidate][s]["mean_map"] - matrix[control][s]["mean_map"] for s in seeds}
    before, after = {}, {}
    for seed in seeds:
        a, b = matrix[control][seed]["draws"], matrix[candidate][seed]["draws"]
        if set(a) != set(b):
            raise ValueError("Unpaired protocol draws")
        for draw in a:
            for target, data in ((before, a), (after, b)):
                target.update({f"{seed}/{draw}/{qid}": value for qid, value in data[draw]["per_query"].items()})
    bootstrap = evaluation.paired_identity_bootstrap(before, after)
    return {"complete": True, "passed": mean(gains.values()) >= minimum and sum(x > 0 for x in gains.values()) >= 2,
            "mean_gain": mean(gains.values()), "seed_gains": gains, "minimum": minimum,
            "conditional_bootstrap": bootstrap}


def select_head(matrix):
    comparisons = {n: paired_comparison(matrix, n) for n in HEAD_WEIGHTS if n != "baseline"}
    if not all(c["complete"] for c in comparisons.values()):
        return {"selected": None, "comparisons": comparisons, "reason": "Incomplete primary; retry same RUN_NAME"}
    eligible = [n for n, c in comparisons.items() if c["passed"]]
    selected = max(eligible, key=lambda n: comparisons[n]["mean_gain"]) if eligible else "baseline"
    return {"selected": selected, "comparisons": comparisons, "rule": "primary mean >= .002; positive on >=2/3 seeds"}


def head_queue(queue):
    context = queue.context
    def one(fold, seed, names):
        head = queue.task(f"head_train_{fold}_{seed}", lambda d: train_head(context, fold, seed, d))
        return None if head is None else queue.task(f"head_eval_{fold}_{seed}", lambda d: head_trial(context, fold, seed, head, d, names))
    primary = {n: {} for n in HEAD_WEIGHTS}
    for seed in context["seeds"]:
        values = one("primary", seed, list(HEAD_WEIGHTS))
        for name in primary:
            primary[name][str(seed)] = None if values is None else values[name]
    decision = select_head(primary)
    if decision["selected"] is not None:
        decision = queue.task("head_selection", lambda d: decision)
    result = {"primary": primary, "selection": decision, "alternate": {}}
    if decision and decision["selected"] not in (None, "baseline") and context["manifest"]["quality_plan"]["auto_confirm"]:
        chosen = decision["selected"]
        alternate = {n: {} for n in ("baseline", chosen)}
        for seed in context["seeds"]:
            values = one("alternate", seed, list(alternate))
            for name in alternate:
                alternate[name][str(seed)] = None if values is None else values[name]
        result.update(alternate=alternate, alternate_gate=paired_comparison(alternate, chosen))
    return result


def evaluate_clock(context, summary, case, directory):
    model, variant, derived = clock.derived_model(context, summary, case, directory)
    rows, reports = old.fold_rows(context, summary["fold"]), {}
    try:
        encoded = old.review.old.encode(model, rows, context, variant)
        vectors = np.stack([encoded[r["image_id"]] for r in rows])
        for draw, query, gallery, qv, gv, _ in old.draw_arrays(context, summary["fold"], rows, vectors):
            ranking = old.policy.rank_vectors(qv, gv, "less_graph")
            reports[draw] = ranking_report(query, gallery, ranking)
            reports[draw]["inner_candidate_diagnostic"] = evaluation.calibrate_inner_confidence(
                query, gallery, ranking, ranking["confidence"], "raw_top1")
        return {"mean_map": evaluation.summarize_draws(reports), "draws": reports, "model": derived,
                "step": summary["stop_step"], "lr_horizon": summary["lr_horizon"], "seed": summary["seed"],
                "candidate_note": "inner identity-separated calibration/evaluation only; not a deployable threshold"}
    finally:
        del model
        gc.collect()


def select_clock(pilots, minimum=.003):
    if any(pilots.get(n) is None for n in clock.CASES):
        return {"selected": None, "reason": "Incomplete primary pilots; retry same RUN_NAME"}
    comparisons = {}
    for name in clock.CASES[1:]:
        controls = ["R1_control"] + (["R1_full1700_bn"] if name.endswith("_avg") else [])
        gains = {c: pilots[name]["mean_map"] - pilots[c]["mean_map"] for c in controls}
        comparisons[name] = {"gains": gains, "passed": all(g >= minimum for g in gains.values()), "controls": controls}
    eligible = [n for n, c in comparisons.items() if c["passed"]]
    selected = max(eligible, key=lambda n: pilots[n]["mean_map"]) if eligible else "R1_control"
    return {"selected": selected, "comparisons": comparisons, "minimum": minimum,
            "scope": "one-seed pilot; not a confirmed win"}


def clock_queue(queue):
    context, summaries = queue.context, {}
    def one(case, seed, fold):
        name = clock.job_name(case)
        key = (name, seed, fold)
        if key not in summaries:
            summaries[key] = queue.task(f"clock_train_{fold}_{name}_{seed}", lambda d: training.fit_job(
                clock.job_context(context, name), clock.jobs()[name], seed, fold=fold))
        summary = summaries[key]
        return None if summary is None else queue.task(f"clock_eval_{fold}_{case}_{seed}",
                                                       lambda d: evaluate_clock(context, summary, case, d))
    first = context["seeds"][0]
    pilots = {case: one(case, first, "primary") for case in clock.CASES}
    decision = select_clock(pilots)
    if decision["selected"] is not None:
        decision = queue.task("clock_selection", lambda d: decision)
    result = {"pilots": pilots, "selection": decision, "primary": {}, "alternate": {}}
    if not decision or decision["selected"] in (None, "R1_control") or not context["manifest"]["quality_plan"]["auto_confirm"]:
        return result
    chosen = decision["selected"]
    controls = decision["comparisons"][chosen]["controls"]
    for case in [*controls, chosen]:
        result["primary"][case] = {str(s): pilots[case] if s == first else one(case, s, "primary") for s in context["seeds"]}
    checks = {c: paired_comparison(result["primary"], chosen, c) for c in controls}
    result["primary_gate"] = checks
    if all(c["passed"] for c in checks.values()):
        for case in [*controls, chosen]:
            result["alternate"][case] = {str(s): one(case, s, "alternate") for s in context["seeds"]}
        result["alternate_gate"] = {c: paired_comparison(result["alternate"], chosen, c) for c in controls}
    return result


def write_report(context, result):
    lines = ["# v20: эксперименты на качество", "", f"Статус: **{result['status']}**.",
             "Нового outer/test mAP нет. MVP, исходная validation и bbox не изменялись.",
             "Пилот/подтверждение — внутренние сравнения, не автоматическое продвижение.",
             "", "## Задачи", "", "| Задача | Статус |", "|---|---|"]
    lines += [f"| {e['task']} | {e['status']} |" for e in result["events"]]
    for family in ("head", "clock"):
        block = result.get(family)
        if not block:
            continue
        lines += ["", f"## {family}", "", f"Выбор: `{(block.get('selection') or {}).get('selected')}`", "",
                  "| Разбиение | Вариант | Seed | mAP@10 |", "|---|---|---:|---:|"]
        for case, value in block.get("pilots", {}).items():
            lines.append(f"| primary pilot | {case} | {context['seeds'][0]} | {value['mean_map']:.5f} |" if value else f"| primary pilot | {case} | — | FAILED |")
        for fold in ("primary", "alternate"):
            for case, seeds in block.get(fold, {}).items():
                for seed, value in seeds.items():
                    lines.append(f"| {fold} | {case} | {seed} | {value['mean_map']:.5f} |" if value else f"| {fold} | {case} | {seed} | FAILED |")
        lines += ["", "Primary/alternate gates и условный bootstrap: см. results.json."]
    lines += ["", "## Как читать результат", "",
              "- Сначала смотреть прирост относительно парного R1-контроля, затем подтверждение на alternate.",
              "- Три seed и повторные query-эпизоды не являются новыми независимыми identity.",
              "- head меняет только ranking; raw-кандидат/отказы не меняются при любом фиксированном пороге.",
              "- Clock C/F1/TNR: отдельная внутренняя диагностика с раздельными identity калибровки/оценки.",
              "- errors_*.html в head_eval содержат реальные crop для визуального разбора. Oracle — только верхняя граница.",
              "- Усреднение сравнивается также с BN-only контролем; параметры не выбираются по outer.",
              "- Номерная зона специально не используется/не обрабатывается; отсутствие остаточного сигнала не доказано.",
              "- Автоматического объединения head и нового encoder, full-train, ONNX-релиза и смены MVP нет.",
              f"- Сохранность источников/изображений: {result.get('protected_unchanged', False)}."]
    (context["output"] / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(context):
    old.review.check_other_runs(context)
    queue = old.Queue(context, wall_hours=None)
    result = {"status": "running", "outer_evaluation": False, "promoted": False}
    with old.review.old.run_lock(context["output"]):
        try:
            old.review.check_inputs(context, rehash=True)
            plan = context["manifest"]["quality_plan"]
            result["diagnostics"] = queue.task("diagnostics_primary", lambda d: diagnose(context, d))
            if plan["run_head"]:
                result["head"] = head_queue(queue)
            if plan["run_clock"]:
                result["clock"] = clock_queue(queue)
            result["status"] = "complete_with_failures" if any(x["status"] == "failed" for x in queue.events) else "complete"
        except BaseException:
            result["status"] = "interrupted_or_integrity_error"
            raise
        finally:
            result.update(events=queue.events, elapsed_seconds=time.monotonic() - queue.started)
            try:
                old.review.check_inputs(context, rehash=True)
                for receipt in (context["output"] / "cache").rglob("*.json"):
                    if sha256(receipt.with_suffix(".npz")) != old.read(receipt)["sha256"]:
                        raise old.IntegrityError("Completed feature cache changed")
                result["protected_unchanged"] = True
            except BaseException:
                result.update(status="integrity_check_failed", protected_unchanged=False)
                raise
            finally:
                write_json(context["output"] / "results.json", result)
                write_report(context, result)
    return result
