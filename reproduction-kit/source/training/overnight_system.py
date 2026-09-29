"""Variant19: resumable, train-only overnight queue. Never promotes the MVP."""
import gc
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.core import ROOT, normalize, sha256
from training import final_seed_confirmation as seeds
from training import retrieval_policy as policy
from training import retrieval_policy_experiment as previous
from training import overnight_evaluation as evaluation
from training.audit import digest
from training.osnet_ablations import AblationDataset
from training.stage6 import write_json


VARIANT = ROOT / "OSNet-AIN-x1.0/variant_19_overnight_system"
MODULES = ("overnight_system", "overnight_evaluation", "overnight_training", "local_verification", "system_benchmark")
review = seeds.review


class IntegrityError(RuntimeError):
    """Corrupted or changed evidence must stop the entire queue, not one trial."""


class BudgetReached(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def prepare(run_name="overnight_v1", source_run="review_v1", policy_run="policy_v1",
            *, training=True, confirmation=True, masks=True, benchmark=True):
    from training.overnight_training import training_jobs

    source, _, protected = seeds.load_source(source_run)
    if seeds.runtime(source["device"]) != source["manifest"]["runtime"]:
        raise IntegrityError("Use the source device/runtime, torch threads and determinism settings")
    old_output = seeds.run_directory(previous.VARIANT, policy_run)
    old_manifest = read(old_output / "manifest.json")
    old_signature = digest(old_manifest)
    if old_manifest["retrieval_plan"]["source_signature"] != source["signature"]:
        raise IntegrityError("Variant18 belongs to a different source experiment")
    receipt = seeds.verify_receipt({"output": old_output, "signature": old_signature}, old_output / "checkpoints_complete.json")
    review.check_inputs({"manifest": old_manifest, "protected": old_manifest["protected"]})
    protected.update(old_manifest["protected"])
    protected.update({str(old_output / p): h for p, h in receipt["artifacts"].items()})
    protected[str(old_output / "checkpoints_complete.json")] = sha256(old_output / "checkpoints_complete.json")
    configuration = {"training": training, "confirmation": confirmation, "masks": masks, "benchmark": benchmark,
                     "local_grid": evaluation.LOCAL_GRID, "training_jobs": [asdict(j) for j in training_jobs().values()],
                     "step": 800, "lr_horizon": 1700, "ranking": "less_graph", "local_min_gain": .002,
                     "training_pilot_min_gain": .003,
                     "confirmation_rule": "pilot >= .003 over family control AND R1 matched control; max one per family",
                     "outer_evaluation": False, "promotion": False, "external_data": False,
                     "confidence": "fixed score functions; identity-separated inner calibration/evaluation",
                     "mask_status": "automatic region proxy, NOT verified plate-only"}
    manifest = {**source["manifest"], "version": 19, "system_plan": configuration,
                "policy_source_signature": old_signature, "protected": {**source["protected"], **protected},
                "source_sha256": {**old_manifest["source_sha256"],
                    **{f"training/{n}.py": sha256(ROOT / "training" / f"{n}.py") for n in MODULES}}}
    output = seeds.run_directory(VARIANT, run_name)
    context = {**source, "source": source, "output": output, "old_output": old_output,
               "manifest": manifest, "signature": digest(manifest), "protected": manifest["protected"],
               "variants": review.variants(), "confirmation": read(source["output"] / "confirm_inner.json")}
    review.check_other_runs(context)
    with review.old.run_lock(output):
        review.old.freeze_json(output / "manifest.json", manifest)
    return context


class Queue:
    def __init__(self, context, wall_hours=None):
        if wall_hours is not None and (not np.isfinite(wall_hours) or wall_hours <= 0):
            raise ValueError("WALL_HOURS must be finite and positive")
        self.context = context
        self.started = time.monotonic()
        self.deadline = float("inf") if wall_hours is None else self.started + wall_hours * 3600
        self.events = []

    def task(self, name, compute):
        directory = self.context["output"] / "tasks" / name
        receipt = directory / "complete.json"
        if receipt.exists():
            saved = read(receipt)
            if saved["signature"] != self.context["signature"]:
                raise IntegrityError(f"Task configuration changed: {name}")
            for relative, expected in saved["artifacts"].items():
                path = self.context["output"] / relative
                if not path.is_file() or sha256(path) != expected:
                    raise IntegrityError(f"Changed task artifact: {relative}")
            result = read(directory / "result.json")
            print(f"[CACHED] {name}", flush=True)
            self.events.append({"task": name, "status": "cached"})
            return result
        if time.monotonic() >= self.deadline:
            raise BudgetReached("Time budget reached; saved tasks will resume on the next Run All")
        # Source edits during an overnight run would otherwise silently mix recipes.
        for relative, expected in self.context["manifest"]["source_sha256"].items():
            if sha256(ROOT / relative) != expected:
                raise IntegrityError(f"Source changed while running: {relative}")
        directory.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        print(f"[START] {name} | elapsed {(start-self.started)/3600:.2f} h", flush=True)
        write_json(directory / "status.json", {"status": "running", "signature": self.context["signature"]})
        try:
            result = compute(directory)
            write_json(directory / "result.json", result)
            files = [p for p in directory.rglob("*") if p.is_file() and p.name not in {"status.json", "error.json", "complete.json"}]
            # Training states live outside tasks; freeze their checksums too.
            for entry in result.get("checkpoints", {}).values():
                path = self.context["output"] / entry["path"]
                if sha256(path) != entry["sha256"]:
                    raise IntegrityError("Training checkpoint differs from its summary")
                files.append(path)
            if result.get("checkpoints"):
                training_directory = (self.context["output"] / next(iter(result["checkpoints"].values()))["path"]).parent
                files.extend(training_directory / name for name in ("last.pt", "history.json", "summary.json"))
            write_json(receipt, {"signature": self.context["signature"],
                                "artifacts": {str(p.relative_to(self.context["output"])): sha256(p) for p in files}})
            status = {"status": "complete", "seconds": time.monotonic() - start}
            write_json(directory / "status.json", status)
            self.events.append({"task": name, **status})
            write_json(self.context["output"] / "progress.json", {"signature": self.context["signature"], "events": self.events})
            print(f"[DONE] {name} | {status['seconds']/60:.1f} min", flush=True)
            return result
        except BudgetReached:
            status = {"status": "paused", "reason": "time budget reached at a saved boundary"}
            write_json(directory / "status.json", status)
            self.events.append({"task": name, **status})
            write_json(self.context["output"] / "progress.json", {"signature": self.context["signature"], "events": self.events})
            raise
        except (IntegrityError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            status = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
            write_json(directory / "error.json", status)
            write_json(directory / "status.json", status)
            self.events.append({"task": name, **status})
            write_json(self.context["output"] / "progress.json", {"signature": self.context["signature"], "events": self.events})
            print(f"[FAILED] {name}: {error}. Independent tasks will continue.", flush=True)
            return None


def fold_rows(context, fold):
    protocols = context["manifest"]["draws"][fold]
    ids = {i for p in protocols.values() if p["selection_eligible"] for k in ("query_ids", "gallery_ids") for i in p[k]}
    return [r for r in context["rows"] if r["image_id"] in ids]


def source_summary(context, fold, seed):
    return next(s for s in context["confirmation"][fold] if s["variant"] == "R1_resolution256" and s["seed"] == seed)


def source_vectors(context, fold, seed):
    path = context["old_output"] / "cache" / fold / "R1_resolution256" / str(seed) / "step_800.npz"
    receipt = read(path.with_suffix(".json"))
    if receipt["signature"] != context["manifest"]["policy_source_signature"] or sha256(path) != receipt["sha256"]:
        raise IntegrityError("Source vector cache changed")
    with np.load(path, allow_pickle=False) as arrays:
        vectors = arrays["vectors"]
    if len(vectors) != len(receipt["ids"]) or not np.isfinite(vectors).all():
        raise IntegrityError("Invalid source vectors")
    return dict(zip(receipt["ids"], vectors))


def cached_features(context, fold, seed, *, masked=False):
    from training.local_verification import extract_local_tokens

    rows = fold_rows(context, fold)
    summary = source_summary(context, fold, seed)
    checkpoint = summary["checkpoints"]["800"]
    mode = "automatic_mask" if masked else "original"
    path = context["output"] / "cache" / fold / str(seed) / f"{mode}.npz"
    receipt = path.with_suffix(".json")
    specification = {"signature": context["signature"], "checkpoint": checkpoint,
                     "ids": [r["image_id"] for r in rows], "grid": [4, 4], "mode": mode}
    if receipt.exists():
        saved = read(receipt)
        if saved["specification"] != specification or sha256(path) != saved["sha256"]:
            raise IntegrityError("Local feature cache changed")
        with np.load(path, allow_pickle=False) as data:
            return rows, data["global_vectors"], data["local_tokens"]
    model, variant = review.load_model(context["source"], {**summary, "stop_step": 800})
    loader = DataLoader(AblationDataset(rows, variant, context["dataset"], masks=context["masks"], masked=masked),
                        batch_size=32, shuffle=False, num_workers=0)
    globals_, tokens = [], []
    try:
        with torch.no_grad():
            for index, (batch, _, _) in enumerate(loader):
                batch = batch.to(context["device"])
                tokens.append(extract_local_tokens(model.backbone, batch, grid=(4, 4)))
                globals_.append(normalize(model.embedding(batch).cpu().numpy()))
                if index % 10 == 0:
                    print(f"  local features {fold}/{seed}/{mode}: {min((index+1)*32,len(rows))}/{len(rows)}", flush=True)
        gv, lv = np.concatenate(globals_), np.concatenate(tokens)
        if not masked:
            old = source_vectors(context, fold, seed)
            reference = np.stack([old[r["image_id"]] for r in rows])
            if not np.allclose(gv, reference, rtol=0, atol=2e-5):
                raise IntegrityError("Frozen encoder/preprocessing no longer reproduces cached R1 embeddings")
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".pending.npz")
        np.savez_compressed(pending, global_vectors=gv, local_tokens=lv)
        pending.replace(path)
        write_json(receipt, {"specification": specification, "sha256": sha256(path)})
        return rows, gv, lv
    finally:
        del model
        gc.collect()


def draw_arrays(context, fold, rows, vectors, tokens=None):
    positions = {r["image_id"]: i for i, r in enumerate(rows)}
    by_id = {r["image_id"]: r for r in rows}
    for name, protocol in context["manifest"]["draws"][fold].items():
        if not protocol["selection_eligible"]:
            continue
        qids, gids = (protocol[k] for k in ("query_ids", "gallery_ids"))
        qi, gi = ([positions[i] for i in ids] for ids in (qids, gids))
        yield name, [by_id[i] for i in qids], [by_id[i] for i in gids], vectors[qi], vectors[gi], (
            None if tokens is None else (tokens[qi], tokens[gi]))


def local_trial(context, fold, seed, configs, directory, *, masked=False, mask_condition="both"):
    from training.local_verification import local_pair_scores, mix_topk_scores, topk_oracle_diagnostics

    rows, vectors, tokens = cached_features(context, fold, seed)
    masked_draws = {}
    if masked:
        mr, mv, mt = cached_features(context, fold, seed, masked=True)
        masked_draws = {d[0]: d for d in draw_arrays(context, fold, mr, mv, mt)}
    reports = {c["name"]: {} for c in configs}
    for draw, query, gallery, qv, gv, local in draw_arrays(context, fold, rows, vectors, tokens):
        if masked:
            _, _, _, mq, mg, ml = masked_draws[draw]
            if mask_condition not in {"query", "gallery", "both"}:
                raise ValueError("Unknown mask diagnostic")
            qv = mq if mask_condition in {"query", "both"} else qv
            gv = mg if mask_condition in {"gallery", "both"} else gv
            local = (ml[0] if mask_condition in {"query", "both"} else local[0],
                     ml[1] if mask_condition in {"gallery", "both"} else local[1])
        began = time.perf_counter()
        ranking, base_scores = evaluation.rank_with_scores(qv, gv)
        baseline_seconds = time.perf_counter() - began
        local_scores, matching_seconds = {}, {}
        for mode in sorted({c["scorer"] for c in configs if c["weight"]}):
            began = time.perf_counter()
            local_scores[mode] = np.stack([local_pair_scores(local[0][i], local[1][order[:50]], scorer=mode)
                                          for i, order in enumerate(ranking["order"])])
            matching_seconds[mode] = time.perf_counter() - began
        for config in configs:
            began = time.perf_counter()
            order = ranking["order"].copy()
            if config["weight"]:
                order = mix_topk_scores(base_scores, ranking["order"], local_scores[config["scorer"]],
                                       top_k=config["top_k"], weight=config["weight"])["order"]
            mix_seconds = time.perf_counter() - began
            changed = {**ranking, "order": order}
            value = policy.evaluate(query, gallery, changed, 2., "raw_top1")["ranking"]
            diagnostics = policy.query_diagnostics(query, gallery, changed)
            oracle = topk_oracle_diagnostics(query, gallery, ranking["order"], ks=(20, 50), comparison_order=order)
            reports[config["name"]][draw] = {"ranking": value, "diagnostics": diagnostics, "oracle": oracle,
                "timing": {"query_count": len(query), "baseline_graph_and_queries_seconds": baseline_seconds,
                           "local_matching_top50_seconds": matching_seconds.get(config["scorer"], 0.) if config["weight"] else 0.,
                           "mix_seconds": mix_seconds, "scope": "CPU cached features; not official extractor latency"}}
            frozen_confidence = context["output"] / "tasks/confidence_primary/result.json"
            if masked and frozen_confidence.exists():
                cal = read(frozen_confidence)["R1_first_seed/cosine/raw_top1"]
                threshold = cal["threshold"]
                reports[config["name"]][draw]["candidates_fixed_original_threshold"] = {
                    "threshold": threshold, "metrics": policy.evaluate(query, gallery, changed, threshold, "raw_top1")["candidates"],
                    "acceptance": evaluation.acceptance_diagnostics(query, gallery, changed, threshold, "raw_top1"),
                    "note": "fixed original inner-cal threshold; diagnostic includes calibration IDs, not an unbiased C estimate"}
            if config["name"] == "baseline":
                reports[config["name"]][draw]["raw_pool_oracle"] = topk_oracle_diagnostics(
                    query, gallery, ranking["raw_order"], ks=(20, 50), comparison_order=ranking["order"])
            write_json(directory / config["name"] / f"{draw}.json", reports[config["name"]][draw])
        print(f"  {fold}/{seed}/{draw}: {len(configs)} local policies evaluated", flush=True)
    return reports


def confidence_trial(context, fold, directory, frozen_choices=None):
    from training.local_verification import local_pair_scores

    if fold == "alternate" and frozen_choices is None:
        raise IntegrityError("Freeze confidence functions on primary before alternate")
    members = [source_vectors(context, fold, s) for s in context["seeds"]]
    rows = fold_rows(context, fold)
    matrices = [np.stack([m[r["image_id"]] for r in rows]) for m in members]
    results = {}
    # One fixed episode, not pooled pseudo-replicates. All IDs remain encoder-held-out.
    for system, values in (("R1_first_seed", matrices[0]), ("R1_equal3", policy.combine_members(matrices))):
        draw, query, gallery, qv, gv, _ = next(draw_arrays(context, fold, rows, values))
        ranking, _ = evaluation.rank_with_scores(qv, gv)
        member_scores = None
        if system == "R1_equal3":
            member_scores = []
            for matrix in matrices:
                _, _, _, mq, mg, _ = next(draw_arrays(context, fold, rows, matrix))
                member_scores.append(mq @ mg.T)
        support = None
        if system == "R1_first_seed":
            local_rows, _, local_tokens = cached_features(context, fold, context["seeds"][0])
            indices = {r["image_id"]: i for i, r in enumerate(local_rows)}
            support = np.array([local_pair_scores(local_tokens[indices[q["image_id"]]],
                local_tokens[[indices[gallery[int(ranking["raw_order"][i, 0])]["image_id"]]]])[0]
                for i, q in enumerate(query)])
        signals = evaluation.confidence_signals(qv, gv, ranking, member_scores, support)
        for signal, confidence in signals.items():
            for candidate in policy.CANDIDATES:
                name = f"{system}/{signal}/{candidate}"
                if frozen_choices is not None and name not in frozen_choices:
                    continue
                result = evaluation.calibrate_inner_confidence(query, gallery, ranking, confidence, candidate)
                indices = [i for i, r in enumerate(query) if r["image_id"] in set(result["evaluation_ids"])]
                held = [query[i] for i in indices]
                selected = evaluation.slice_ranking(ranking, np.asarray(indices), confidence)
                export = directory / system / signal / candidate / "csv"
                # Recover an interrupted export within this NEW task only, without deleting data.
                if export.exists() and any(export.iterdir()):
                    export = directory / system / signal / candidate / f"csv_retry_{time.time_ns()}"
                actual = policy.export_csv(export, held, gallery, selected, result["threshold"], candidate)
                if actual != result["evaluation"]:
                    raise IntegrityError("Exported candidate CSV differs from internal official metrics")
                result["episode"] = draw
                results[name] = result
    return results


def choose_confidence(primary):
    selected = []
    for system in ("R1_first_seed", "R1_equal3"):
        names = [name for name in primary if name.startswith(system + "/")]
        best = max(names, key=lambda name: primary[name]["evaluation"]["candidates"]["C"])
        selected.extend([f"{system}/cosine/raw_top1", f"{system}/cosine/ranking_top1", best])
    return {"selected": list(dict.fromkeys(selected)),
            "rule": "maximize primary identity-held-out C per system; freeze score/candidate functions before alternate",
            "thresholds": "each fold calibrates only its disjoint inner calibration identities; never deploy these thresholds"}


def summarize_local_alternate(matrix, selected):
    if len(matrix) != 3 or any(value is None for value in matrix.values()):
        return {"complete": False, "passed": False}
    values = {seed: {name: evaluation.summarize_draws(reports[name]) for name in ("baseline", selected)}
              for seed, reports in matrix.items()}
    gains = {seed: v[selected] - v["baseline"] for seed, v in values.items()}
    paired = ({}, {})
    for seed, reports in matrix.items():
        for draw in reports["baseline"]:
            for dest, name in zip(paired, ("baseline", selected)):
                for qid, value in reports[name][draw]["diagnostics"]["per_query"].items():
                    dest[f"{seed}/{draw}/{qid}"] = {"vehicle_id": value["vehicle_id"], "ap": value["ranking"]["ap"]}
    return {"complete": True, "selected": selected, "seed_values": values, "seed_gains": gains,
            "mean_gain": mean(gains.values()),
            "passed": selected != "baseline" and mean(gains.values()) >= .002 and sum(v > 0 for v in gains.values()) >= 2,
            "conditional_bootstrap": evaluation.paired_identity_bootstrap(*paired), "promoted": False}


def evaluate_trained(context, summary, directory):
    from training.overnight_training import load_job_model

    model, variant = load_job_model(context, summary)
    fold = summary["fold"]
    rows = fold_rows(context, fold)
    try:
        encoded = review.old.encode(model, rows, context, variant)
        vectors = np.stack([encoded[r["image_id"]] for r in rows])
        reports = {}
        for draw, query, gallery, qv, gv, _ in draw_arrays(context, fold, rows, vectors):
            ranking = policy.rank_vectors(qv, gv, "less_graph")
            reports[draw] = {"ranking": policy.evaluate(query, gallery, ranking, 2., "raw_top1")["ranking"],
                             "diagnostics": policy.query_diagnostics(query, gallery, ranking)}
        return {"mean_map": evaluation.summarize_draws(reports), "draws": reports,
                "fold": fold, "seed": summary["seed"], "step": 800}
    finally:
        del model
        gc.collect()


def choose_training_pilots(results, minimum=.003):
    """Fixed family controls and one winner per family, selected only on primary."""
    if any(results.get(n) is None for n in ("R1_control", "K2_ce1")):
        return {"selected": [], "reason": "missing matched control; no automatic confirmation"}
    baseline = results["R1_control"]["mean_map"]
    families = {"ce": ("K2_ce1", ["K2_ce_sqrt544"]), "distillation": ("R1_control", ["R1_kd01", "R1_kd1"])}
    selected, details = [], {}
    for family, (control, candidates) in families.items():
        available = [n for n in candidates if results.get(n) is not None]
        if not available:
            details[family] = {"passed": False, "reason": "no completed candidate"}
            continue
        best = max(available, key=lambda n: results[n]["mean_map"])
        gain = results[best]["mean_map"] - results[control]["mean_map"]
        r1_gain = results[best]["mean_map"] - baseline
        passed = gain >= minimum and r1_gain >= minimum
        details[family] = {"candidate": best, "control": control, "gain": gain, "gain_over_r1": r1_gain, "passed": passed}
        if passed:
            selected.append(best)
    return {"selected": selected, "families": details, "minimum_gain": minimum, "scope": "one-seed pilot, not a final win"}


def confirmation_gate(matrix, candidate, controls, minimum=.002):
    seeds_ = list(matrix[candidate])
    if len(seeds_) != 3 or any(matrix[name].get(seed) is None for name in [candidate, *controls] for seed in seeds_):
        return {"passed": False, "complete": False, "reason": "missing three-seed primary results"}
    comparisons = {}
    for control in controls:
        gains = {seed: matrix[candidate][seed]["mean_map"] - matrix[control][seed]["mean_map"] for seed in seeds_}
        comparisons[control] = {"mean_gain": mean(gains.values()), "seed_gains": gains,
                                "passed": mean(gains.values()) >= minimum and sum(v > 0 for v in gains.values()) >= 2}
    return {"passed": all(c["passed"] for c in comparisons.values()), "complete": True,
            "comparisons": comparisons, "minimum_gain": minimum}


def training_queue(queue):
    from training.overnight_training import TrainingPaused, fit_job, training_jobs

    context = queue.context
    jobs = training_jobs()
    first = context["seeds"][0]
    results = {}

    def one(name, seed, fold):
        summaries = [source_summary(context, fold, s) for s in context["seeds"]]
        teacher_args = {"teacher_context": context["source"], "teacher_summaries": summaries} if jobs[name].relation_weight else {}
        def train(directory):
            try:
                return fit_job(context, jobs[name], seed, fold=fold, **teacher_args,
                               should_stop=lambda: time.monotonic() >= queue.deadline)
            except TrainingPaused as error:
                raise BudgetReached(str(error)) from error
        summary = queue.task(f"train_{fold}_{name}_{seed}", train)
        if summary is None:
            return None
        return queue.task(f"evaluate_{fold}_{name}_{seed}", lambda directory: evaluate_trained(context, summary, directory))

    for name in jobs:
        results[name] = one(name, first, "primary")
    if any(value is None for value in results.values()):
        return {"pilots": results, "selection": None, "confirmation": {},
                "reason": "Retry failed pilots with the same RUN_NAME before freezing a selection"}
    decision = queue.task("training_pilot_selection", lambda directory: choose_training_pilots(results))
    if decision is None or not context["manifest"]["system_plan"]["confirmation"]:
        return {"pilots": results, "selection": decision, "confirmation": {}}
    confirmed = {}
    for name in decision["selected"]:
        controls = list(dict.fromkeys(["R1_control", "K2_ce1" if name.startswith("K2") else "R1_control"]))
        confirmed[name] = {"primary": {}, "alternate": {}}
        for model in [*controls, name]:
            confirmed[name]["primary"][model] = {str(seed): results[model] if seed == first else one(model, seed, "primary")
                                                 for seed in context["seeds"]}
        gate = confirmation_gate(confirmed[name]["primary"], name, controls)
        if gate["complete"]:
            gate = queue.task(f"primary_confirmation_gate_{name}", lambda directory, g=gate: g)
        confirmed[name]["primary_gate"] = gate
        if gate is not None and gate["passed"]:
            for model in [*controls, name]:
                confirmed[name]["alternate"][model] = {str(seed): one(model, seed, "alternate") for seed in context["seeds"]}
            confirmed[name]["alternate_gate"] = confirmation_gate(confirmed[name]["alternate"], name, controls)
    return {"pilots": results, "selection": decision, "confirmation": confirmed}


def benchmark_task(context, directory):
    from training.system_benchmark import audit_streaming, benchmark_bundle, load_encoder, timed_export

    result = {}
    for name in (f"R1_resolution256_{context['seeds'][0]}", "R1_equal3"):
        bundle = context["old_output"] / "final" / name / "less_graph/raw_top1/bundle.json"
        result[name] = benchmark_bundle(bundle, context["dataset"], provider="CPUExecutionProvider",
                                        rows=fold_rows(context, "primary")[:64], official_hardware=False)
        protocol = next(p for p in context["manifest"]["draws"]["primary"].values() if p["selection_eligible"])
        query, gallery = previous.rows_for(context, protocol)
        encoder = load_encoder(bundle, "CPUExecutionProvider")
        result[name]["streaming_audit"] = audit_streaming(encoder, query[:4], gallery[:24], context["dataset"], batch_size=3)
        if not result[name]["streaming_audit"]["passed"]:
            raise IntegrityError("Frozen runtime decisions depend on query order or batch size")
        del encoder
        export = directory / name / "public_test_timing_export"
        if export.exists() and any(export.iterdir()):
            export = directory / name / f"public_test_timing_export_retry_{time.time_ns()}"
        result[name]["full_inference"] = timed_export(bundle, context["dataset"], export, "CPUExecutionProvider")
    return result


def write_report(context, result, events):
    lines = ["# Ночная серия OSNet / variant19", "", f"Статус: **{result['status']}**.",
             "Outer validation не оценивалась. Автоматического переключения MVP нет.",
             "Сохранность исходной разметки/MVP/старых результатов подтверждена SHA256." if result.get("protected_unchanged")
             else "Проверка сохранности защищённых файлов не подтверждена; см. статус и ошибку.",
             "", "## Задачи", "", "| Задача | Статус |", "|---|---|"]
    lines += [f"| {e['task']} | {e['status']} |" for e in events]
    local = result.get("local_selection")
    if local:
        lines += ["", "## Локальный scorer: primary", "", "| Настройка | Средний mAP@10 |", "|---|---:|"]
        lines += [f"| {name} | {value:.5f} |" for name, value in local["mean_map"].items()]
        lines += ["", f"Замороженный выбор для alternate: `{local['selected']}`."]
        alternate = result.get("local_alternate_summary", {})
        if alternate.get("complete"):
            lines += [f"Alternate: среднее изменение {alternate['mean_gain']*100:+.3f} п.п.; gate={alternate['passed']}.",
                      "Парный bootstrap по identity: " + str(alternate["conditional_bootstrap"])]
    for fold in ("primary", "alternate"):
        confidence = result.get(f"confidence_{fold}")
        if confidence:
            lines += ["", f"## Кандидаты: {fold}, отдельные held-out identity", "",
                      "| Политика | F1 | TNR | C | Coverage | Ошибки среди принятых |", "|---|---:|---:|---:|---:|---:|"]
            for name, value in confidence.items():
                c, a = value["evaluation"]["candidates"], value["acceptance"]
                error = "—" if a["accepted_error_rate"] is None else f"{a['accepted_error_rate']:.4f}"
                lines += [f"| {name} | {c['F1']:.4f} | {c['TNR']:.4f} | {c['C']:.4f} | {a['coverage']:.4f} | {error} |"]
    training = result.get("training")
    if training:
        lines += ["", "## Обучение: односидовые пилоты", "", "| Рецепт | mAP@10 primary |", "|---|---:|"]
        lines += [f"| {name} | {value['mean_map']:.5f} |" if value else f"| {name} | FAILED |"
                  for name, value in training["pilots"].items()]
        lines += ["", "Отбор пилотов: " + str(training.get("selection"))]
        for name, matrix in training["confirmation"].items():
            for fold in ("primary", "alternate"):
                lines += ["", f"### {name}: {fold}", "", "| Рецепт | Seed | mAP@10 |", "|---|---:|---:|"]
                for model, seeds_ in matrix[fold].items():
                    lines += [f"| {model} | {seed} | {v['mean_map']:.5f} |" if v else f"| {model} | {seed} | FAILED |"
                              for seed, v in seeds_.items()]
            lines += ["", "Primary gate: " + str(matrix.get("primary_gate")), "Alternate gate: " + str(matrix.get("alternate_gate"))]
    lines += ["", "## Границы вывода", "",
              "- Все метрики новых вариантов относятся к held-out train IDs; это не новая outer-метрика.",
              "- Локальный scorer заморожен и не обучает pair-head. DINO/обучаемая OOF-голова не запускались.",
              "- Автоматические маски — proxy, не plate-only аудит; координат подтверждённых номеров нет.",
              "- CPU benchmark не подтверждает баллы RTX A5000/CUDA и не переключает приложение.",
              "- Smooth-AP и GPU dependency lock отложены; веса/порог не выбираются по outer.",
              "- По умолчанию ограничения времени нет. После прерывания повторите Run All с тем же RUN_NAME; завершённые задачи пропускаются."]
    path = context["output"] / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")


def run(context, wall_hours=None, *, allow_cpu_training=False):
    plan = context["manifest"]["system_plan"]
    if plan["training"] and context["device"].type == "cpu" and not allow_cpu_training:
        raise ValueError("CPU training requires explicit allow_cpu_training=True")
    queue = Queue(context, wall_hours)
    result = {"signature": context["signature"], "status": "running", "outer_evaluated": False, "promoted": False}
    with review.old.run_lock(context["output"]):
        review.check_inputs(context)
        try:
            if plan["benchmark"]:
                result["benchmark"] = queue.task("benchmark", lambda directory: benchmark_task(context, directory))
            primary = {}
            for seed in context["seeds"]:
                value = queue.task(f"local_primary_{seed}", lambda directory, s=seed: local_trial(
                    context, "primary", s, evaluation.LOCAL_GRID, directory))
                if value is not None:
                    primary[str(seed)] = value
            if len(primary) == 3:
                selected = queue.task("local_selection", lambda directory: evaluation.select_local(primary))
                result["local_selection"] = selected
                if selected:
                    configs = [c for c in evaluation.LOCAL_GRID if c["name"] in {"baseline", selected["selected"]}]
                    result["local_alternate"] = {str(seed): queue.task(f"local_alternate_{seed}",
                        lambda directory, s=seed: local_trial(context, "alternate", s, configs, directory)) for seed in context["seeds"]}
                    result["local_alternate_summary"] = summarize_local_alternate(result["local_alternate"], selected["selected"])
            result["confidence_primary"] = queue.task("confidence_primary", lambda directory: confidence_trial(context, "primary", directory))
            if result["confidence_primary"] is not None:
                decision = queue.task("confidence_selection", lambda directory: choose_confidence(result["confidence_primary"]))
                if decision is not None:
                    result["confidence_selection"] = decision
                    result["confidence_alternate"] = queue.task("confidence_alternate", lambda directory: confidence_trial(
                        context, "alternate", directory, decision["selected"]))
            if plan["masks"] and result.get("local_selection"):
                configs = [c for c in evaluation.LOCAL_GRID if c["name"] in {"baseline", result["local_selection"]["selected"]}]
                result["automatic_mask_proxy"] = {condition: queue.task(f"automatic_mask_{condition}",
                    lambda directory, c=condition: local_trial(context, "primary", context["seeds"][0], configs,
                                                              directory, masked=True, mask_condition=c))
                    for condition in ("query", "gallery", "both")}
            if plan["training"]:
                result["training"] = training_queue(queue)
            result["status"] = "complete_with_failures" if any(e["status"] == "failed" for e in queue.events) else "complete"
        except BudgetReached as error:
            result.update(status="time_budget_reached", message=str(error))
        except BaseException:
            result["status"] = "interrupted_or_integrity_error"
            raise
        finally:
            result["events"] = queue.events
            result["elapsed_seconds"] = time.monotonic() - queue.started
            try:
                review.check_inputs(context, rehash=True)
                result["protected_unchanged"] = True
            except BaseException:
                result["status"] = "integrity_check_failed"
                result["protected_unchanged"] = False
                raise
            finally:
                write_json(context["output"] / "results.json", result)
                write_report(context, result, queue.events)
    return result
