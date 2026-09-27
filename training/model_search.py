"""v21: bounded search over concrete saved models, never a new training recipe."""
import gc
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.core import ROOT, normalize, sha256
from backend.rerank import KReciprocalReranker
from training import quality_experiment as quality
from training import quality_clock as clock
from training import quality_verifier as head
from training.audit import digest
from training.local_verification import mix_topk_scores
from training.osnet_ablations import AblationDataset
from training.stage6 import write_json

old = quality.old
VARIANT = ROOT / "OSNet-AIN-x1.0/variant_21_model_search"
LAMBDAS = (.75, .50, .65, .85)  # Frozen tie order: retain the current policy.
AVERAGE_STEPS = (1400, 1600, 1700)


def components(seeds):
    return {f"{case}_{seed}": {"case": case, "seed": seed}
            for case in ("R1_control", "R1_full1700", "R1_full1700_bn", "R1_full1700_avg", "R1_cosine800")
            for seed in (seeds[:1] if case == "R1_cosine800" else seeds)}


def system(name, members, weights=None, head_weight=0.):
    return {"name": name, "members": list(members),
            "weights": list(weights) if weights is not None else [1 / len(members)] * len(members),
            "head_weight": head_weight}


def screening_systems(seeds):
    base = [f"R1_control_{s}" for s in seeds]
    specs = [system("R1_equal3", base)] + [system(n, [n]) for n in components(seeds)]
    specs += [system(f"head_w{int(w*100):02d}_{s}", [f"R1_control_{s}"], head_weight=w)
              for s in seeds for w in (.05, .10)]
    return specs


def ensemble_systems(seeds, candidate):
    """Five fixed mixtures, not an unbounded subset/weight optimizer."""
    base = [f"R1_control_{s}" for s in seeds]
    if candidate in base:
        raise ValueError("Ensemble challenger must be a new encoder")
    specs = [system(f"replace_{s}", [candidate if j == i else n for j, n in enumerate(base)])
             for i, s in enumerate(seeds)]
    specs += [system(f"add_w{int(w*100):02d}", [*base, candidate], [(1-w)/3]*3 + [w]) for w in (.10, .25)]
    return specs


def best(reports, names):
    """Choose actual weights by mean query-draw score, never mean across seeds."""
    if not names or any(reports.get(n) is None for n in names):
        raise ValueError("Selection requires every planned comparison; resume the failed task")
    if any(not np.isfinite(reports[n]["mean_map"]) for n in names):
        raise ValueError("Non-finite ranking metric")
    return max(names, key=lambda n: reports[n]["mean_map"])  # Stable ties.


def verify_task_source(output, signature, path):
    receipt = old.read(path)
    if receipt["signature"] != signature or not receipt["artifacts"]:
        raise old.IntegrityError("Source task signature changed")
    protected = {str(path): sha256(path)}
    for relative, expected in receipt["artifacts"].items():
        target = (output / relative).resolve()
        if not target.is_relative_to(output.resolve()) or not target.is_file() or sha256(target) != expected:
            raise old.IntegrityError(f"Source artifact changed: {relative}")
        protected[str(target)] = expected
    return protected


def prepare(run_name="search_v1", *, source_run="review_v1", policy_run="policy_v1", quality_run="quality_v1"):
    source, _, protected = old.seeds.load_source(source_run)
    quality_output = old.seeds.run_directory(quality.VARIANT, quality_run)
    qm = old.read(quality_output / "manifest.json")
    qs = digest(qm)
    policy_output = old.seeds.run_directory(old.previous.VARIANT, policy_run)
    pm = old.read(policy_output / "manifest.json")
    if (qm["policy_source_signature"] != digest(pm)
            or pm["retrieval_plan"]["source_signature"] != source["signature"]
            or any(qm[k] != source["manifest"][k] for k in ("inner", "draws", "frames_sha256", "runtime"))):
        raise old.IntegrityError("v16/v18/v20 use different source data or runtime")
    if old.seeds.runtime(source["device"]) != qm["runtime"]:
        raise old.IntegrityError("Use the original research kernel, device, threads and determinism")
    if source["device"].type == "mps" and not torch.backends.mps.is_available():
        raise old.IntegrityError("MPS unavailable; no silent CPU fallback")
    result = old.read(quality_output / "results.json")
    if result["status"] != "complete" or not result["protected_unchanged"]:
        raise old.IntegrityError("Complete v20 before starting v21")
    protected.update(qm["protected"])
    for path in quality_output.glob("tasks/*/complete.json"):
        protected.update(verify_task_source(quality_output, qs, path))
    for path in (quality_output / "manifest.json", quality_output / "results.json", policy_output / "manifest.json"):
        protected[str(path)] = sha256(path)
    for seed in source["seeds"]:
        for role in ("evaluation",):
            path = quality_output / "cache/primary" / str(seed) / f"{role}.npz"
            receipt = old.read(path.with_suffix(".json"))
            if receipt["specification"]["signature"] != qs or sha256(path) != receipt["sha256"]:
                raise old.IntegrityError("v20 evaluation feature cache changed")
            protected[str(path)] = receipt["sha256"]
            protected[str(path.with_suffix(".json"))] = sha256(path.with_suffix(".json"))
    plan = {"source_run": source_run, "policy_run": policy_run, "quality_run": quality_run,
            "components": components(source["seeds"]), "screening": screening_systems(source["seeds"]),
            "lambdas": list(LAMBDAS), "k1": 20, "k2": 3, "average_steps": list(AVERAGE_STEPS),
            "ensemble_search": "best new standalone encoder: three replacements, additions at .10/.25",
            "finalists": "current equal3 plus two best other concrete systems; four lambdas each",
            "selection": "maximum mean mAP across the same three primary query draws; no seed-average gate",
            "scope": "post-hoc development search, not an independent confirmation or hidden-test estimate",
            "optimizer_updates": 0, "bn": "train-only statistics for two missing averages",
            "outer_evaluation": False, "alternate_evaluation": False, "promoted": False,
            "candidate": "raw_top1/cosine; finalist calibration diagnostics only, no release threshold",
            "wall_time_limit": None}
    manifest = {**qm, "version": 21, "search_plan": plan, "quality_source_signature": qs,
                "protected": {**source["protected"], **protected},
                "source_sha256": {**qm["source_sha256"], "training/model_search.py": sha256(Path(__file__))}}
    output = old.seeds.run_directory(VARIANT, run_name)
    context = {**source, "source": source, "output": output, "old_output": policy_output,
               "quality_output": quality_output, "quality_manifest": qm, "quality_result": result,
               "manifest": manifest, "signature": digest(manifest), "protected": manifest["protected"],
               "confirmation": old.read(source["output"] / "confirm_inner.json")}
    old.review.check_inputs(context)
    rows = old.fold_rows(context, "primary")
    allowed = quality.training.fold_training_ids(context, "primary")
    if {r["vehicle_id"] for r in rows} & allowed:
        raise old.IntegrityError("Evaluation identity leakage")
    # Check completeness before publishing a manifest, not after expensive extraction.
    for spec in plan["components"].values():
        if spec["case"] != "R1_control":
            load_summary(context, spec)
    old.review.check_other_runs(context)
    with old.review.old.run_lock(output):
        old.review.old.freeze_json(output / "manifest.json", manifest)
    return context


def quality_context(context):
    return {**context, "output": context["quality_output"], "manifest": context["quality_manifest"],
            "signature": context["manifest"]["quality_source_signature"]}


def load_summary(context, spec):
    job = clock.job_name(spec["case"])
    path = context["quality_output"] / f"tasks/clock_train_primary_{job}_{spec['seed']}/result.json"
    if str(path) not in context["protected"] or sha256(path) != context["protected"][str(path)]:
        raise old.IntegrityError("Missing verified source training summary")
    summary = old.read(path)
    allowed = quality.training.fold_training_ids(context, "primary")
    if (summary["fold"] != "primary" or summary["seed"] != spec["seed"]
            or summary["train_identity_digest"] != digest(sorted(allowed))):
        raise old.IntegrityError("Source model belongs to a different identity split")
    return summary


def load_encoder(context, spec, directory):
    """Reuse old derived states; only missing averages write weights into v21."""
    qc = quality_context(context)
    summary = load_summary(context, spec)
    model, variant = quality.training.load_job_model(clock.job_context(qc, clock.job_name(spec["case"])), summary)
    provenance = {"source_summary": str(context["quality_output"] / f"tasks/clock_train_primary_{clock.job_name(spec['case'])}_{spec['seed']}/result.json"),
                  "source_signature": summary["signature"], "case": spec["case"], "seed": spec["seed"]}
    if spec["case"].endswith(("_bn", "_avg")):
        previous = context["quality_output"] / f"tasks/clock_eval_primary_{spec['case']}_{spec['seed']}/derived.pt"
        if str(previous) in context["protected"]:
            if sha256(previous) != context["protected"][str(previous)]:
                raise old.IntegrityError("Derived source weights changed")
            saved = torch.load(previous, map_location="cpu", weights_only=True)
            if saved["metadata"]["case"] != spec["case"] or saved["metadata"]["context_signature"] != qc["signature"]:
                raise old.IntegrityError("Derived source metadata changed")
            model.load_state_dict(saved["model"])
            provenance.update(path=str(previous), sha256=sha256(previous), reused=True)
        else:
            if spec["case"] != "R1_full1700_avg":
                raise old.IntegrityError("Missing BN-only state from completed v20")
            states = []
            for step in AVERAGE_STEPS:
                entry = summary["checkpoints"][str(step)]
                path = context["quality_output"] / entry["path"]
                if sha256(path) != entry["sha256"]:
                    raise old.IntegrityError("Averaging checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["signature"] != summary["signature"] or saved["step"] != step:
                    raise old.IntegrityError("Averaging checkpoint metadata changed")
                states.append(saved["model"])
            clock.average_parameters(model, states)
            allowed = quality.training.fold_training_ids(context, "primary")
            rows = [r for r in context["rows"] if r["vehicle_id"] in allowed]
            loader = DataLoader(AblationDataset(rows, variant, context["dataset"], augment=False),
                                batch_sampler=clock.bn_batches(len(rows)), num_workers=0)
            def batches():
                for i, batch in enumerate(loader):
                    if i % 20 == 0:
                        print(f"  BN {spec['seed']}: {i}/{len(loader)} train batches", flush=True)
                    yield batch[0]
            count = clock.recalibrate_bn(model, batches(), context["device"])
            provenance.update(steps=list(AVERAGE_STEPS), bn_train_images=count, bn_identity_digest=digest(sorted(allowed)))
            path = directory / "derived.pt"
            if not path.resolve().is_relative_to(context["output"].resolve()):
                raise old.IntegrityError("New derived weights must stay inside v21")
            old.review.old.save_checkpoint(path, {"model": model.state_dict(), "metadata": provenance})
            provenance.update(path=str(path), sha256=sha256(path), reused=False)
    else:
        entry = summary["checkpoints"][str(summary["stop_step"]) ]
        provenance.update(path=str(context["quality_output"] / entry["path"]), sha256=entry["sha256"])
    return model.eval(), variant, provenance


def feature_task(context, name, spec, directory):
    rows = old.fold_rows(context, "primary")
    if spec["case"] == "R1_control":
        source = old.source_vectors(context, "primary", spec["seed"])
        vectors = np.stack([source[r["image_id"]] for r in rows])
        entry = old.source_summary(context, "primary", spec["seed"])["checkpoints"]["800"]
        provenance = {"path": str(context["source"]["output"] / entry["path"]), "sha256": entry["sha256"], "reused": True}
    else:
        model, variant, provenance = load_encoder(context, spec, directory)
        try:
            loader = DataLoader(AblationDataset(rows, variant, context["dataset"]), batch_size=32, shuffle=False, num_workers=0)
            values = []
            with torch.no_grad():
                for i, (batch, _, _) in enumerate(loader):
                    values.append(normalize(model.embedding(batch.to(context["device"])).cpu().numpy()))
                    if i % 10 == 0 or i + 1 == len(loader):
                        print(f"  {name}: {min((i+1)*32,len(rows))}/{len(rows)} images", flush=True)
            vectors = np.concatenate(values)
        finally:
            del model
            gc.collect()
    validate_vectors(vectors, len(rows))
    path = directory / "features.npz"
    np.savez_compressed(path, vectors=vectors, ids=np.array([r["image_id"] for r in rows]))
    return {"path": str(path), "sha256": sha256(path), "model": provenance, "component": spec}


def validate_vectors(values, count):
    if (values.shape != (count, 512) or values.dtype != np.float32 or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1., rtol=0, atol=2e-5)):
        raise old.IntegrityError("Invalid or zero embedding; no artificial replacements allowed")


def combine(members, weights):
    values = [np.asarray(v, dtype=np.float32) for v in members]
    weights = np.asarray(weights, dtype=np.float64)
    if (not values or weights.shape != (len(values),) or not np.isfinite(weights).all()
            or (weights <= 0).any() or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)
            or any(v.ndim != 2 or v.shape != values[0].shape or not np.isfinite(v).all()
                   or (np.linalg.norm(v, axis=1) == 0).any() for v in values)):
        raise ValueError("Need aligned finite nonzero members and positive weights summing to one")
    if len(values) in (1, 3) and np.all(weights == weights[0]):
        return old.policy.combine_members(values)  # Exact historical equal3 arithmetic.
    return normalize(np.concatenate([normalize(v) * np.sqrt(w) for v, w in zip(values, weights)], axis=1).astype(np.float32))


def rank(qv, gv, lam):
    if lam not in LAMBDAS:
        raise ValueError("Lambda outside the frozen search grid")
    qv, gv = normalize(qv), normalize(gv)
    cosine = np.clip(qv @ gv.T, -1, 1)
    raw = np.argsort(-cosine, axis=1, kind="stable")
    k1 = min(20, len(gv)-1)
    graph = KReciprocalReranker(gv, k1, min(3, k1+1))
    scores = -np.stack([graph.distances(v, lam) for v in qv])
    return {"order": np.argsort(-scores, axis=1, kind="stable"), "raw_order": raw,
            "confidence": cosine.max(axis=1)}, scores


def load_head_data(context, spec, rows, vectors):
    if len(spec["members"]) != 1 or not spec["members"][0].startswith("R1_control_"):
        raise old.IntegrityError("Pair head is tied to its original single encoder")
    seed = int(spec["members"][0].rsplit("_", 1)[1])
    path = context["quality_output"] / f"cache/primary/{seed}/evaluation.npz"
    receipt = old.read(path.with_suffix(".json"))
    if receipt["specification"]["rows"] != rows or sha256(path) != receipt["sha256"]:
        raise old.IntegrityError("Head feature row order changed")
    with np.load(path, allow_pickle=False) as data:
        tokens = data["tokens"]
        if not np.allclose(data["vectors"], vectors, rtol=0, atol=2e-5):
            raise old.IntegrityError("Head paired with a different encoder")
    summary = old.read(context["quality_output"] / f"tasks/head_train_primary_{seed}/result.json")
    if {r["vehicle_id"] for r in rows} & set(summary["train_identities"]):
        raise old.IntegrityError("Head has seen evaluation identities")
    return head.load_head(summary), tokens


def evaluate_system(context, spec, features, directory, lam=.75, candidate_diagnostics=False):
    rows = old.fold_rows(context, "primary")
    vectors = combine([features[n] for n in spec["members"]], spec["weights"])
    model, tokens = load_head_data(context, spec, rows, vectors) if spec["head_weight"] else (None, None)
    reports = {}
    for draw, query, gallery, qv, gv, local in old.draw_arrays(context, "primary", rows, vectors, tokens):
        ranking, scores = rank(qv, gv, lam)
        if model is not None:
            qt, gt = local
            pool = ranking["order"][:, :min(50, len(gallery))]
            evidence = np.stack([head.score_pairs(model, qt[i], gt[js], gv[js] @ qv[i]) for i, js in enumerate(pool)])
            ranking["order"] = mix_topk_scores(scores, ranking["order"], evidence, top_k=50, weight=spec["head_weight"])["order"]
        report = quality.ranking_report(query, gallery, ranking)
        raw = {**ranking, "order": ranking["raw_order"]}
        report["raw_map"] = old.policy.evaluate(query, gallery, raw, 2., "raw_top1")["ranking"]["mAP@10"]
        if candidate_diagnostics:
            diagnostic = quality.evaluation.calibrate_inner_confidence(query, gallery, ranking, ranking["confidence"], "raw_top1")
            diagnostic.pop("curve")
            report["candidate_diagnostic"] = diagnostic
        gids = np.array([r["image_id"] for r in gallery])
        np.savez_compressed(directory / f"{draw}.npz", query_ids=np.array([r["image_id"] for r in query]),
                            gallery_ids=gids, top10=gids[ranking["order"][:, :10]],
                            raw_top1=gids[ranking["raw_order"][:, 0]], confidence=ranking["confidence"])
        reports[draw] = report
        print(f"  {spec['name']} lambda={lam} {draw}: {report['ranking']['mAP@10']:.6f}", flush=True)
    value = quality.evaluation.summarize_draws(reports)
    # Frozen references must reproduce, not be silently replaced by a new baseline.
    if lam == .75:
        expected = reference_score(context, spec)
        if expected is not None and abs(value - expected) > 1e-10:
            raise old.IntegrityError(f"Historical score mismatch for {spec['name']}: {value} != {expected}")
    return {"system": spec, "lambda": lam, "mean_map": value, "draws": reports}


def reference_score(context, spec):
    qr = context["quality_result"]
    if spec["name"] == "R1_equal3":
        return old.read(context["old_output"] / "selection.json")["R1_equal3"]["scores"]["less_graph"]["mean"]
    if len(spec["members"]) != 1:
        return None
    component = context["manifest"]["search_plan"]["components"][spec["members"][0]]
    seed, case = str(component["seed"]), component["case"]
    if spec["head_weight"] or case == "R1_control":
        key = f"head_w{int(spec['head_weight']*100):02d}" if spec["head_weight"] else "baseline"
        return qr["head"]["primary"][key][seed]["mean_map"]
    report = qr["clock"]["primary"].get(case, {}).get(seed)
    if report is None and int(seed) == context["seeds"][0]:
        report = qr["clock"]["pilots"].get(case)
    return None if report is None else report["mean_map"]


def comparison(before, after):
    a, b = {}, {}
    for target, report in ((a, before), (b, after)):
        for draw, values in report["draws"].items():
            target.update({f"{draw}/{qid}": v for qid, v in values["per_query"].items()})
    if set(a) != set(b):
        raise ValueError("Unpaired query draws")
    deltas = [b[k]["ap"] - v["ap"] for k, v in a.items() if v["ap"] is not None]
    return {"mean_gain": after["mean_map"] - before["mean_map"],
            "improved_episodes": sum(x > 1e-12 for x in deltas), "worsened_episodes": sum(x < -1e-12 for x in deltas),
            "unchanged_episodes": sum(abs(x) <= 1e-12 for x in deltas),
            "note": "descriptive post-selection comparison; repeated episodes are not independent identities"}


def write_report(context, result):
    lines = ["# v21 — выбор конкретной модели и ансамбля", "", f"Статус: **{result['status']}**.",
             "Выбор по primary development; не новый независимый тест. Среднего gate по seed нет.",
             "Optimizer updates: 0. Outer/alternate/test не оцениваются. MVP и исходная разметка не меняются.",
             "", "| Конфигурация | λ | mAP@10 |", "|---|---:|---:|"]
    for key, report in sorted(result.get("evaluations", {}).items(), key=lambda item: -(item[1] or {}).get("mean_map", -1)):
        if report:
            lines.append(f"| {key} | {report['lambda']:.2f} | {report['mean_map']:.6f} |")
        else:
            lines.append(f"| {key} | — | FAILED |")
    selection = result.get("selection")
    if selection:
        lines += ["", f"Лучший внутренний кандидат: **{selection['selected']}**.",
                  f"Изменение к equal3/λ0.75: {selection['comparison']['mean_gain']*100:+.3f} п.п.",
                  "Точный состав, веса, источники checkpoints и настройки: selected_candidate.json."]
    lines += ["", "## Ограничения", "",
              "- Максимум выбран на уже исследуемых данных; возможна подгонка при выборе. Это не ожидаемый hidden-test score.",
              "- mAP — среднее трёх query/gallery эпизодов одной конкретной модели, не среднее нескольких seed.",
              "- Конкретные primary-веса нельзя проверять на alternate: они могли обучаться на его identity.",
              "- Candidate F1/TNR/C — отдельная внутренняя диагностика, не критерий выбора и не конкурсный threshold.",
              "- Head применяется только к своему исходному R1 encoder; в ансамбль head не добавляется.",
              "- Нового обучения/экспорта/релиза нет; selected_candidate.json — описание исследовательского кандидата, не deployment bundle.",
              "- Время без общего лимита, выполненные задачи возобновляются по checksum.",
              f"- Защищённые данные/источники сохранены: {result.get('protected_unchanged', False)}."]
    (context["output"] / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(context):
    queue = old.Queue(context, wall_hours=None)
    result = {"status": "running", "evaluations": {}, "optimizer_updates": 0, "outer_evaluation": False, "promoted": False}
    with old.review.old.run_lock(context["output"]):
        try:
            old.review.check_inputs(context, rehash=True)
            plan = context["manifest"]["search_plan"]
            features, artifacts = {}, {}
            ids = [r["image_id"] for r in old.fold_rows(context, "primary")]
            for name, spec in plan["components"].items():
                item = queue.task(f"features_{name}", lambda d: feature_task(context, name, spec, d))
                if item is not None:
                    with np.load(item["path"], allow_pickle=False) as arrays:
                        if arrays["ids"].tolist() != ids:
                            raise old.IntegrityError("Feature cache row order changed")
                        features[name] = arrays["vectors"]
                    validate_vectors(features[name], len(ids))
                    artifacts[name] = item
            if len(features) != len(plan["components"]):
                raise ValueError("Incomplete component features; resume this RUN_NAME")
            reports = result["evaluations"]
            def evaluate(spec, lam=.75, diagnostics=False):
                key = f"{spec['name']}_lambda{int(lam*100):02d}"
                if key not in reports:
                    reports[key] = queue.task(f"evaluate_{key}", lambda d: evaluate_system(context, spec, features, d, lam, diagnostics))
                return key
            print("\nSTAGE 1/3: concrete saved models and heads", flush=True)
            screen = plan["screening"]
            names = [evaluate(s) for s in screen]
            best(reports, names)  # No partial selection if any task failed.
            novel = [f"{n}_lambda75" for n, s in plan["components"].items() if s["case"] != "R1_control"]
            chosen = reports[best(reports, novel)]["system"]["members"][0]
            result["ensemble_challenger"] = chosen
            print(f"\nSTAGE 2/3: five mixtures with {chosen}", flush=True)
            mixtures = ensemble_systems(context["seeds"], chosen)
            names += [evaluate(s) for s in mixtures]
            best(reports, names)
            baseline = "R1_equal3_lambda75"
            others = sorted((n for n in names if n != baseline), key=lambda n: -reports[n]["mean_map"])[:2]
            finalists = [reports[n]["system"] for n in [baseline, *others]]
            result["finalists"] = [s["name"] for s in finalists]
            print(f"\nSTAGE 3/3: reranking finalists {result['finalists']}", flush=True)
            for spec in finalists:
                names += [evaluate(spec, lam) for lam in plan["lambdas"]]
            winner = best(reports, list(dict.fromkeys(names)))
            winning_report = reports[winner]
            diagnostic = queue.task("winner_candidate_diagnostics", lambda d: evaluate_system(
                context, winning_report["system"], features, d, winning_report["lambda"], True))
            if diagnostic is None:
                raise ValueError("Winner diagnostics failed; resume before finalizing")
            selection = {"selected": winner, "system": winning_report["system"],
                         "ranking": {"k1": 20, "k2": 3, "lambda": winning_report["lambda"]},
                         "candidate_policy": "raw_top1", "threshold": None,
                         "mean_map": winning_report["mean_map"], "baseline_map": reports[baseline]["mean_map"],
                         "comparison": comparison(reports[baseline], winning_report),
                         "members": {n: artifacts[n]["model"] for n in winning_report["system"]["members"]},
                         "source_signature": context["signature"], "promoted": False,
                         "scope": plan["scope"], "full_train": False}
            if winning_report["system"]["head_weight"]:
                seed = winning_report["system"]["members"][0].rsplit("_", 1)[1]
                p = context["quality_output"] / f"tasks/head_train_primary_{seed}/last.pt"
                selection["head"] = {"path": str(p), "sha256": sha256(p)}
            result.update(selection=selection, candidate_diagnostics=diagnostic, status="complete")
            old.review.old.freeze_json(context["output"] / "selected_candidate.json", selection)
        except (KeyboardInterrupt, SystemExit, old.IntegrityError):
            result["status"] = "interrupted_or_integrity_error"
            raise
        except Exception:
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
