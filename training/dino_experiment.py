"""D0 fixed inner-only screening; no training, outer selection, threshold or MVP mutation."""
import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import copy
import importlib.metadata
from pathlib import Path
import platform
import re

import numpy as np
import torch

from training import dino_frozen as dino, nive_mixed as base, nive_system as previous

VARIANT = dino.VARIANT
SYSTEMS = ("R1_control", "D0_frozen", "R1_90_D0_10")


def check_inputs(context):
    m = context["manifest"]
    if base.digest(m) != context["signature"]:
        raise ValueError("Frozen D0 context changed")
    base.verify_files(m["protected"])
    base.verify_files(m["source_sha256"])
    if dino.model_files(m["model_directory"]) != m["model_files"]:
        raise ValueError("DINO weights or model card changed")


def prepare(model_directory, run_name="d0_v1", device="cpu"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a simple RUN_NAME")
    model_files = dino.model_files(model_directory)  # Fail clearly before any expensive preflight.
    deps = dino.activate_dependencies()
    dino.device_for(device)
    path = VARIANT / "configs/d0_v1.json"
    settings = base.old.load_json(path)
    if (settings["dino_weight"] != .1 or settings["image_size"] != 256 or settings["vector_atol"] != 2e-5
            or settings["graph"] != base.policy.POLICIES["less_graph"] or settings["optimizer_updates"]
            or settings["bn_updates"] or settings["threshold_fit"] or settings["original_outer_evaluation"]
            or settings["promoted"] or settings["wall_time_limit"] is not None):
        raise ValueError("D0 is one frozen construction, not a training or parameter search")
    source = base.ROOT / settings["source_run"]
    base.verify_files({str(source / "manifest.json"): settings["source_manifest_sha256"],
                       str(source / "results.json"): settings["source_results_sha256"]})
    m34, r34 = (base.old.load_json(source / name) for name in ("manifest.json", "results.json"))
    if r34["status"] != "complete" or r34["signature"] != base.digest(m34):
        raise ValueError("Need the completed v34 reference")
    inner_source = Path(m34["source_directory"])
    inner_manifest = base.old.load_json(inner_source / "manifest.json")
    if base.digest(inner_manifest) != m34["source_signature"]:
        raise ValueError("Inner parent provenance changed")
    rows = base.read_rows(base.DATASET / "train.csv")
    split = base.old.load_json(base.ARTIFACTS / "splits.json")
    base.parent_provenance(inner_manifest["plan"], rows, split)
    train_ids, holdout = set(inner_manifest["parent"]["train_ids"]), set(inner_manifest["inner"]["validation"])
    if train_ids & holdout or not holdout <= set(split["identities"]["train"]):
        raise ValueError("R1 saw the proposed inner holdout or the split uses original outer IDs")
    score_context = {"manifest": inner_manifest, "rows": rows}
    development = base.development_rows(score_context)
    reference = inner_source / "evaluation/N_ref"
    base.verify_files({str(reference / name): settings[key] for name, key in (
        ("features.npy", "reference_features_sha256"), ("order.json", "reference_order_sha256"),
        ("metrics.json", "reference_metrics_sha256"))})
    if base.old.load_json(reference / "order.json") != [r["image_id"] for r in development]:
        raise ValueError("R1 cached feature order differs from the frozen inner protocol")
    # Preserve existing artifacts; NiVe photographs are not an input/dependency of D0.
    nive_root = base.ROOT / "NiVe1303"
    protected = {p: h for p, h in m34["protected"].items() if not Path(p).is_relative_to(nive_root)}
    protected.update({str(p): base.sha256(p) for p in source.rglob("*") if p.is_file()})
    sources = {**m34["source_sha256"], **{str(p): base.sha256(p) for p in (
        Path(__file__).resolve(), Path(dino.__file__).resolve(), path, VARIANT / "requirements-d0.txt")}}
    runtime = {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "numpy", "pillow", *deps)}
    runtime.update(device=device, python=platform.python_version(), platform=platform.platform(),
                   torch_threads=torch.get_num_threads(), attention="eager", dtype="float32")
    manifest = {"version": 35, "settings": settings, "systems": list(SYSTEMS), "runtime": runtime,
        "model_id": dino.MODEL_ID, "revision": dino.REVISION, "model_files": model_files,
        "model_directory": str(Path(model_directory).resolve()), "preprocessing": copy.deepcopy(dino.PREPROCESS),
        "source_directory": str(source), "reference_directory": str(reference),
        "draws": inner_manifest["draws"], "inner_train_ids": sorted(train_ids), "inner_holdout_ids": sorted(holdout),
        "image_order": [r["image_id"] for r in development], "protected": protected, "source_sha256": sources,
        "source_scope": "fixed three inner draws; no active full-train v25 control on its seen inner IDs",
        "pretraining_overlap": "Unknown for opaque LVD corpus; no claim of externally audited disjointness",
        "threshold": None, "promoted": False, "optimizer_updates": 0}
    output = VARIANT / "runs" / run_name
    if not output.resolve().is_relative_to((VARIANT / "runs").resolve()):
        raise ValueError("Output escapes v35")
    context = {"manifest": manifest, "signature": base.digest(manifest), "output": output,
               "rows": development, "dataset": base.DATASET, "score_context": score_context}
    print(f"PREFLIGHT: D0 + fold-matched R1, {len(development)} inner images; no original outer evaluation", flush=True)
    check_inputs(context)
    with base.old.run_lock(output):
        base.old.freeze_json(output / "manifest.json", manifest)
    return context


def extract(context, encoder):
    def action(directory):
        blocks = []
        chunk, batch = (context["manifest"]["settings"][k] for k in ("chunk_size", "batch_size"))
        for start in range(0, len(context["rows"]), chunk):
            rows = context["rows"][start:start+chunk]
            path = directory / f"block_{start:05d}.npy"
            receipt = path.with_suffix(".json")
            identity = {"signature": context["signature"], "ids": [r["image_id"] for r in rows]}
            cached = receipt.exists()
            if cached:
                saved = base.old.load_json(receipt)
                if saved["identity"] != identity:
                    raise ValueError("DINO block fingerprint changed")
                base.verify_files({str(path): saved["sha256"]})
                values = np.load(path, allow_pickle=False)
            else:
                values = encoder.encode_rows(rows, context["dataset"], batch)
                pending = path.with_suffix(".pending.npy"); np.save(pending, values); pending.replace(path)
                base.write_json(receipt, {"identity": identity, "sha256": base.sha256(path)})
            dino.validate_block(values, 768)
            if len(values) != len(rows): raise ValueError("DINO block row count mismatch")
            blocks.append(values)
            print(f"D0 images: {min(start+chunk,len(context['rows']))}/{len(context['rows'])} | {'cache' if cached else 'fresh'}", flush=True)
        values = np.concatenate(blocks); path = directory / "features.npy"
        if path.exists() and not np.array_equal(values, np.load(path, allow_pickle=False)):
            raise ValueError("DINO aggregate changed")
        if not path.exists(): np.save(path, values)
        base.old.freeze_json(directory / "order.json", [r["image_id"] for r in context["rows"]])
        return {"path": str(path), "sha256": base.sha256(path), "rows": len(values), "dimension": 768}
    result = previous.task(context, "features_D0", action)
    base.verify_files({result["path"]: result["sha256"]})
    return np.load(result["path"], allow_pickle=False)


def mixed_features(r1, d0):
    dino.validate_block(r1, 512); dino.validate_block(d0, 768)
    if len(r1) != len(d0): raise ValueError("Mixed feature rows differ")
    return base.normalize(np.concatenate([r1*np.float32(np.sqrt(.9)), d0*np.float32(np.sqrt(.1))], axis=1))


def probe(context, encoder, r1, vectors):
    """True batch and stream invariance, evaluated on fixed actual images/gallery."""
    def check(directory):
        row_index = {r["image_id"]: i for i, r in enumerate(context["rows"])}
        protocol = next(iter(context["manifest"]["draws"].values()))
        qi = [row_index[i] for i in protocol["query_ids"][:32]]
        gi = [row_index[i] for i in protocol["gallery_ids"]]
        images = [context["rows"][i] for i in qi]
        systems = {"D0_frozen": vectors, "R1_90_D0_10": mixed_features(r1, vectors)}
        expected = {n: base.policy.rank_vectors(v[qi], v[gi], "less_graph") for n, v in systems.items()}
        cases = [(np.arange(len(qi)), size) for size in (1, 8, 16, 32)]
        cases += [(np.arange(len(qi))[::-1], 16), (np.array([0]), 1)]
        max_error = 0.
        for indices, batch in cases:
            actual = encoder.encode_rows([images[i] for i in indices], context["dataset"], batch)
            ref = vectors[np.asarray(qi)[indices]]
            error = float(abs(actual-ref).max()); max_error = max(max_error, error)
            if not np.allclose(actual, ref, rtol=0, atol=2e-5):
                raise ValueError("DINO batch/permutation feature drift; fixed tolerance, no fallback")
            q_vectors = {"D0_frozen": actual, "R1_90_D0_10": mixed_features(r1[np.asarray(qi)[indices]], actual)}
            for name, q in q_vectors.items():
                current = base.policy.rank_vectors(q, systems[name][gi], "less_graph")
                for mode in ("order", "raw_order"):
                    if not np.array_equal(current[mode][:, :10], expected[name][mode][indices, :10]):
                        raise ValueError(f"DINO image batching/permutation changed {name}/{mode} top10")
            print(f"D0 stream probe: {len(indices)} queries, batch {batch}, error {error:.3g}", flush=True)
        return {"status": "passed", "max_vector_error": max_error, "batch_sizes": [1, 8, 16, 32],
                "query_permutation_and_removal": True, "exact_raw_and_graph_top10": True}
    return previous.task(context, "stream_probe", check)


def comparison(before, after):
    result = {}
    for name, left in before["draws"].items():
        right = after["draws"][name]
        if set(left["per_query"]) != set(right["per_query"]):
            raise ValueError("Unpaired D0 results")
        modes = {}
        for mode in ("raw", "ranking"):
            ids = [k for k, v in left["per_query"].items() if v["known"]]
            delta = np.array([right["per_query"][k][mode]["ap"] - left["per_query"][k][mode]["ap"] for k in ids])
            errors = [{k for k in ids if source["per_query"][k][mode]["ap"] < 1-1e-12} for source in (left, right)]
            modes[mode] = {"delta_map": float(delta.mean()), "better": int((delta > 1e-12).sum()),
                "worse": int((delta < -1e-12).sum()), "equal": int((abs(delta) <= 1e-12).sum()),
                "error_intersection": len(errors[0] & errors[1]), "fixed_perfect": len(errors[0]-errors[1]),
                "new_errors": len(errors[1]-errors[0])}
        result[name] = modes
    return result


def run(context):
    check_inputs(context)
    with base.old.run_lock(context["output"]):
        reference_dir = Path(context["manifest"]["reference_directory"])
        r1 = np.load(reference_dir / "features.npy", allow_pickle=False)
        dino.validate_block(r1, 512)
        encoder = dino.FrozenDino(context["manifest"]["model_directory"], context["manifest"]["runtime"]["device"])
        # Parameters/buffers must not be changed by either extraction or probes.
        initial = {k: v.detach().cpu().clone() for k, v in encoder.model.state_dict().items()}
        values = extract(context, encoder)
        stream = probe(context, encoder, r1, values)
        if any(not torch.equal(initial[k], v.detach().cpu()) for k, v in encoder.model.state_dict().items()):
            raise ValueError("Frozen backbone state changed")
        del initial, encoder
        vectors = {"R1_control": r1, "D0_frozen": values, "R1_90_D0_10": mixed_features(r1, values)}
        reports = {name: previous.task(context, f"evaluate_{name}", lambda directory, v=v: base.score_features(
            context["score_context"], v, context["rows"])) for name, v in vectors.items()}
        if reports["R1_control"] != base.old.load_json(reference_dir / "metrics.json"):
            raise ValueError("Fold-matched R1 no longer reproduces original metrics and exact top10")
        comparisons = {n: comparison(reports["R1_control"], reports[n]) for n in SYSTEMS[1:]}
        check_inputs(context)
        result = {"status": "complete", "signature": context["signature"], "reports": reports,
            "comparisons": comparisons, "stream_probe": stream, "optimizer_updates": 0, "bn_updates": 0,
            "threshold_fit": False, "original_outer_evaluation": False, "promoted": False,
            "protected_unchanged": True, "backbone_state_unchanged": True,
            "scope": "D0 inner development screening, not an active v25 or hidden-test comparison"}
        lines = ["# D0 — замороженный DINOv3 и R1", "", "Обучения, подбора порога, original validation и переключения MVP не было.",
                 "Три прежних inner draw используют одни и те же 185 holdout-ID: это не три независимых теста.", "",
                 "| Система | Mean raw mAP@10 | Mean graph mAP@10 | Δ raw к R1, п.п. | Δ graph к R1, п.п. |", "|---|---:|---:|---:|---:|"]
        ref = reports["R1_control"]
        for name, r in reports.items():
            lines.append(f"| {name} | {r['mean_raw_map']:.6f} | {r['mean_map']:.6f} | "
                         f"{100*(r['mean_raw_map']-ref['mean_raw_map']):+.4f} | {100*(r['mean_map']-ref['mean_map']):+.4f} |")
        lines += ["", "| Draw | Система | Raw mAP | Graph mAP | Raw Hit@50 | Graph Hit@50 |", "|---|---|---:|---:|---:|---:|"]
        for name, r in reports.items():
            for draw, v in r["draws"].items():
                lines.append(f"| {draw} | {name} | {v['raw']['mAP@10']:.6f} | {v['ranking']['mAP@10']:.6f} | "
                             f"{v['raw']['Hit@50']:.6f} | {v['ranking']['Hit@50']:.6f} |")
        lines += ["", "Фиксированная смесь: 90% cosine fold-matched R1 + 10% DINO. Граф 20/3/0.75, без подбора.",
            "DINO: CLS384 + patch-mean384, отдельно L2; register tokens исключены. Обе модели получают только bbox RGB.",
            "Per-query AP, top10, Hit@10/50 и пересечение ошибок сохранены в results.json.",
            "Ни порог, ни candidate-policy здесь не выбирались: screening оценивает ranking, а не готовый конкурсный профиль.",
            "R1 не обучался на этом holdout. Active full-train v25 здесь не используется: он видел часть этих ID.",
            "Состав LVD-предобучения непрозрачен; полное отсутствие внешнего overlap и устойчивость к номерному сигналу не доказаны.",
            "Слабый одиночный D0 сам по себе не закрывает добавку/небольшую обучаемую голову D1. Автоматического D1 нет.",
            "Результат не сравнивается напрямую с 0.828957 v25 на другом протоколе. Для полной системы нужен отдельный этап."]
        report = "\n".join(lines)+"\n"; path = context["output"] / "REPORT.md"
        if path.exists() and path.read_text(encoding="utf-8") != report: raise ValueError("D0 report changed")
        if not path.exists():
            pending = path.with_suffix(".md.tmp"); pending.write_text(report, encoding="utf-8"); pending.replace(path)
        base.old.freeze_json(context["output"] / "results.json", result)
        print("D0 complete. Active MVP unchanged. Report:", path, flush=True)
    return result
