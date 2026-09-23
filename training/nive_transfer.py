"""Isolated NiVe1303 transfer pilot. Never edit shared trainers or organizer data."""
import fcntl
import gc
import platform
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from backend.core import ARTIFACTS, DATASET, MODEL, ROOT, STOCK_MODEL, read_rows, sha256
from backend.evaluate import SEED, make_protocol
from training import osnet_ablation_suite as suite
from training.audit import digest
from training.hpo import ExperimentConfig
from training.osnet_ablations import Ablation, AblationDataset, MemoryBank, initialize, losses, optimizer_for, transform_for
from training.pipeline import format_duration, set_seed
from training.stage6 import StepPKBatchSampler, audit_partitions, set_step_learning_rates, write_json


VARIANT = ROOT / "OSNet-AIN-x1.0/variant_15_nive_transfer"
NIVE = ROOT / "NiVe1303"
SOURCE_URL = "https://data.mendeley.com/datasets/42wv2svztx/1"
EXPECTED = {"train": (17070, 703), "test/query": (2887, 600), "test/gallery": (11778, 600)}
ARMS = {
    "B0_control": "Stock → organizer train",
    "C1_matched_budget": "Stock → organizer train, fixed pretrain → organizer train",
    "E1_nive_transfer": "Stock → NiVe train, fixed pretrain → organizer train",
}


def audit_nive(root, organizer_hashes):
    """Photo directories define splits. *_MK_PURE and upload sidecars are never training data."""
    root = Path(root).resolve()
    files, counts, identities, seen, train = {}, {}, {}, {}, []
    forbidden = set(organizer_hashes)
    for split, expected in EXPECTED.items():
        paths = sorted((root / split).glob("*/*.jpg"))
        ids = set()
        for path in tqdm(paths, desc=f"NiVe integrity: {split}"):
            if not path.resolve().is_relative_to(root):
                raise ValueError("NiVe image resolves outside its dataset root")
            identity, group = path.parent.name, path.stem.split("_")[0]
            if not identity.isdigit() or group not in {"EN", "ES", "N", "S"}:
                raise ValueError(f"Unknown NiVe identity/view naming: {path}")
            checksum = sha256(path)
            if checksum in forbidden:
                raise ValueError(f"External/organizer byte overlap: {path}")
            if checksum in seen:
                raise ValueError(f"Duplicate NiVe image: {path} / {seen[checksum]}")
            seen[checksum] = str(path)
            with Image.open(path) as image:
                if image.size != (256, 256) or image.mode != "RGB":
                    raise ValueError(f"Unexpected NiVe image format: {path}")
                image.verify()
            relative = str(path.relative_to(root))
            files[relative] = checksum
            ids.add(identity)
            if split == "train":
                train.append({"image_id": f"nive/{identity}/{path.stem}", "vehicle_id": f"nive:{identity}",
                              "path": relative, "camera_id": group})
        counts[split] = {"images": len(paths), "identities": len(ids)}
        identities[split] = ids
        if (len(paths), len(ids)) != expected:
            raise ValueError(f"Incomplete/different NiVe v1 {split}: {counts[split]}, expected {expected}")
    if identities["train"] & (identities["test/query"] | identities["test/gallery"]):
        raise ValueError("NiVe train/test identity leakage")
    if identities["test/query"] != identities["test/gallery"]:
        raise ValueError("NiVe query/gallery identity mismatch")
    return train, {"files": files, "files_fingerprint": digest(files), "counts": counts,
                   "duplicate_bytes": 0, "organizer_byte_overlap": 0,
                   "limitation": "Byte audit does not rule out near-duplicates; image headers verified",
                   "sampling_group": "Filename prefix is a view proxy, NOT a verified physical camera ID",
                   "masks": "Unused; their directory split disagrees with photo split"}


class NiVeDataset(Dataset):
    def __init__(self, rows, variant, root):
        self.rows, self.root = rows, Path(root)
        self.clean = transform_for(variant, augment=True)
        self.robust = transform_for(variant, robust=True, augment=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Filenames/identity/view/time never enter the network; full public crop only.
        if Path(row["path"]).parts[0] != "train":
            raise ValueError("Only NiVe train photos may enter a training loader")
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to((self.root / "train").resolve()):
            raise ValueError("NiVe path escapes train")
        with Image.open(path) as image:
            crop = image.convert("RGB")
        return self.clean(crop), self.robust(crop), row["label"], row["image_id"]


def prepare(run_name="nive_pilot_v1", device="mps", pretrain_budget=suite.Budget(850, 200, 100),
            target_budget=suite.Budget(), seeds=(SEED, SEED + 1, SEED + 2),
            source_confirmed=False, dataset=DATASET, nive=NIVE):
    """Audits are read-only; all generated artifacts belong to variant 15."""
    import onnxruntime
    import torchvision

    pretrain_budget.validate()
    target_budget.validate()
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Three distinct seeds must be fixed before the pilot")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("RUN_NAME must be a simple directory name")
    output = VARIANT / "runs" / run_name
    if not output.resolve().is_relative_to((VARIANT / "runs").resolve()):
        raise ValueError("Output must stay inside variant_15/runs")
    dataset, nive = Path(dataset).resolve(), Path(nive).resolve()
    rows, split = read_rows(dataset / "train.csv"), suite.load_json(ARTIFACTS / "splits.json")
    if sha256(dataset / "train.csv") != split["train_csv_sha256"]:
        raise ValueError("Organizer CSV changed")
    frames = {r["image_id"]: sha256(dataset / "images" / f"{r['image_id']}.jpg")
              for r in tqdm(rows, desc="Organizer integrity (no edits)")}
    if frames != split["frame_sha256"]:
        raise ValueError("Organizer images changed")
    audit_partitions(rows, frames, split["identities"])
    if sha256(STOCK_MODEL) != suite.STOCK_SHA256 or sha256(MODEL) != suite.MVP_SHA256:
        raise ValueError("Unexpected stock/MVP weights")
    baseline = suite.load_json(ARTIFACTS / "baseline_metrics.json")
    if (baseline["model_sha256"] != suite.MVP_SHA256
            or baseline["evaluator"]["sha256"] != sha256(ROOT / "evaluate.py")
            or {k: baseline["search"][k] for k in ("k1", "k2", "lambda")} !=
            {"k1": 20, "k2": 3, "lambda": .5}):
        raise ValueError("Frozen MVP/evaluator reference mismatch")
    masks, inner = suite.load_masks(rows, split, frames)
    protocols = {}
    for name, identities in {**{n: s["validation"] for n, s in inner.items()},
                             **{n: split["identities"][n] for n in ("calibration", "validation")}}.items():
        query, gallery = make_protocol(rows, identities, SEED)
        protocols[name] = {"query_ids": [r["image_id"] for r in query],
                           "gallery_ids": [r["image_id"] for r in gallery]}
        if name in split["protocols"] and protocols[name] != split["protocols"][name]:
            raise ValueError("Original outer protocol changed")
    nive_rows, inventory = audit_nive(nive, frames.values())
    base = ExperimentConfig(**suite.load_json(suite.RECIPE)["config"])
    variants = {n: Ablation(name=n, description=description) for n, description in ARMS.items()}
    sources = [ROOT / "training" / f"{n}.py" for n in
               ("nive_transfer", "osnet_ablations", "osnet_ablation_suite", "osnet", "hpo", "pipeline",
                "preprocessing", "stage6", "masked_hpo", "mask_reid_ablation", "audit")]
    sources += [ROOT / "backend" / f"{n}.py" for n in ("core", "evaluate", "scoring", "rerank")]
    protected = [STOCK_MODEL, MODEL, ARTIFACTS / "splits.json", ARTIFACTS / "baseline_metrics.json",
                 dataset / "train.csv", ROOT / "evaluate.py", ROOT / "ORGANIZER_QA.md", suite.RECIPE,
                 suite.MASK_ROOT / "cache/automatic_masks.json", suite.DETECTOR / "annotation/mask_plan.json"]
    manifest = {"version": 1, "pretrain_budget": asdict(pretrain_budget), "target_budget": asdict(target_budget),
                "seeds": list(seeds), "variants": {n: asdict(v) for n, v in variants.items()},
                "base_recipe": asdict(base), "outer": split["identities"], "inner": inner, "protocols": protocols,
                "frames_sha256": digest(frames), "train_csv_sha256": sha256(dataset / "train.csv"),
                "protected": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p)
                              for p in protected},
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                "nive": {**inventory, "source": {"url": SOURCE_URL, "doi": "10.17632/42wv2svztx.1",
                         "author": "Ruozheng LI", "version": 1, "license": "CC BY 4.0",
                         "local_copy_source_confirmed_by_user": bool(source_confirmed),
                         "archive_sha256": None, "access": "Public download; extracted files hashed individually"}},
                "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "torchvision": torchvision.__version__, "onnxruntime": onnxruntime.__version__,
                            "numpy": np.__version__, "device": str(device), "cuda": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version(),
                            "deterministic": torch.are_deterministic_algorithms_enabled()},
                "policy": {"pilot": "one primary seed; no outer evaluation",
                           "confirmation": "all 3 arms, 3 paired seeds, primary then alternate",
                           "pretrain_selection": "fixed last step, no validation; NiVe test never used",
                           "transfer": "backbone+BNNeck including buffers; fresh classifier/optimizer/LR",
                           "comparison": "matched optimizer updates and batch size, NOT matched wall time",
                           "final_steps": "per-arm median primary best step; fixed pretrain budget retained",
                           "final_seed": seeds[0], "annotation_edits": 0, "removed_rows": 0,
                           "outer_selection": False, "promoted": False}}
    with suite.run_lock(output):
        suite.freeze_json(output / "manifest.json", manifest)
    return {"output": output, "manifest": manifest, "signature": digest(manifest), "rows": rows,
            "split": split, "masks": masks, "base": base, "variants": variants, "seeds": seeds,
            "device": torch.device(device), "dataset": dataset, "nive_root": nive, "nive_rows": nive_rows,
            "budget": target_budget, "pretrain_budget": pretrain_budget,
            "protected": {str(p): sha256(p) for p in protected}}


def verify_inputs(context, rehash_frames=False):
    suite.verify_protected(context, rehash_frames=rehash_frames)
    if rehash_frames:
        _, inventory = audit_nive(context["nive_root"], suite.load_json(ARTIFACTS / "splits.json")["frame_sha256"].values())
        if inventory != {k: v for k, v in context["manifest"]["nive"].items() if k != "source"}:
            raise ValueError("NiVe inventory changed; use original files, not a silent repair")


def ensure_previous_suite_idle():
    """Check existing locks read-only; never create/write files in variant 14."""
    for path in (suite.VARIANT / "runs").glob("*/.lock"):
        with path.open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Wait for the active variant 14 run to finish: {path.parent.name}") from error
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def training_rows(context, arm, fold, stage):
    if stage == "pretrain" and arm == "E1_nive_transfer":
        rows = context["nive_rows"]
        source = "nive"
    else:
        identities = (context["split"]["identities"]["train"] if fold == "final" else
                      context["manifest"]["inner"][fold]["train"])
        allowed = set(identities)
        rows = [r for r in context["rows"] if r["vehicle_id"] in allowed]
        source = "organizer"
    labels = {identity: i for i, identity in enumerate(sorted({r["vehicle_id"] for r in rows}))}
    return [{**r, "label": labels[r["vehicle_id"]]} for r in rows], source, len(labels)


def transfer_representation(model, state):
    expected = {k for k in model.state_dict() if not k.startswith("classifier.")}
    selected = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    if selected.keys() != expected:
        raise ValueError("Pretraining representation architecture mismatch")
    result = model.load_state_dict(selected, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != {k for k in model.state_dict() if k.startswith("classifier.")}:
        raise ValueError("Only the identity classifier may be reset")


def fit_stage(context, arm, seed, fold="primary", stage="target", parent=None, final_steps=None):
    """Fixed source pretraining or inner-selected target training; resume at block boundaries."""
    if (arm not in context["variants"] or fold not in {"primary", "alternate", "final"}
            or stage not in {"pretrain", "target"} or seed not in context["seeds"]):
        raise ValueError("Unregistered training stage")
    if stage == "pretrain" and (arm == "B0_control" or parent is not None):
        raise ValueError("Pretraining is stock-start and only exists for C1/E1")
    needs_parent = stage == "target" and arm != "B0_control"
    if needs_parent != (parent is not None):
        raise ValueError("Wrong initialization: C1/E1 target requires its pretraining checkpoint")
    budget = context["pretrain_budget"] if stage == "pretrain" else context["budget"]
    if final_steps is not None and (fold != "final" or stage != "target"):
        raise ValueError("Final step override only applies to final target refit")
    if fold == "final" and stage == "target" and final_steps is None:
        raise ValueError("Freeze final steps on primary before final refit")
    stop = budget.max_steps if final_steps is None else int(final_steps)
    if not 0 <= stop <= budget.max_steps:
        raise ValueError("Invalid stop step")
    variant, device = context["variants"][arm], context["device"]
    config = variant.recipe(context["base"], seed)
    rows, source, classes = training_rows(context, arm, fold, stage)
    source_fold = "nive" if source == "nive" else fold
    phase = f"pretrain/{source_fold}" if stage == "pretrain" else fold
    directory = context["output"] / phase / arm / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    parent_hash = None
    if parent is not None:
        expected_phase = "pretrain/nive" if arm == "E1_nive_transfer" else f"pretrain/{fold}"
        if (parent["variant"] != arm or parent["seed"] != seed or parent["phase"] != expected_phase
                or parent["updates"] != context["pretrain_budget"].max_steps
                or parent["best_step"] != parent["updates"]
                or parent["context_signature"] != context["signature"]):
            raise ValueError("Wrong source/fold/seed pretraining checkpoint")
        parent_path = context["output"] / parent["checkpoint"]
        parent_hash = sha256(parent_path)
        if parent_hash != parent["checkpoint_sha256"]:
            raise ValueError("Pretraining checkpoint checksum mismatch")
    signature = digest({"context": context["signature"], "variant": asdict(variant), "seed": seed,
                        "phase": phase, "stop": stop, "rows": rows, "parent_sha256": parent_hash})
    last_path = directory / "last.pt"
    if not last_path.exists() and (directory / "summary.json").exists():
        raise ValueError("Missing authoritative last.pt; restore it or choose a new RUN_NAME")
    previous = torch.load(last_path, map_location="cpu", weights_only=True) if last_path.exists() else None
    if previous is not None and previous["signature"] != signature:
        raise ValueError("Checkpoint/configuration mismatch")
    set_seed(seed)
    model = initialize(classes, config, variant, device)
    if parent is not None:
        payload = torch.load(parent_path, map_location="cpu", weights_only=True)
        if payload["signature"] != parent["signature"]:
            raise ValueError("Pretraining checkpoint signature mismatch")
        transfer_representation(model, payload["model"])
        del payload
    # Fresh target optimizer and LR schedule; source optimizer is never transferred.
    optimizer, bank = optimizer_for(model, config), MemoryBank(0)
    select_inner = stage == "target" and fold != "final"
    if previous is not None:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        start, history, best = previous["step"], previous["history"], previous["best"]
        seen, elapsed = set(previous["seen"]), previous["elapsed"]
    else:
        initial = suite.evaluate_inner(model, context, variant, fold) if select_inner else None
        if initial is not None and not np.isfinite(initial["mAP_at_10"]):
            raise FloatingPointError("Non-finite initial inner mAP")
        best = {"step": 0, "map": initial["mAP_at_10"] if initial else None,
                "metrics": initial, "state": suite.cpu_copy(model.state_dict())}
        start, history, seen, elapsed = 0, [], set(), 0.
    dataset = (NiVeDataset(rows, variant, context["nive_root"]) if source == "nive" else
               AblationDataset(rows, variant, context["dataset"], augment=True))
    row_indices = {r["image_id"]: i for i, r in enumerate(rows)}
    for end in budget.boundaries(stop):
        if end <= start:
            continue
        set_seed(seed + start)
        sampler = StepPKBatchSampler(rows, config, end - start)
        sampler.set_epoch(start)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        totals, began = {}, time.perf_counter()
        for offset, (clean, robust, target, image_ids) in enumerate(loader):
            step = start + offset
            set_step_learning_rates(optimizer, config, budget, step)
            ids = torch.tensor([row_indices[i] for i in image_ids], device=device)
            values, _ = losses(model, clean.to(device), robust.to(device), target.to(device), ids,
                               config, variant, bank, step)
            if not torch.isfinite(values["loss"]):
                raise FloatingPointError(f"Non-finite loss: {phase}/{arm}/{step}")
            optimizer.zero_grad(set_to_none=True)
            values["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            seen.update(image_ids)
            for name, value in {**values, "gradient_norm": norm}.items():
                totals[name] = totals.get(name, 0.) + float(value.detach())
        validation = suite.evaluate_inner(model, context, variant, fold) if select_inner else None
        if validation is not None and not np.isfinite(validation["mAP_at_10"]):
            raise FloatingPointError("Non-finite inner mAP")
        seconds = time.perf_counter() - began
        elapsed += seconds
        if not select_inner or validation["mAP_at_10"] > best["map"]:
            best = {"step": end, "map": validation["mAP_at_10"] if validation else None,
                    "metrics": validation, "state": suite.cpu_copy(model.state_dict())}
        history.append({"step": end, "samples_seen": end * config.batch_size,
                        "unique_train_images": len(seen), "train": {k: v / (end - start) for k, v in totals.items()},
                        "validation": validation, "block_seconds": seconds, "elapsed_seconds": elapsed})
        suite.save_checkpoint(last_path, {"signature": signature, "step": end, "model": model.state_dict(),
                                         "optimizer": optimizer.state_dict(), "best": best, "history": history,
                                         "seen": sorted(seen), "elapsed": elapsed})
        write_json(directory / "history.json", history)
        score = f"inner mAP={validation['mAP_at_10']:.4f}, best={best['map']:.4f}" if validation else "fixed steps; no evaluation"
        print(f"{phase}/{arm}/{seed}: {end}/{stop} | {score} | elapsed {format_duration(elapsed)} | "
              f"ETA {format_duration(elapsed / end * (stop - end))}", flush=True)
        start = end
    if not last_path.exists():
        suite.save_checkpoint(last_path, {"signature": signature, "step": 0, "model": model.state_dict(),
                                         "optimizer": optimizer.state_dict(), "best": best, "history": history,
                                         "seen": [], "elapsed": 0.})
    path = directory / "best.pt"
    suite.save_checkpoint(path, {"signature": signature, "model": best["state"]})
    summary = {"variant": arm, "seed": seed, "phase": phase, "signature": signature,
               "context_signature": context["signature"], "best_step": best["step"], "inner_map": best["map"],
               "inner_metrics": best["metrics"], "updates": stop, "samples_seen": stop * config.batch_size,
               "train_images": len(rows), "train_identities": classes, "unique_images": len(seen),
               "elapsed_seconds": elapsed, "checkpoint": str(path.relative_to(context["output"])),
               "checkpoint_sha256": sha256(path), "source": source, "parent_sha256": parent_hash,
               "source_updates": parent["updates"] if parent else 0, "config": asdict(config), "ablation": asdict(variant)}
    write_json(directory / "summary.json", summary)
    write_json(directory / "history.json", history)
    del model, optimizer, previous, best
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def fit_arm(context, arm, seed, fold="primary", final_steps=None):
    parent = None if arm == "B0_control" else fit_stage(context, arm, seed, fold, stage="pretrain")
    return fit_stage(context, arm, seed, fold, parent=parent, final_steps=final_steps)


def write_report(context, runs, phase, selection=None, outer=None):
    lines = ["# NiVe → OSNet: " + phase, "", "BBox, исходная validation и MVP не изменены.",
             "NiVe test и маски частей кузова не используются для обучения/выбора.", "",
             "| Вариант | seed | inner mAP@10 | best target step | pretrain + target updates |",
             "|---|---:|---:|---:|---:|"]
    for s in runs:
        lines.append(f"| {s['variant']} | {s['seed']} | {s['inner_map']:.5f} | {s['best_step']} | "
                     f"{s['source_updates']} + {s['updates']} |")
    lines += ["", "C1/E1 имеют одинаковые бюджеты стадий, batch, reset classifier/optimizer/LR;",
              "C1 — двухэтапный контроль, не непрерывный длинный training run. Wall time может различаться."]
    if selection:
        lines += ["", f"Выбор по primary: {selection['winner']} (не автоматическое продвижение).", "",
                  "| Вариант | primary mean | std | final target steps |", "|---|---:|---:|---:|"]
        for name, a in selection["aggregate"].items():
            lines.append(f"| {name} | {a['mean']:.5f} | {a['std']:.5f} | {a['final_steps']} |")
        lines += ["", "Сравни E1 и с B0, и с C1. Paired seed deltas: selection.json; alternate: alternate.json."]
    if outer:
        lines += ["", "| Вариант / input | raw mAP@10 | reranked mAP@10 | F1 | TNR |", "|---|---:|---:|---:|---:|"]
        reference = suite.load_json(ARTIFACTS / "baseline_metrics.json")
        raw, rr = reference["raw_baseline"]["validation"], reference["validation"]
        lines.append(f"| MVP / frozen reference | {raw['mAP_at_10']:.5f} | {rr['mAP_at_10']:.5f} | "
                     f"{rr['candidate_F1']:.5f} | {rr['TNR']:.5f} |")
        for name, report in outer.items():
            for condition, value in report["conditions"].items():
                raw, rr = value["raw"], value["reranked"]
                lines.append(f"| {name} / {condition} | {raw['mAP_at_10']:.5f} | {rr['mAP_at_10']:.5f} | "
                             f"{rr['candidate_F1']:.5f} | {rr['TNR']:.5f} |")
    lines += ["", "Пилот на одном seed не доказывает улучшение. Validation — development, не независимый тест.",
              "Статистический bootstrap условен на фиксированной gallery; исходные пороги к новым весам не переносятся."]
    filename = "PILOT_RESULTS.md" if phase == "pilot" else "RESULTS.md"
    (context["output"] / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(context, phase="pilot"):
    if phase not in {"pilot", "confirm"}:
        raise ValueError("PHASE must be pilot or confirm")
    if not context["manifest"]["nive"]["source"]["local_copy_source_confirmed_by_user"]:
        raise ValueError("Confirm the public source of the local NiVe copy before training")
    device = context["device"]
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    ensure_previous_suite_idle()
    with suite.run_lock(context["output"]):
        verify_inputs(context, rehash_frames=True)
        seeds = context["seeds"][:1] if phase == "pilot" else context["seeds"]
        primary = [fit_arm(context, name, seed) for seed in seeds for name in context["variants"]]
        write_json(context["output"] / f"{phase}_primary.json", primary)
        if phase == "pilot":
            verify_inputs(context, rehash_frames=True)
            write_report(context, primary, phase)
            result = {"pilot_complete": True, "confirmation_complete": False, "outer_evaluated": False,
                      "promoted": False, "signature": context["signature"], "primary": primary}
            write_json(context["output"] / "pilot_summary.json", result)
            return result
        winner, aggregate = suite.select_winner(primary, context["seeds"])
        by_run = {(s["variant"], s["seed"]): s["inner_map"] for s in primary}
        deltas = {name: [by_run["E1_nive_transfer", seed] - by_run[name, seed] for seed in context["seeds"]]
                  for name in ("B0_control", "C1_matched_budget")}
        selection = {"signature": context["signature"], "winner": winner, "aggregate": aggregate,
                     "E1_paired_seed_deltas": deltas, "basis": "primary inner raw mAP only"}
        suite.freeze_json(context["output"] / "selection.json", selection)
        alternate = [fit_arm(context, name, seed, fold="alternate")
                     for seed in context["seeds"] for name in context["variants"]]
        write_json(context["output"] / "alternate.json", alternate)
        final = [fit_arm(context, name, context["seeds"][0], fold="final",
                         final_steps=aggregate[name]["final_steps"]) for name in context["variants"]]
        suite.freeze_json(context["output"] / "final_selection.json", final)
        outer = {}
        for summary in final:
            outer[summary["variant"]] = suite.evaluate_final(context, summary)
            suite.export_final(context, summary)
        paired = {}
        for name in ("B0_control", "C1_matched_budget"):
            paired[name] = {}
            for method in ("raw", "reranked"):
                reports = [suite.load_json(context["output"] / "final" / arm / f"seed_{context['seeds'][0]}" /
                                           f"per_query_{method}.json") for arm in (name, "E1_nive_transfer")]
                paired[name][method] = suite.paired_bootstrap(*reports)
        write_json(context["output"] / "paired_bootstrap.json", paired)
        verify_inputs(context, rehash_frames=True)
        write_report(context, primary, phase, selection, outer)
        result = {"signature": context["signature"], "selection": selection, "outer": outer,
                  "confirmation_complete": True, "outer_evaluated": True, "promoted": False}
        write_json(context["output"] / "summary.json", result)
        return result
