"""Local YOLO11 fine-tuning. Import never installs, downloads, or starts training."""
import os
import sys
import time
from pathlib import Path

from backend.core import ROOT, sha256
from training.mask_finetune import EXPERIMENT, export_reviewed, load_json, verify_sample
from training.stage6 import write_json

VERSION = "8.3.241"
TRAIN_ARGS = dict(epochs=50, patience=10, imgsz=640, batch=8, workers=0, optimizer="AdamW",
                  lr0=.001, lrf=.05, weight_decay=.0005, cos_lr=True, warmup_epochs=3,
                  seed=20260920, deterministic=True, amp=False, cache=False,
                  mosaic=0., mixup=0., copy_paste=0., degrees=0., translate=.05,
                  scale=.15, shear=0., perspective=0., flipud=0., fliplr=.5,
                  hsv_h=.015, hsv_s=.3, hsv_v=.15, save=True, save_period=-1, plots=True)


def runtime():
    """Reuse the existing torch; keep extra libraries and settings inside artifacts."""
    for folder in ("mask_detector_runtime", "yolo11_runtime"):
        path = str(ROOT / "artifacts" / folder)
        if path not in sys.path:
            sys.path.insert(0, path)
    config_dir = ROOT / "artifacts/yolo11_settings"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/mask_plot_cache"))
    # Check plotting dependencies before spending time on training, not at its end.
    import importlib
    for module in ("scipy.ndimage", "polars", "cv2"):
        importlib.import_module(module)
    import ultralytics
    from ultralytics import settings
    if ultralytics.__version__ != VERSION:
        raise RuntimeError(f"Use ultralytics=={VERSION}; found {ultralytics.__version__}")
    keys = ("sync", "hub", "wandb", "mlflow", "comet", "clearml", "neptune", "dvc", "tensorboard", "raytune")
    settings.update({k: False for k in keys if k in settings})
    return ultralytics.YOLO


def device_name():
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EpochTimer:
    """Counts train+validation; ignores final best.pt evaluation duplicate callback."""
    def __init__(self, path):
        self.path = Path(path)
        self.records = load_json(path) if self.path.exists() else []
        self.started = None

    def start(self, trainer):
        self.started = time.perf_counter()

    def end(self, trainer):
        if self.started is None:
            return
        seconds = time.perf_counter() - self.started
        self.started = None
        epoch = trainer.epoch + 1
        remaining = max(0, trainer.epochs - epoch)
        record = {"epoch": epoch, "epoch_seconds": seconds,
                  "metrics": {k: float(v) for k, v in trainer.metrics.items()}}
        self.records = [r for r in self.records if r["epoch"] < epoch] + [record]
        write_json(self.path, self.records)
        mean = sum(r["epoch_seconds"] for r in self.records) / len(self.records)
        print(f"Эпоха {epoch}/{trainer.epochs}: {seconds/60:.2f} мин · осталось до {remaining} эпох · "
              f"ETA до лимита {mean*remaining/60:.1f} мин (early stopping может закончить раньше)", flush=True)


def train_masks(annotations, run_name="pilot_01", resume=False, device=None):
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run_name must be a single directory name")
    plan_path = EXPERIMENT / "annotation/mask_plan.json"
    if not Path(annotations).is_file():
        raise ValueError("Сначала проверьте все 240 кадров, скачайте JSON и задайте ANNOTATIONS")
    # This gate runs before importing YOLO or loading any checkpoint.
    manifest = export_reviewed(plan_path, annotations, EXPERIMENT / "data")
    provenance = load_json(EXPERIMENT / "weights/pretrained/model.json")
    pretrained = EXPERIMENT / "weights/pretrained/yolo11n.pt"
    if sha256(pretrained) != provenance["sha256"]:
        raise ValueError("Pretrained weights checksum mismatch")
    run = EXPERIMENT / "runs" / run_name
    signature = {"plan_fingerprint": manifest["plan_fingerprint"],
                 "annotations_sha256": manifest["annotations_sha256"],
                 "pretrained_sha256": provenance["sha256"], "ultralytics": VERSION, "args": TRAIN_ARGS}
    if resume:
        if not (run / "weights/last.pt").exists() or load_json(run / "experiment.json")["signature"] != signature:
            raise ValueError("Resume requires matching labels/config and existing last.pt")
        if load_json(run / "experiment.json")["status"] == "complete":
            raise ValueError("Run already completed; choose a new run name for a new experiment")
    elif run.exists():
        raise ValueError("Run exists. Use resume=True after interruption, or a new run name")
    YOLO = runtime()
    device = device or device_name()
    run.mkdir(parents=True, exist_ok=True)
    state = {"signature": signature, "device": device, "status": "running"}
    write_json(run / "experiment.json", state)
    print(f"Device: {device}; train/val only, holdout not used. Output: {run}")
    model = YOLO(str(run / "weights/last.pt" if resume else pretrained))
    timer = EpochTimer(run / "epoch_times.json")
    model.add_callback("on_train_epoch_start", timer.start)
    model.add_callback("on_fit_epoch_end", timer.end)
    try:
        if resume:
            model.train(resume=True, device=device)
        else:
            model.train(data=str(EXPERIMENT / "data/data.yaml"), project=str(run.parent),
                        name=run.name, exist_ok=True, device=device, **TRAIN_ARGS)
    except BaseException:
        write_json(run / "experiment.json", {**state, "status": "interrupted_or_failed"})
        raise
    write_json(run / "experiment.json", {**state, "status": "complete", "best_sha256": sha256(run / "weights/best.pt")})
    return run


def evaluate_masks(weights, annotations, output, split="val", confidence=.25, margin=0., device=None):
    """Threshold/margin choices belong on val; holdout must be explicitly requested."""
    from training.audit import validate_annotations
    from training.mask_detection import compare_regions, restore_boxes, summarize
    from training.mask_finetune import new_directory

    if split not in ("val", "holdout") or not 0 < confidence < 1 or not 0 <= margin <= .25:
        raise ValueError("Invalid evaluation split/confidence/margin")
    plan = load_json(EXPERIMENT / "annotation/mask_plan.json")
    verify_sample(plan)
    manual = validate_annotations(plan, load_json(annotations))
    manifest = export_reviewed(EXPERIMENT / "annotation/mask_plan.json", annotations, EXPERIMENT / "data")
    output = new_directory(output)
    model = runtime()(str(weights))
    folder = "test" if split == "holdout" else "val"
    predicted, per_image = {}, {}
    for i, item in plan["images"].items():
        if item["split"] != split:
            continue
        result = model.predict(str(EXPERIMENT / "data/images" / folder / f"{i}.png"),
                               imgsz=640, conf=confidence, iou=.45, device=device or device_name(), verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy().copy()
        scores = result.boxes.conf.cpu().numpy()
        grow = (boxes[:, 2:] - boxes[:, :2]) * margin
        boxes[:, :2] -= grow
        boxes[:, 2:] += grow
        rects, conf = restore_boxes(boxes, scores, 1., (0, 0), (item["width"], item["height"]))
        predicted[i] = {"reviewed": False, "rectangles": rects, "confidences": conf}
        per_image[i] = compare_regions((item["width"], item["height"]), manual[i]["rectangles"], rects, conf)
    negative_ids = [i for i in predicted if not manual[i]["rectangles"]]
    summary = summarize(per_image)
    summary["negative_images"] = len(negative_ids)
    summary["negative_images_with_false_masks"] = sum(bool(predicted[i]["rectangles"]) for i in negative_ids)
    summary["regions_covered_90pct_fraction"] = summary["regions_covered_90pct"] / summary["reference_regions"]
    standard = model.val(data=str(EXPERIMENT / "data" / ("holdout.yaml" if split == "holdout" else "data.yaml")),
                         imgsz=640, batch=8, workers=0, device=device or device_name(), plots=False,
                         project=str(output), name="standard_metrics", exist_ok=True, verbose=False)
    report = {"split": split, "weights_sha256": sha256(weights), "annotations_sha256": manifest["annotations_sha256"],
              "confidence": confidence, "margin_fraction_each_side": margin, "iou_nms": .45,
              "coverage": summary, "standard_unexpanded_detector": {k: float(v) for k, v in standard.results_dict.items()},
              "note": "Mask coverage uses fixed confidence/margin; standard mAP uses raw detector boxes and its confidence sweep"}
    write_json(output / "predictions.json", {"fingerprint": plan["fingerprint"], "images": predicted})
    write_json(output / "metrics.json", report)
    return report
