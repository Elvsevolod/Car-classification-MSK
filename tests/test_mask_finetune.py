"""No user annotations are modified; YOLO smoke training uses synthetic images only."""
import copy
import json
import os
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from backend.core import sha256
from training import mask_finetune as data
from training import yolo_masks as training


@pytest.fixture
def sample(tmp_path):
    dataset = tmp_path / "source"
    (dataset / "images").mkdir(parents=True)
    rows, images = [], {}
    for n, split in enumerate(("train", "train", "train", "train", "val", "val", "holdout", "holdout")):
        image_id = f"image{n}"
        image = Image.new("RGB", (96, 80), (30+n*10, 80, 50))
        ImageDraw.Draw(image).rectangle((22, 24, 41, 39), fill=(110, 110, 110))
        image.save(dataset / "images" / f"{image_id}.jpg")
        rows.append(f"{image_id},10,10,64,64,{n},{n%2}")
        images[image_id] = {"bbox": [10, 10, 64, 64], "width": 64, "height": 64,
            "vehicle_id": n, "camera_id": n%2, "split": split,
            "frame_sha256": sha256(dataset / "images" / f"{image_id}.jpg")}
    (dataset / "train.csv").write_text("image_id,x,y,w,h,vehicle_id,camera_id\n"+"\n".join(rows))
    plan = {"images": images, "train_csv_sha256": sha256(dataset / "train.csv")}
    plan["fingerprint"] = data.digest(plan)
    annotations = {"version": 1, "fingerprint": plan["fingerprint"], "images": {
        i: {"reviewed": True, "rectangles": [[12, 14, 32, 30]]} for i in images}}
    plan_path, masks_path = tmp_path / "plan.json", tmp_path / "masks.json"
    data.write_json(plan_path, plan)
    data.write_json(masks_path, annotations)
    return dataset, plan_path, masks_path, plan, annotations


def test_selection_deterministic_and_connected_frames_stay_together():
    rows = [{"image_id": f"{i}-{j}", "vehicle_id": i, "camera_id": j,
             "w": 100+i*10, "h": 90+i*5} for i in range(40) for j in range(2)]
    hashes = {r["image_id"]: r["image_id"] for r in rows}
    hashes["1-0"] = hashes["0-0"]
    hashes["2-1"] = hashes["1-1"]  # Transitive linkage across distinct frames.
    counts = {"train": 12, "val": 4, "holdout": 4}
    selected, partitions = data.select_rows(rows, hashes, counts)
    assert (selected, partitions) == data.select_rows(rows, hashes, counts)
    owners = {identity: name for name, ids in partitions.items() for identity in ids}
    assert owners[0] == owners[1] == owners[2]
    selected_ids = set()
    selected_frames = set()
    for name, items in selected.items():
        assert len(items) == counts[name]
        for row in items:
            assert row["image_id"] not in selected_ids
            assert hashes[row["image_id"]] not in selected_frames
            assert owners[row["vehicle_id"]] == name
            selected_ids.add(row["image_id"])
            selected_frames.add(hashes[row["image_id"]])


def test_yolo_coordinates_use_crop_not_frame():
    assert data.yolo_line([10, 20, 30, 60], 100, 200) == "0 0.200000000 0.200000000 0.200000000 0.200000000"


@pytest.mark.parametrize("problem", ["unreviewed", "missing", "wrong_sample", "outside", "duplicate", "bool_coordinate"])
def test_export_rejects_bad_labels_before_creating_output(sample, tmp_path, problem):
    dataset, plan_path, masks_path, plan, annotations = sample
    if problem == "unreviewed": annotations["images"]["image0"]["reviewed"] = False
    if problem == "missing": del annotations["images"]["image0"]
    if problem == "wrong_sample": annotations["fingerprint"] = "old-audit"
    if problem == "outside": annotations["images"]["image0"]["rectangles"] = [[0, 0, 65, 64]]
    if problem == "duplicate": annotations["images"]["image0"]["rectangles"] *= 2
    if problem == "bool_coordinate": annotations["images"]["image0"]["rectangles"] = [[False, 0, 20, 20]]
    data.write_json(masks_path, annotations)
    with pytest.raises(ValueError):
        data.export_reviewed(plan_path, masks_path, tmp_path / "export", dataset)
    assert not (tmp_path / "export").exists()


def test_export_negative_requires_review_and_holdout_not_in_training_yaml(sample, tmp_path):
    dataset, plan_path, masks_path, plan, annotations = sample
    annotations["images"]["image0"]["rectangles"] = []
    data.write_json(masks_path, annotations)
    target = tmp_path / "export"
    manifest = data.export_reviewed(plan_path, masks_path, target, dataset)
    assert manifest["splits"]["train"]["negative_images"] == 1
    assert (target / "labels/train/image0.txt").read_text() == ""
    assert Image.open(target / "images/train/image0.png").size == (64, 64)
    assert data.load_json(target / "data.yaml")["val"] == "images/val"
    assert "test" not in data.load_json(target / "data.yaml")
    assert data.load_json(target / "holdout.yaml")["val"] == "images/test"
    assert data.export_reviewed(plan_path, masks_path, target, dataset) == manifest
    (target / "labels/train/image0.txt").write_text("tampered")
    with pytest.raises(ValueError, match="data changed"):
        data.export_reviewed(plan_path, masks_path, target, dataset)


def test_export_changed_annotations_cannot_overwrite(sample, tmp_path):
    dataset, plan_path, masks_path, plan, annotations = sample
    data.export_reviewed(plan_path, masks_path, tmp_path / "export", dataset)
    annotations["images"]["image0"]["rectangles"] = [[1, 1, 3, 3]]
    data.write_json(masks_path, annotations)
    with pytest.raises(ValueError, match="Annotations changed"):
        data.export_reviewed(plan_path, masks_path, tmp_path / "export", dataset)


def test_changed_image_or_fingerprint_rejected(sample):
    dataset, _, _, plan, _ = sample
    data.verify_sample(plan, dataset)
    changed = copy.deepcopy(plan)
    changed["images"]["image0"]["bbox"][0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        data.verify_sample(changed, dataset)
    Image.new("RGB", (96, 80)).save(dataset / "images/image0.jpg")
    with pytest.raises(ValueError, match="Frame changed"):
        data.verify_sample(plan, dataset)


def test_annotator_keeps_exact_sizes_and_has_no_holdout_proposals(sample, tmp_path):
    dataset, _, _, plan, _ = sample
    proposals = {i: {"deepmosaics": [[12, 14, 32, 30]], "yolo11n": []}
                 for i, item in plan["images"].items() if item["split"] != "holdout"}
    data.write_annotator(plan, proposals, tmp_path, dataset)
    page = (tmp_path / "annotate_masks.html").read_text()
    payload = json.loads(page.split("const data = ", 1)[1].split(";\nconst plan=", 1)[0])
    assert payload["plan"] == plan
    assert len(payload["pictures"]) == 8
    assert set(payload["suggestions"]) == set(list(plan["images"])[:6])
    assert 'reviewed:false' in page and 'id="unreviewed"' in page


def test_missing_review_blocks_before_loading_model(monkeypatch, tmp_path):
    def forbidden():
        raise AssertionError("Runtime must not load")
    monkeypatch.setattr(training, "runtime", forbidden)
    with pytest.raises(ValueError, match="240"):
        training.train_masks(tmp_path / "missing.json")


def test_epoch_timer_does_not_duplicate_final_validation(tmp_path):
    timer = training.EpochTimer(tmp_path / "times.json")
    trainer = SimpleNamespace(epoch=0, epochs=2, metrics={"mAP": .5})
    timer.start(trainer)
    timer.end(trainer)
    timer.end(trainer)
    assert len(data.load_json(tmp_path / "times.json")) == 1
    assert timer.records[0]["epoch"] == 1
    assert timer.records[0]["epoch_seconds"] >= 0


def test_notebook_is_clean_and_training_is_opt_in():
    import nbformat
    notebook = nbformat.read(data.EXPERIMENT / "train_yolo11_masks.ipynb", as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'RUN_TRAINING = False' in source and 'RUN_HOLDOUT = False' in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, "notebook", "exec")


@pytest.mark.skipif(os.environ.get("RUN_YOLO_SMOKE") != "1", reason="Explicit synthetic-only integration check")
def test_real_yolo_one_epoch_on_synthetic_data(sample, tmp_path):
    dataset, plan_path, masks_path, _, _ = sample
    target = tmp_path / "export"
    data.export_reviewed(plan_path, masks_path, target, dataset)
    YOLO = training.runtime()
    model = YOLO(str(data.EXPERIMENT / "weights/pretrained/yolo11n.pt"))
    timer = training.EpochTimer(tmp_path / "times.json")
    model.add_callback("on_train_epoch_start", timer.start)
    model.add_callback("on_fit_epoch_end", timer.end)
    args = {**training.TRAIN_ARGS, "epochs": 1, "imgsz": 64, "batch": 2}
    model.train(data=str(target / "data.yaml"), device=os.environ.get("YOLO_SMOKE_DEVICE", "cpu"),
                project=str(tmp_path), name="smoke", **args)
    assert (tmp_path / "smoke/weights/best.pt").is_file()
    assert (tmp_path / "smoke/weights/last.pt").is_file()
    assert len(data.load_json(tmp_path / "times.json")) == 1
    assert model.names == {0: "anonymized_region"}
