"""Fast CPU regression tests; no external downloads and no full-model training."""
import copy
import base64
import io
import json
import random
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageOps

torch = pytest.importorskip("torch")
from training import audit, clip_experiment as experiment
from training.clip_reid import prompt_loss, triplet_loss


def test_opaque_rectangles_are_exact_and_do_not_change_original():
    original = Image.new("RGB", (10, 8), (123, 45, 67))
    result = np.asarray(audit.opaque_mask(original, [[2, 3, 6, 7]]))
    assert (result[3:7, 2:6] == 0).all()
    assert (result[2, 2] == [123, 45, 67]).all()
    assert (result[7, 6] == [123, 45, 67]).all()
    assert (np.asarray(original) == [123, 45, 67]).all()


def test_annotation_validation_requires_every_query_and_gallery_reviewed():
    plan = {"fingerprint": "a", "images": {"q": {"width": 10, "height": 8},
                                           "g": {"width": 10, "height": 8}}}
    valid = {"fingerprint": "a", "images": {
        "q": {"reviewed": True, "rectangles": [[0, 0, 10, 8]]},
        "g": {"reviewed": True, "rectangles": []}}}
    assert audit.validate_annotations(plan, valid) == valid["images"]
    invalid = copy.deepcopy(valid)
    invalid["fingerprint"] = "old"
    with pytest.raises(ValueError, match="fingerprint"):
        audit.validate_annotations(plan, invalid)
    invalid = copy.deepcopy(valid)
    del invalid["images"]["g"]
    with pytest.raises(ValueError, match="every selected"):
        audit.validate_annotations(plan, invalid)
    for rectangle in ([0, 0, 11, 8], [0, 0, 0, 8], [0., 0, 10, 8], [False, 0, 10, 8]):
        invalid = copy.deepcopy(valid)
        invalid["images"]["q"]["rectangles"] = [rectangle]
        with pytest.raises(ValueError):
            audit.validate_annotations(plan, invalid)
    invalid = copy.deepcopy(valid)
    invalid["images"]["g"]["reviewed"] = False
    with pytest.raises(ValueError, match="Unreviewed"):
        audit.validate_annotations(plan, invalid)


def test_paired_audit_rejects_changed_source_before_encoding(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (10, 8)).save(tmp_path / "images/a.jpg")
    rows = [{"image_id": "a", "x": 0, "y": 0, "w": 10, "h": 8}]
    plan = {"fingerprint": "a", "images": {"a": {"width": 10, "height": 8,
            "bbox": [0, 0, 10, 8], "frame_sha256": "old"}}}
    annotations = {"fingerprint": "a", "images": {"a": {"reviewed": True, "rectangles": [[0, 0, 2, 2]]}}}
    with pytest.raises(ValueError, match="changed since annotation"):
        audit.paired_mask_audit(rows, plan, annotations, tmp_path, dataset=tmp_path, threshold=.5)


@pytest.mark.parametrize("orientation", [1, 6])
def test_mask_annotator_adds_full_frame_without_changing_existing_masks(tmp_path, orientation):
    (tmp_path / "images").mkdir()
    source = Image.new("RGB", (1600, 900), (123, 45, 67))
    exif = Image.Exif()
    exif[274] = orientation
    source.save(tmp_path / "images/q.jpg", exif=exif)
    with Image.open(tmp_path / "images/q.jpg") as image:
        frame = ImageOps.exif_transpose(image).convert("RGB")
    plan = {"images": {"q": {"bbox": [100, 200, 300, 150], "width": 300, "height": 150,
                            "frame_sha256": audit.sha256(tmp_path / "images/q.jpg")}}}
    plan["fingerprint"] = audit.digest(plan)
    masks = {"fingerprint": plan["fingerprint"], "images": {
        "q": {"reviewed": True, "rectangles": [[10, 20, 30, 40]]}}}
    audit.write_json(tmp_path / "mask_plan.json", plan)
    audit.write_json(tmp_path / "masks.json", masks)
    protected = {path: path.read_bytes() for path in (
        tmp_path / "mask_plan.json", tmp_path / "masks.json", tmp_path / "images/q.jpg")}

    audit.write_mask_annotator(plan, tmp_path, dataset=tmp_path)
    page = (tmp_path / "annotate_masks.html").read_text()
    payload = json.loads(page.split("const data = ", 1)[1].split(";\nconst plan", 1)[0])
    assert payload["plan"] == plan
    assert payload["pictures"]["q"] == audit._picture(frame.crop((100, 200, 400, 350)))
    context = payload["frames"]["q"]
    assert (context["width"], context["height"]) == frame.size
    preview = Image.open(io.BytesIO(base64.b64decode(context["picture"].split(",", 1)[1])))
    assert preview.size == ((640, 360) if orientation == 1 else (270, 480))
    assert 'id="frame"' in page and 'id="canvas"' in page
    assert audit.validate_annotations(plan, masks) == masks["images"]
    assert all(path.read_bytes() == content for path, content in protected.items())

    changed = copy.deepcopy(plan)
    changed["images"]["q"]["frame_sha256"] = "changed"
    with pytest.raises(ValueError, match="Image changed"):
        audit.write_mask_annotator(changed, tmp_path, dataset=tmp_path)
    assert (tmp_path / "annotate_masks.html").read_text() == page


def test_prompt_loss_matches_official_multi_positive_raw_dot_products():
    torch.manual_seed(7)
    images = torch.randn(4, 5)
    texts = torch.randn(4, 5, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    positive = labels[:, None].eq(labels[None, :]).float()
    logits = images @ texts.T
    expected = sum(-((torch.log_softmax(matrix, 1) * positive).sum(1) / 2).mean()
                   for matrix in (logits, logits.T))
    actual = prompt_loss(images, texts, labels)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert texts.grad is not None and torch.isfinite(texts.grad).all()


def test_triplet_matches_official_unnormalized_euclidean_margin():
    torch.manual_seed(8)
    vectors = torch.randn(8, 6, requires_grad=True)
    labels = torch.arange(4).repeat_interleave(2)
    distances = torch.cdist(vectors, vectors)
    positive = labels[:, None].eq(labels[None, :])
    expected = torch.relu(distances.masked_fill(~positive, -torch.inf).max(1).values
                          - distances.masked_fill(positive, torch.inf).min(1).values + .3).mean()
    loss = triplet_loss(vectors, labels)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(vectors.grad).all()
    with pytest.raises(ValueError, match="P>=2"):
        triplet_loss(vectors[:2], torch.tensor([0, 0]))


class TinyPrompt(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_ctx = torch.nn.Parameter(torch.randn(2, 3))


class TinyClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Linear(2, 3)
        self.classifier = torch.nn.Linear(3, 2)
        self.prompt_learner = TinyPrompt()

    def text(self, labels):
        return self.prompt_learner.cls_ctx[labels]

    def set_stage(self, stage):
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith("prompt_learner") if stage == 1
                             else not name.startswith("prompt_learner"))

    def forward(self, images):
        features = self.image_encoder(images)
        return [self.classifier(features)] * 2, [features] * 3, features

    def image_state(self):
        return {key: value.detach().clone() for key, value in self.state_dict().items()
                if not key.startswith("prompt_learner")}

    def load_image_state(self, state):
        self.load_state_dict(state, strict=False)


def tiny_setup(monkeypatch):
    rows = [{"vehicle_id": i, "label": i, "image_id": f"{i}-{j}", "camera_id": j}
            for i in range(2) for j in (1, 2)]
    config = experiment.ClipConfig(prompt_epochs=2, image_epochs=3, prompt_batch=2,
                                    identities_per_batch=2, images_per_identity=2)
    protocol = {"inner": {"train": [0, 1], "validation": [2, 3]}}

    class Dataset:
        def __init__(self, rows, *_args, **_kwargs):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            return torch.rand(2) + random.random() + np.random.rand(), row["label"], row["image_id"]

    def evaluate(model, *_args, **_kwargs):
        return {"mAP_at_10": float(torch.sigmoid(model.image_encoder.weight.sum()).detach())}

    def encode(_model, selected, *_args, **_kwargs):
        return {r["image_id"]: np.array([r["label"], .5, 1], np.float32) for r in selected}

    monkeypatch.setattr(experiment, "ClipDataset", Dataset)
    monkeypatch.setattr(experiment, "evaluate_inner", evaluate)
    monkeypatch.setattr(experiment, "encode_rows", encode)
    return rows, config, protocol


def test_prompt_and_image_resume_replay_unfinished_epoch(tmp_path, monkeypatch):
    rows, config, protocol = tiny_setup(monkeypatch)
    torch.manual_seed(9)
    initial = copy.deepcopy(TinyClip().state_dict())

    def run(directory):
        model = TinyClip()
        model.load_state_dict(initial)
        experiment.run_prompt_stage(model, rows, "cpu", config, protocol,
                                    directory / "weights", directory / "results")
        result = experiment.run_image_stage(model, rows, rows, "cpu", config, protocol,
                                            directory / "weights", directory / "results")
        return result

    run(tmp_path / "continuous")
    original = experiment.train_image_epoch

    def fail_after_update(*args):
        result = original(*args)
        if args[-1] == 2:
            raise RuntimeError("Simulated interruption mid-epoch")
        return result

    monkeypatch.setattr(experiment, "train_image_epoch", fail_after_update)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        run(tmp_path / "resumed")
    monkeypatch.setattr(experiment, "train_image_epoch", original)
    result = run(tmp_path / "resumed")
    assert result["completed_epochs"] == 3
    a = torch.load(tmp_path / "continuous/weights/image_last.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed/weights/image_last.pt", weights_only=True)
    assert a["best"] == b["best"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    history_path = tmp_path / "resumed/results/image_history.json"
    history_path.unlink()
    run(tmp_path / "resumed")
    assert len(json.loads(history_path.read_text())) == 3
    with pytest.raises(RuntimeError, match="Resume configuration"):
        experiment._resume(tmp_path / "resumed/weights/image_last.pt", {"changed": True})


def test_checkpoint_selection_retains_epoch_zero_and_requires_prompts(tmp_path, monkeypatch):
    rows, config, protocol = tiny_setup(monkeypatch)
    model = TinyClip()
    with pytest.raises(RuntimeError, match="Complete prompt"):
        experiment.run_image_stage(model, rows, rows, "cpu", config, protocol,
                                    tmp_path / "weights", tmp_path / "results")
    experiment.run_prompt_stage(model, rows, "cpu", config, protocol, tmp_path / "weights", tmp_path / "results")
    initial = model.image_state()
    calls = iter([.8, .5, .6, .7])
    monkeypatch.setattr(experiment, "evaluate_inner", lambda *_a, **_k: {"mAP_at_10": next(calls)})
    result = experiment.run_image_stage(model, rows, rows, "cpu", config, protocol,
                                         tmp_path / "weights", tmp_path / "results")
    assert result["best"]["epoch"] == 0 and not result["beats_own_initialization"]
    best = torch.load(tmp_path / "weights/image_best.pt", weights_only=True)
    for key in initial:
        torch.testing.assert_close(best["model"][key], initial[key], rtol=0, atol=0)


def test_lr_schedule_keeps_two_official_drops_and_small_lr():
    config = experiment.ClipConfig()
    assert config.image_lr == 5e-6
    assert experiment.image_lr_factor(29) == 1
    assert experiment.image_lr_factor(30) == pytest.approx(.1)
    assert experiment.image_lr_factor(50) == pytest.approx(.01)
    assert experiment.prompt_learning_rate(config, 5) == pytest.approx(3.5e-4)
    assert experiment.prompt_learning_rate(config, 60) == pytest.approx(1e-6)
    with pytest.raises(ValueError):
        replace(config, identities_per_batch=1).validate()


def test_prompt_resume_replays_partial_epoch(tmp_path, monkeypatch):
    rows, config, protocol = tiny_setup(monkeypatch)
    torch.manual_seed(33)
    initial = copy.deepcopy(TinyClip().state_dict())

    def fit(name):
        model = TinyClip()
        model.load_state_dict(initial)
        experiment.run_prompt_stage(model, rows, "cpu", config, protocol,
                                    tmp_path / name / "weights", tmp_path / name / "results")
        return model.prompt_learner.cls_ctx.detach()

    continuous = fit("full")
    original = experiment.prompt_loss
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Interrupted prompt epoch")
        return original(*args)

    monkeypatch.setattr(experiment, "prompt_loss", fail)
    with pytest.raises(RuntimeError, match="Interrupted prompt"):
        fit("resumed")
    monkeypatch.setattr(experiment, "prompt_loss", original)
    torch.testing.assert_close(fit("resumed"), continuous, atol=0, rtol=0)


def test_notebooks_are_valid_and_all_python_cells_compile():
    import ast
    import nbformat
    from backend.core import ROOT
    paths = [ROOT / "CLIP-ReID-ViT-B-16/variant_01_vehicle_transfer/train_clip_reid.ipynb",
             ROOT / "OSNet-AIN-x1.0/audit_07_errors_masks/audit_osnet.ipynb"]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=str(path))


def test_public_checkpoint_hash_and_strict_load_when_available():
    from backend.core import ROOT
    from training.clip_reid import load_pretrained
    path = ROOT / "CLIP-ReID-ViT-B-16/variant_01_vehicle_transfer/weights/VeRi_clipreid_ViT-B-16_60.pth"
    if not path.exists():
        pytest.skip("Optional public weights are not downloaded; no network in tests")
    model = load_pretrained(path, classes=7)
    assert model.classifier.out_features == 7
    assert model.prompt_learner.cls_ctx.shape == (7, 4, 512)
    assert not any("cv_embed" in key for key in model.state_dict())
    model.set_stage(1)
    assert [name for name, p in model.named_parameters() if p.requires_grad] == ["prompt_learner.cls_ctx"]
    model.set_stage(2)
    assert not any(p.requires_grad for p in model.text_encoder.parameters())
    assert not model.prompt_learner.cls_ctx.requires_grad
