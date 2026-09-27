"""One-variable correction contracts; real photos are never trained by pytest."""
import copy
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch

from training import nive_mixed_low_aux as low
from test_nive_mixed import context, threads, TinyDataset, evaluation_context, equal

n = low.base


def matched_manifests():
    previous = {"plan": {"experiment": "v32", "aux_weight": .25, "seed": 7, "checkpoints": [0, 2]}}
    for key in ("parent", "variant", "config", "inner", "draws", "nive", "sampling", "runtime", "baseline"):
        previous[key] = {"fixed": key}
    previous["policy"] = {"bn": "same", "gradients": "L_main + .25 L_aux"}
    current = copy.deepcopy(previous)
    current["plan"].update(experiment="v33_nive_low_aux_primary_v1", aux_weight=.025)
    current["policy"]["gradients"] = "L_main + .025 L_aux; two backwards, exactly one optimizer step"
    return previous, current


def test_only_auxiliary_weight_is_changed():
    previous, current = matched_manifests()
    low.assert_matched(previous, current)
    current["plan"]["aux_weight"] = .05
    with pytest.raises(ValueError, match="registered"):
        low.assert_matched(previous, current)


@pytest.mark.parametrize("field", ["parent", "variant", "config", "inner", "draws", "nive", "sampling", "runtime", "baseline", "policy"])
def test_other_scientific_changes_rejected(field):
    previous, current = matched_manifests()
    current[field]["changed"] = True
    with pytest.raises(ValueError, match="changed"):
        low.assert_matched(previous, current)


def test_settings_pin_completed_first_run():
    settings = n.old.load_json(low.VARIANT / "configs/low_aux_v1.json")
    assert settings["aux_weight"] == .025 and settings["corrections_remaining_after_this_run"] == 0
    directory = n.ROOT / settings["previous_run"]
    n.verify_files({str(directory / "manifest.json"): settings["previous_manifest_sha256"],
                    str(directory / "results.json"): settings["previous_results_sha256"]})
    previous = n.old.load_json(directory / "manifest.json")
    # The completed trainer/data code must remain byte-identical, not rewritten in place.
    n.verify_files({str(Path(n.__file__)): previous["source_sha256"][str(Path(n.__file__))],
                    str(Path(n.data.__file__)): previous["source_sha256"][str(Path(n.data.__file__))]})


def test_aux_gradient_tenfold_for_both_arms(context):
    for arm in n.ARMS:
        main = n.load_batch(TinyDataset(context["target"]), context["main_schedule"][0], "cpu", 71)
        rows = context["target"] if arm == n.ARMS[0] else context["external"]
        aux = n.load_batch(TinyDataset(rows), context["aux_schedules"][arm][0], "cpu", 72)
        values = []
        for weight in (.25, .025):
            model = n.new_model(context, arm)
            optimizer = n.optimizer_for(model, context["config"])
            values.append(n.update(model, optimizer, main, aux, context["config"], weight, diagnostics=True))
        assert values[1]["weighted_aux_backbone_grad_norm"] == pytest.approx(values[0]["weighted_aux_backbone_grad_norm"] / 10, rel=2e-6)
        assert values[1]["main_loss"] == values[0]["main_loss"]
        assert values[1]["total_loss"] == pytest.approx(values[1]["main_loss"] + .025*values[1]["aux_loss"])


@pytest.mark.parametrize("arm", n.ARMS)
def test_low_aux_resume_still_exact(context, arm):
    context["plan"]["aux_weight"] = .025
    full = n.train_arm(context, arm)
    expected = torch.load(n.resume_path(context["output"] / "training" / arm), weights_only=True)
    context["output"] = context["output"].with_name("resumed_low")
    n.train_arm(context, arm, stop_after=2)
    actual = n.train_arm(context, arm)
    saved = torch.load(n.resume_path(context["output"] / "training" / arm), weights_only=True)
    for key in ("model", "optimizer", "rng", "history", "initial_bn"):
        equal(expected[key], saved[key])
    assert actual["logical_images"] == full["logical_images"]


def install_fake_encoding(monkeypatch):
    monkeypatch.setattr(n, "check_inputs", lambda c: None)
    def encode(model, rows, *args, **kwargs):
        model.eval()
        x = torch.tensor([[r["vehicle_id"], r["camera_id"], i/10, 1.] for i, r in enumerate(rows)])
        with torch.no_grad():
            vectors = n.normalize(model.embedding(x).numpy())
        return {r["image_id"]: v for r, v in zip(rows, vectors)}
    monkeypatch.setattr(n.old, "encode", encode)
    monkeypatch.setattr(n, "export_encoder", lambda c, arm, ckpt: {"arm": arm, "checkpoint": ckpt})


def test_smoke_uses_new_weight_and_no_full_run(context, monkeypatch):
    install_fake_encoding(monkeypatch)
    context["plan"]["aux_weight"] = .025
    result = low.technical_smoke(context)
    assert result["aux_weight"] == .025 and result["status"] == "passed"
    assert not (context["output"] / "training").exists()
    for arm in n.ARMS:
        assert result["arms"][arm]["training"]["updates"] == 3
        history = n.old.load_json(context["output"] / "technical_smoke/training" / arm / "history.json")
        for i, block in enumerate(history):
            row = block["updates"][0]
            assert row["total_loss"] == pytest.approx(row["main_loss"] + .025*row.get("aux_loss", 0.))
            assert ("aux_loss" in row) == (i < 2)
    assert low.technical_smoke(context) == result


def test_full_run_compares_history_and_resumes(context, monkeypatch):
    install_fake_encoding(monkeypatch)
    e = evaluation_context()
    context["rows"] = e["rows"]; context["manifest"].update(e["manifest"])
    previous = n.run_experiment(context)
    previous_path = context["output"]
    previous_files = {str(p): n.sha256(p) for p in previous_path.rglob("*") if p.is_file()}
    context["output"] = previous_path.with_name("low_aux")
    context["plan"]["aux_weight"] = .025
    context["signature"] = "low_aux_synthetic"
    context["manifest"]["previous_run"] = {"path": str(previous_path)}
    result = low.run_experiment(context)
    assert result["status"] == "complete" and not result["promoted"] and not result["original_validation_evaluated"]
    assert result["reference"] == previous["reference"]
    assert all(set(result["comparison_to_v32"][a]["same_step_delta_map"]) == {"0", "2", "4", "6"} for a in n.ARMS)
    assert result["decision_basis"]["corrections_remaining"] == 0
    text = (context["output"] / "REPORT.md").read_text()
    assert "L_main + 0.025 L_aux" in text and "Сравнение с первым опытом" in text
    assert low.run_experiment(context) == result
    n.verify_files(previous_files)
    (context["output"] / "evaluation/N_ref/metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="Protected file"):
        low.run_experiment(context)


def test_reference_drift_fails_before_training(context, monkeypatch):
    install_fake_encoding(monkeypatch)
    e = evaluation_context()
    context["rows"] = e["rows"]; context["manifest"].update(e["manifest"])
    directory = context["output"] / "previous"
    (directory / "evaluation/N_ref").mkdir(parents=True)
    n.write_json(directory / "evaluation/N_ref/order.json", [r["image_id"] for r in n.development_rows(context)])
    np.save(directory / "evaluation/N_ref/features.npy", np.zeros((len(e["rows"]), 4), np.float32))
    context["manifest"]["previous_run"] = {"path": str(directory)}
    with pytest.raises(ValueError, match="does not reproduce"):
        low.run_experiment(context)
    assert not (context["output"] / "training").exists()


def test_notebook_has_clean_run_all_and_new_entrypoint():
    path = low.VARIANT / "train_nive_low_aux.ipynb"
    nb = nbformat.read(path, as_version=4); nbformat.validate(nb)
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "nive_mixed_low_aux" in text and "RUN_NAME = 'low_aux_v1'" in text
    assert text.index("ORT_DISABLE_TELEMETRY") < text.index("from training")
    assert text.index("pytest") < text.index("experiment.prepare") < text.index("technical_smoke") < text.index("run_experiment")
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
            assert cell.execution_count is None and cell.outputs == []
