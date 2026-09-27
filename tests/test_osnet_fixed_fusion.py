"""Synthetic CPU geometry/gradient/export checks; no organizer-data training."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from training import osnet_ablations as blocks
from training.hpo import ExperimentConfig
from training.osnet_fixed_fusion import (FixedFusionModel, initialize_fixed_fusion,
                                         unit_branch)
from training.pipeline import set_seed


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_model(branch, **changes):
    variant = blocks.Ablation(name=f"fixed_{branch}", branch=branch, **changes)
    config = variant.recipe(ExperimentConfig(), 71)
    set_seed(71)
    return initialize_fixed_fusion(3, config, variant, "cpu"), config, variant


def nontrivial_bn(model):
    with torch.no_grad():
        for bn in (model.main_bn, model.aux[1]):
            width = bn.num_features
            bn.running_mean.copy_(torch.linspace(-1., 1., width))
            bn.running_var.copy_(torch.linspace(.1, 2., width))
            bn.weight.copy_(torch.linspace(.2, 3., width))
            bn.bias.copy_(torch.linspace(-.4, .7, width))


def test_branch_none_is_bit_exact_legacy_initialization_loss_and_gradient():
    variant = blocks.Ablation()
    config = variant.recipe(ExperimentConfig(), 71)
    set_seed(71)
    expected = blocks.initialize(3, config, variant, "cpu")
    set_seed(71)
    actual = initialize_fixed_fusion(3, config, variant, "cpu")
    assert type(actual) is blocks.AblationModel
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])
    images = torch.randn(6, 3, 64, 64)
    labels, ids = torch.tensor([0, 0, 1, 1, 2, 2]), torch.arange(6)
    arguments = (images, images.flip(-1), labels, ids, config, variant, blocks.MemoryBank(0), 0)
    before, _ = blocks.losses(expected, *arguments)
    after, _ = blocks.losses(actual, *arguments)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    before["loss"].backward()
    after["loss"].backward()
    for (_, left), (_, right) in zip(expected.named_parameters(), actual.named_parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize("branch,width", [("local", 128), ("color", 32)])
@pytest.mark.parametrize("training", [False, True])
def test_separate_bn_preserves_energy_and_weighted_cosine(branch, width, training):
    model, _, _ = make_model(branch)
    nontrivial_bn(model)
    assert isinstance(model.bnneck, nn.Identity)
    model.train(training)
    images = torch.randn(6, 3, 64, 64)
    with torch.no_grad():
        main, extra = model.branch_embeddings(images)
        embedding = model.embedding(images)
        inferred = blocks.InferenceEncoder(model)(images)
    assert embedding.shape == (6, 512 + width)
    assert torch.all(main.norm(dim=1) > 1e-12) and torch.all(extra.norm(dim=1) > 1e-12)
    torch.testing.assert_close(embedding[:, :512].square().sum(1), torch.full((6,), .8))
    torch.testing.assert_close(embedding[:, 512:].square().sum(1), torch.full((6,), .2))
    torch.testing.assert_close(embedding, inferred)
    expected = .8 * (F.normalize(main, dim=1) @ F.normalize(main, dim=1).T)
    expected += .2 * (F.normalize(extra, dim=1) @ F.normalize(extra, dim=1).T)
    torch.testing.assert_close(embedding @ embedding.T, expected)


@pytest.mark.parametrize("branch", ["local", "color"])
@pytest.mark.parametrize("classifier", ["ce", "am_softmax"])
def test_loss_backward_and_optimizer_include_independent_branches(branch, classifier):
    model, config, variant = make_model(branch, classifier=classifier)
    optimizer = blocks.optimizer_for(model, config)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    head_ids = {id(p) for group in optimizer.param_groups if group["name"] == "head"
                for p in group["params"]}
    assert all(id(p) in head_ids for bn in (model.main_bn, model.aux[1]) for p in bn.parameters())
    images = torch.randn(6, 3, 64, 64)
    values, raw = blocks.losses(model, images, images.flip(-1), torch.tensor([0, 0, 1, 1, 2, 2]),
                                torch.arange(6), config, variant, blocks.MemoryBank(0), 0)
    assert all(torch.isfinite(value) for value in values.values())
    torch.testing.assert_close(raw.norm(dim=1), torch.ones(6))
    values["loss"].backward()
    for parameter in (model.main_bn.weight, model.aux[1].weight, model.aux[0].weight,
                      model.classifier.weight):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    model.eval()
    inputs = images[:1].clone().requires_grad_(True)
    output = model.inference_module()(inputs)
    output[:, 0].sum().backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0


@pytest.mark.parametrize("policy", ["none", "backbone", "all"])
def test_bn_freeze_policy_includes_new_main_bn(policy):
    model, _, _ = make_model("color", freeze_bn=policy)
    model.train()
    assert model.main_bn.training == (policy != "all")
    assert model.aux[1].training == (policy != "all")
    backbone_bn = next(m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d))
    assert backbone_bn.training == (policy == "none")


def test_zero_and_tiny_branches_have_explicit_finite_fallback():
    values = torch.tensor([[0., 0., 0.], [1e-20, -1e-20, 0.], [0., 3., 4.]], requires_grad=True)
    actual = unit_branch(values)
    torch.testing.assert_close(actual, torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., .6, .8]]))
    actual.sum().backward()
    assert torch.isfinite(values.grad).all()
    # The fallback preserves energies but its similarity is not a zero-vector cosine.
    fused = torch.cat([unit_branch(torch.zeros(2, 512)) * .8 ** .5,
                       unit_branch(torch.zeros(2, 32)) * .2 ** .5], dim=1)
    torch.testing.assert_close(fused.norm(dim=1), torch.ones(2))
    assert (fused[0] @ fused[1]).item() == pytest.approx(1.)


def test_invalid_branch_or_disabled_bn_is_not_silently_accepted():
    variant = blocks.Ablation(branch="color")
    with pytest.raises(ValueError, match="independent BN"):
        FixedFusionModel(3, replace(ExperimentConfig(), use_bnneck=False), variant)
    with pytest.raises(ValueError, match="local or color"):
        FixedFusionModel(3, ExperimentConfig(), blocks.Ablation())


@pytest.mark.parametrize("branch,width", [("local", 128), ("color", 32)])
def test_onnx_cpu_parity_for_dynamic_batches(branch, width, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model, _, _ = make_model(branch)
    nontrivial_bn(model)
    encoder = blocks.InferenceEncoder(model).eval()
    path = tmp_path / f"{branch}.onnx"
    torch.onnx.export(encoder, torch.zeros(1, 3, 208, 208), path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for batch in (1, 3, 8):
        images = torch.randn(batch, 3, 208, 208)
        with torch.no_grad():
            expected = encoder(images).numpy()
        actual = session.run(["output"], {"input": images.numpy()})[0]
        assert actual.shape == (batch, 512 + width) and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose((actual[:, :512] ** 2).sum(1), .8, atol=2e-6)
        np.testing.assert_allclose((actual[:, 512:] ** 2).sum(1), .2, atol=2e-6)
