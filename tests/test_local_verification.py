import numpy as np
import pytest
import torch
from torch import nn

from training.local_verification import (
    extract_local_tokens, local_pair_scores, mix_topk_scores,
    rerank_topk, topk_oracle_diagnostics,
)


class TinySpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, batch):
        return self.global_pool(self.conv5(batch)).flatten(1)


def test_extraction_frozen_eval_and_batch_independent():
    torch.manual_seed(17)
    model = TinySpatialEncoder().eval()
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    inputs = torch.randn(3, 3, 8, 8)
    features = extract_local_tokens(model, inputs)
    separate = np.concatenate([extract_local_tokens(model, x[None]) for x in inputs])
    permuted = extract_local_tokens(model, inputs[[2, 0, 1]])
    assert features.shape == (3, 16, 4)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features, separate, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(permuted, features[[2, 0, 1]], rtol=1e-6, atol=1e-6)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert not model.conv5._forward_hooks
    model.train()
    with pytest.raises(ValueError, match="eval-mode"):
        extract_local_tokens(model, inputs)


def test_extraction_cleans_hook_after_exception_and_rejects_invalid_inputs():
    model = TinySpatialEncoder().eval()
    with pytest.raises(ValueError, match="grid"):
        extract_local_tokens(model, torch.ones(1, 3, 2, 2))
    assert not model.conv5._forward_hooks
    with pytest.raises(ValueError, match="finite"):
        extract_local_tokens(model, torch.full((1, 3, 8, 8), float("nan")))
    model.conv5[1].train()
    with pytest.raises(ValueError, match="eval-mode"):
        extract_local_tokens(model, torch.ones(1, 3, 8, 8))


@pytest.mark.parametrize("scorer", ["mean_best", "partial"])
def test_local_similarity_symmetric_bounded_and_zero_safe(scorer):
    q = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
    g = np.array([[1, 1], [1, 0]], dtype=np.float32)
    score = local_pair_scores(q, g[None], scorer)[0]
    reverse = local_pair_scores(g, q[None], scorer)[0]
    assert 0 <= score <= 1
    assert score == pytest.approx(reverse)
    assert local_pair_scores(q, np.zeros((1, 2, 2)), scorer)[0] == 0
    assert local_pair_scores(np.zeros_like(q), g[None], scorer)[0] == 0
    assert local_pair_scores(np.ones((2, 2)), -np.ones((1, 2, 2)), scorer)[0] == 0
    assert local_pair_scores(q, q[None], scorer)[0] == pytest.approx(1)


def test_partial_support_distinct_from_mean_best():
    query = np.eye(2)
    gallery = np.array([[[1, 0], [1, 0]]])
    assert local_pair_scores(query, gallery, "mean_best")[0] == pytest.approx(.75)
    assert local_pair_scores(query, gallery, "partial")[0] == pytest.approx(1)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_features_fail_closed(bad):
    with pytest.raises(ValueError, match="finite"):
        local_pair_scores(np.array([[bad, 1]]), np.ones((1, 1, 2)))
    with pytest.raises(ValueError, match="finite"):
        local_pair_scores(np.ones((1, 2)), np.array([[[bad, 1]]]))


def test_large_finite_features_and_bad_shapes():
    assert local_pair_scores(np.full((2, 2), 1e30), np.full((1, 2, 2), 1e30))[0] == pytest.approx(1)
    with pytest.raises(ValueError, match="TxD"):
        local_pair_scores(np.ones((2, 3)), np.ones((1, 2, 4)))
    with pytest.raises(ValueError, match="Unknown"):
        local_pair_scores(np.ones((2, 2)), np.ones((1, 2, 2)), "learned")
    with pytest.raises(ValueError, match="nonempty"):
        local_pair_scores(np.ones((0, 2)), np.ones((1, 2, 2)))


def test_mixing_preserves_pool_tail_and_bounded_change():
    scores = np.array([[.8, .79, .78, .77, .76]])
    order = np.array([[0, 1, 2, 3, 4]])
    local = np.array([[0, 1, .5, 1, 1]])
    result = mix_topk_scores(scores, order, local, top_k=3, weight=.1)
    assert result["order"].tolist() == [[1, 2, 0, 3, 4]]
    np.testing.assert_array_equal(result["order"][:, 3:], order[:, 3:])
    np.testing.assert_array_equal(result["scores"][:, 3:], scores[:, 3:])
    changes = result["scores"] - scores
    assert changes.min() >= 0
    assert changes.max() <= .1 + 1e-12
    zero = mix_topk_scores(scores, order, local, top_k=50, weight=0)
    np.testing.assert_array_equal(zero["order"], order)
    np.testing.assert_array_equal(zero["scores"], scores)


def test_mixing_ties_keep_existing_baseline_order():
    scores = np.zeros((1, 4))
    order = np.array([[2, 0, 3, 1]])
    result = mix_topk_scores(scores, order, np.ones((1, 4)), top_k=3, weight=.1)
    np.testing.assert_array_equal(result["order"], order)


@pytest.mark.parametrize("scorer", ["mean_best", "partial"])
def test_query_permutation_and_single_query_results_match(scorer):
    rng = np.random.default_rng(123)
    q, g = rng.normal(size=(3, 5, 8)), rng.normal(size=(7, 4, 8))
    scores = rng.normal(size=(3, 7))
    order = np.argsort(-scores, axis=1, kind="stable")
    result = rerank_topk(scores, order, q, g, top_k=4, weight=.2, scorer=scorer)
    perm = [2, 0, 1]
    other = rerank_topk(scores[perm], order[perm], q[perm], g, top_k=4, weight=.2, scorer=scorer)
    for key in ("order", "scores", "local_scores"):
        np.testing.assert_array_equal(other[key], result[key][perm])
    for i in range(3):
        single = rerank_topk(scores[i:i+1], order[i:i+1], q[i:i+1], g, top_k=4, weight=.2, scorer=scorer)
        np.testing.assert_array_equal(single["order"][0], result["order"][i])
        np.testing.assert_array_equal(single["scores"][0], result["scores"][i])
    mixed = mix_topk_scores(scores, order, result["local_scores"], top_k=2, weight=.1)
    direct = rerank_topk(scores, order, q, g, top_k=2, weight=.1, scorer=scorer)
    np.testing.assert_array_equal(mixed["order"], direct["order"])
    np.testing.assert_array_equal(mixed["scores"], direct["scores"])


@pytest.mark.parametrize("weight", [-.1, 1.1, float("nan")])
def test_mixing_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match="mixing"):
        mix_topk_scores(np.zeros((1, 2)), np.array([[0, 1]]), np.ones((1, 2)), top_k=2, weight=weight)


def test_mixing_rejects_bad_order_and_bad_score_alignment():
    scores = np.array([[2., 1.]])
    with pytest.raises(ValueError, match="permutation"):
        mix_topk_scores(scores, np.array([[0, 0]]), np.ones((1, 2)), top_k=2, weight=.1)
    with pytest.raises(ValueError, match="descending"):
        mix_topk_scores(scores, np.array([[1, 0]]), np.ones((1, 2)), top_k=2, weight=.1)
    with pytest.raises(ValueError, match="local scores"):
        mix_topk_scores(scores, np.array([[0, 1]]), np.ones((1, 1)), top_k=2, weight=.1)
    with pytest.raises(ValueError, match="local scores"):
        mix_topk_scores(scores, np.array([[0, 1]]), np.array([[1, np.nan]]), top_k=2, weight=.1)


def row(identifier, vehicle, camera):
    return {"image_id": identifier, "vehicle_id": vehicle, "camera_id": camera}


def test_oracle_denominator_all_gallery_positive_and_unknown_exclusion():
    queries = [row("q", 1, 0), row("unknown", 9, 0)]
    gallery = [row("junk", 1, 0), row("positive1", 1, 1), row("same_cam_negative", 2, 0)]
    gallery += [row(f"positive{i}", 1, i) for i in range(2, 6)]
    order = np.tile(np.arange(7), (2, 1))
    result = topk_oracle_diagnostics(queries, gallery, order, ks=(1, 3, 20))
    entry = result["per_query"]["q"]
    assert entry["valid_gallery_positives"] == 5
    assert entry["pools"]["1"]["AP10_oracle"] == 0
    assert entry["pools"]["3"]["valid_positives"] == 1
    assert entry["pools"]["3"]["AP10_oracle"] == .2
    assert entry["pools"]["20"]["AP10_oracle"] == 1
    assert entry["baseline"]["AP10"] < 1  # Same-camera different-ID is not junk.
    assert result["summary"]["known_queries"] == 1
    assert result["summary"]["unknown_queries"] == 1
    assert result["per_query"]["unknown"]["pools"]["3"]["AP10_oracle"] is None
    assert result["summary"]["pools"]["3"]["mean_AP10_oracle"] == .2


def test_physical_ten_before_official_junk_and_top1_transitions():
    queries = [row("q", 1, 0)]
    gallery = [row("junk", 1, 0)] + [row(f"n{i}", 2, 0) for i in range(9)] + [row("p", 1, 1)]
    base = np.arange(11)[None]
    changed = np.array([[10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    report = topk_oracle_diagnostics(queries, gallery, base, ks=(10, 11), comparison_order=changed)
    result = report["per_query"]["q"]
    assert result["baseline"]["AP10"] == 0  # Position 11 is never exported.
    assert result["pools"]["10"]["AP10_oracle"] == 0
    assert result["pools"]["11"]["AP10_oracle"] == 1
    assert result["baseline"]["candidate_top1_correct"] is True  # Candidate evaluator accepts same ID.
    assert result["baseline"]["ranking_top1_correct"] is False
    assert result["comparison"]["AP10"] == 1
    assert report["top1_transitions"]["ranking_wrong_to_correct"] == 1
    assert report["top1_transitions"]["candidate_wrong_to_correct"] == 0


def test_oracle_caps_at_ten_and_all_unknown_is_none():
    gallery = [row(f"p{i}", 1, 1) for i in range(12)]
    report = topk_oracle_diagnostics([row("q", 1, 0)], gallery, np.arange(12)[None], ks=(5, 12))
    assert report["summary"]["pools"]["5"]["mean_AP10_oracle"] == .5
    assert report["summary"]["pools"]["12"]["mean_AP10_oracle"] == 1
    unknown = topk_oracle_diagnostics([row("q", 1, 1)], gallery, np.arange(12)[None], ks=(12,))
    assert unknown["summary"]["known_queries"] == 0
    assert unknown["summary"]["baseline_mAP10"] is None
    assert unknown["summary"]["pools"]["12"]["mean_AP10_oracle"] is None
