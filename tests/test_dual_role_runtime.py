"""Promotion contracts: preserved legacy ranking, separate R1 confidence, unit blocks."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend.core import ROOT, normalize
from backend.dual_role import split_unit, ranking_blocks
from backend.runtime import DEFAULT_PROFILE, ExactScorer, Runtime, validate_vectors


def vectors(count, dim, seed):
    return normalize(np.random.default_rng(seed).normal(size=(count, dim)).astype(np.float32))


def test_promoted_default_and_metrics_have_v25_provenance():
    runtime = Runtime()
    assert DEFAULT_PROFILE == runtime.profile == "MVP_fusion_v25"
    assert runtime.dimension == 2048 and runtime.encoder.size == [208, 256]
    assert runtime.threshold == .534365177154541
    assert runtime.report["validation"]["mAP_at_10"] == .8289573904235558
    assert runtime.metadata()["ranking_r1_weight"] == .5
    assert runtime.metadata()["l2_normalized"] is False
    assert len(runtime.encoder.members) == 4
    decision = json.loads((ROOT/"release_decision.json").read_text())
    assert decision["active_profile"] == DEFAULT_PROFILE and decision["rollback_profile"] == "MVP_dual_role_v24"
    assert decision["promoted"] and not decision["official_submission_ready"]


def test_v24_is_preserved_and_v25_candidates_and_vectors_are_identical():
    old, new = Runtime("MVP_dual_role_v24"), Runtime()
    assert old.report["validation"]["mAP_at_10"] == .8146886982413298
    assert old.encoder.fingerprint == new.encoder.fingerprint and old.threshold == new.threshold
    assert old.fingerprint != new.fingerprint
    values = np.concatenate([vectors(30, 512, 61), vectors(30, 1536, 62)], axis=1)
    rows = [dict(image_id=str(i), x=0, y=0, w=1, h=1) for i in range(25)]
    scorers = [ExactScorer(r, rows, values[5:]) for r in (old, new)]
    from backend.rerank import KReciprocalReranker
    mixed, candidate = ranking_blocks(values, .5)
    expected = KReciprocalReranker(mixed[5:], 20, 3)
    for i, vector in enumerate(values[:5]):
        a, b = [s.decide(vector, threshold=-1) for s in scorers]
        assert a['confidence'] == b['confidence']
        assert a['accepted_candidate']['image_id'] == b['accepted_candidate']['image_id']
        order = np.argsort(expected.distances(mixed[i], .5), kind='stable')[:10]
        assert [x['image_id'] for x in b['results']] == [rows[j]['image_id'] for j in order]
    _, original_r1 = split_unit(values)
    np.testing.assert_array_equal(candidate, original_r1)
    with pytest.raises(ValueError, match='frozen v25'):
        ranking_blocks(values, .75)


def test_two_units_required_and_global_normalization_cannot_corrupt_cache():
    values = np.concatenate([vectors(12, 512, 1), vectors(12, 1536, 2)], axis=1)
    validate_vectors(values, 12, 2048)
    with pytest.raises(ValueError, match="two unit blocks"):
        validate_vectors(normalize(values), 12, 2048)
    with pytest.raises(ValueError):
        validate_vectors(values.astype(np.float64), 12, 2048)


def test_mvp_ranking_r1_candidate_confidence_and_refusal_are_separate():
    runtime = SimpleNamespace(dimension=2048, threshold=.5,
                              spec={"kind": "dual_role", "ranking": "legacy", "candidate_policy": "raw_top1"})
    mvp, r1 = vectors(12, 512, 1), vectors(12, 1536, 2)
    rows = [dict(image_id=str(i), x=0, y=0, w=1, h=1) for i in range(12)]
    gallery = np.concatenate([mvp, r1], axis=1)
    scorer = ExactScorer(runtime, rows, gallery)
    query = np.concatenate([mvp[0], r1[11]])
    scorer.graph = SimpleNamespace(distances=lambda vector, lam: np.arange(12))
    result = scorer.decide(query)
    assert result["results"][0]["image_id"] == "0"
    assert result["accepted_candidate"]["image_id"] == "11"
    assert result["accepted_candidate"]["rank"] == 12
    assert result["accepted_candidate"]["similarity"] == result["confidence"] == pytest.approx(1, abs=1e-6)
    refused = scorer.decide(np.concatenate([mvp[0], -r1[11]]))
    assert refused["accepted_candidate"] is None and refused["refused"]
    assert [x["image_id"] for x in refused["results"]] == [x["image_id"] for x in result["results"]]
    a, b = split_unit(gallery)
    np.testing.assert_allclose(a, mvp, rtol=0, atol=1e-7)
    np.testing.assert_allclose(b, r1, rtol=0, atol=1e-7)
