"""Synthetic policy checks: no organizer images, training, or source mutations."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

from backend.core import normalize
from training import retrieval_policy as policy
from training import policy_inference as inference
from training import frozen_inference as frozen
from training import retrieval_policy_experiment as experiment


@pytest.fixture
def protocol():
    query = [{"image_id": f"q{i}", "vehicle_id": i, "camera_id": 0} for i in (0, 1, 99)]
    gallery = [{"image_id": f"g{i}", "vehicle_id": i // 2, "camera_id": 1} for i in range(12)]
    rng = np.random.default_rng(30)
    gv = normalize(rng.normal(size=(12, 8)).astype(np.float32))
    qv = normalize(np.stack([gv[0] + .1, gv[2] + .2, rng.normal(size=8)]).astype(np.float32))
    return query, gallery, qv, gv


def test_equal_concat_matches_mean_cosine_not_coordinate_average():
    rng = np.random.default_rng(14)
    members = [normalize(rng.normal(size=(9, 5)).astype(np.float32)) for _ in range(3)]
    merged = policy.combine_members(members)
    assert merged.shape == (9, 15)
    np.testing.assert_allclose(merged @ merged.T, sum(x @ x.T for x in members) / 3, atol=3e-7)


@pytest.mark.parametrize("count", [0, 2, 4])
def test_ensemble_cannot_pick_subset(count):
    with pytest.raises(ValueError):
        policy.combine_members([np.ones((3, 4), np.float32)] * count)


@pytest.mark.parametrize("setting", list(policy.POLICIES))
def test_streaming_query_independence(protocol, setting):
    _, _, qv, gv = protocol
    whole = policy.rank_vectors(qv, gv, setting)
    for index in range(len(qv)):
        one = policy.rank_vectors(qv[index:index+1], gv, setting)
        np.testing.assert_array_equal(whole["order"][index], one["order"][0])


def test_raw_control_does_not_construct_graph(protocol, monkeypatch):
    _, _, qv, gv = protocol
    monkeypatch.setattr(policy, "KReciprocalReranker", lambda *_: pytest.fail("Raw must bypass graph"))
    r = policy.rank_vectors(qv, gv, "raw")
    np.testing.assert_array_equal(r["order"], np.argsort(-(qv @ gv.T), axis=1, kind="stable"))


def test_candidate_policy_independent_of_top10_and_csv_official_roundtrip(protocol, tmp_path):
    query, gallery, qv, gv = protocol
    ranking = policy.rank_vectors(qv, gv, "raw")
    ranking["order"] = ranking["order"].copy()
    ranking["order"][0, [0, 1]] = ranking["order"][0, [1, 0]]
    before = policy.evaluate(query, gallery, ranking, -2, "ranking_top1")
    after = policy.evaluate(query, gallery, ranking, -2, "raw_top1")
    assert before["ranking"] == after["ranking"]
    exported = policy.export_csv(tmp_path / "export", query, gallery, ranking, -2, "raw_top1")
    assert exported == after
    predictions, accepted = policy.predictions(query, gallery, ranking, -2, "raw_top1")
    assert accepted["q0"][0][0] != predictions["q0"][0]


@pytest.mark.parametrize("candidate", policy.CANDIDATES)
def test_threshold_curve_matches_official_and_does_not_use_outer(protocol, candidate):
    query, gallery, qv, gv = protocol
    ranking = policy.rank_vectors(qv, gv, "legacy")
    with pytest.raises(ValueError, match="calibration-only"):
        policy.calibrate_policy(query, gallery, ranking, candidate, split="validation")
    cal = policy.calibrate_policy(query, gallery, ranking, candidate, split="calibration")
    for point in cal["curve"]:
        actual = policy.evaluate(query, gallery, ranking, point["threshold"], candidate)["candidates"]
        assert actual == {k: v for k, v in point.items() if k != "threshold"}
    assert cal["selected"] == max(cal["curve"], key=lambda x: (x["C"], x["F1"], x["threshold"]))
    assert cal["curve"][-1]["TN"] == 1 and cal["curve"][-1]["FN"] == 2


def test_refusal_keeps_all_top10_rows(protocol, tmp_path):
    query, gallery, qv, gv = protocol
    r = policy.export_csv(tmp_path, query, gallery, policy.rank_vectors(qv, gv, "legacy"), 2, "raw_top1")
    assert r["candidates"]["TN"] == 1
    assert len((tmp_path / "submission.csv").read_text().splitlines()) == 3
    assert len((tmp_path / "candidates.csv").read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="new/empty"):
        policy.export_csv(tmp_path, query, gallery, policy.rank_vectors(qv, gv, "raw"), 2, "raw_top1")


def test_gallery_diagnostics_never_filter_ranking(protocol):
    query, gallery, qv, gv = protocol
    ranked = policy.rank_vectors(qv, gv, "legacy")
    before = ranked["order"].copy()
    d = policy.gallery_diagnostics(gallery, ranked["graph"])
    np.testing.assert_array_equal(before, ranked["order"])
    assert sum(d["neighbor_frequency"].values()) == len(gallery) * (len(gallery) - 1)
    assert d["same_identity_same_camera_junk_edges"] <= d["same_identity_edges"]


def test_diagnostics_keep_all_unknown_scores_and_overlap(protocol):
    query, gallery, qv, gv = protocol
    d = policy.query_diagnostics(query, gallery, policy.rank_vectors(qv, gv, "raw"))
    assert d["per_query"]["q99"]["raw"]["ap"] is None
    assert d["confidence_before_threshold"]["unknown"]["count"] == 1
    overlap = policy.error_overlap({"a": d, "b": d})["a/b"]
    assert overlap["intersection"] == overlap["union"]


def tiny_model(path, offset=0.):
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1, 3], [offset, 0., 0.])
    graph = helper.make_graph([
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[2, 3], keepdims=0),
        helper.make_node("Add", ["mean", "bias"], ["embedding"]),
    ], "tiny", [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 8, 8])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 3])], [bias])
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), path)


@pytest.fixture
def bundle(tmp_path):
    paths = []
    for i in range(3):
        model, path = tmp_path / f"model{i}.onnx", tmp_path / f"source{i}.json"
        tiny_model(model, i * .1)
        frozen.write_bundle(path, model, image_size=8, resize_mode="square", threshold=.5,
            calibration={"split": "calibration", "protocol_sha256": "a" * 64, "method": "synthetic"})
        paths.append(path)
    output = tmp_path / "policy.json"
    inference.write_bundle(output, paths, ranking="less_graph", candidate_policy="raw_top1", threshold=.4,
        calibration={"split": "calibration", "candidate_policy": "raw_top1", "protocol_sha256": "b" * 64, "method": "synthetic"})
    return output


def test_bundle2_leaves_source_immutable_and_equal_three_features(bundle):
    before = {p: p.read_bytes() for p in bundle.parent.glob('source*.json')}
    encoder = inference.PolicyEncoder(bundle)
    batch = np.random.default_rng(12).normal(size=(3, 3, 8, 8)).astype(np.float32)
    actual = encoder.encode_batch(batch)
    assert actual.shape == (3, 9)
    expected = policy.combine_members([e.encode_batch(batch) for e in encoder.members])
    np.testing.assert_allclose(actual, expected)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    old = json.loads(bundle.read_text())
    with pytest.raises(ValueError, match="replace"):
        inference.write_bundle(bundle, list(before), ranking="raw", candidate_policy="raw_top1", threshold=.4,
                                calibration=old["calibration"])


@pytest.mark.parametrize("change", [
    lambda b: b.update(schema=1), lambda b: b.update(threshold=float('nan')),
    lambda b: b.update(candidate_policy="guess"), lambda b: b['calibration'].update(split="validation"),
    lambda b: b['calibration'].update(candidate_policy="ranking_top1"),
    lambda b: b['ranking_parameters'].update(k1=99),
    lambda b: b['members'][0].update(sha256="0" * 64),
    lambda b: b['members'].__setitem__(1, b['members'][0]),
])
def test_bundle2_rejects_changed_contract(bundle, change):
    value = json.loads(bundle.read_text()); change(value); bundle.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        inference.PolicyEncoder(bundle)


def test_parity_checks_decisions_not_just_tensors(protocol):
    query, gallery, qv, gv = protocol
    same = inference.compare_decisions(query, gallery, (qv, gv), (qv, gv), "raw", .5, "raw_top1")
    assert same["passed"]
    changed = qv.copy(); changed[0] = gv[5]
    diff = inference.compare_decisions(query, gallery, (qv, gv), (changed, gv), "raw", .5, "raw_top1", atol=10.)
    assert not diff["passed"] and diff["changed_top10"]


def test_bundle_export_never_reads_train_or_labels(bundle, tmp_path, monkeypatch):
    data = tmp_path / "data"; (data / "images").mkdir(parents=True)
    for split, count in (("query", 3), ("gallery", 12)):
        with (data / f"test_{split}.csv").open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['image_id', 'x', 'y', 'w', 'h'])
            for i in range(count):
                image_id = f'{split}{i}'
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (10, 10, 3), dtype=np.uint8)).save(data / 'images' / f'{image_id}.jpg')
                writer.writerow([image_id, 0, 0, 10, 10])
    monkeypatch.setattr(policy, 'calibrate_policy', lambda *_a, **_k: pytest.fail('No calibration in export'))
    output = tmp_path / 'output'
    inference.export_frozen(bundle, data, output)
    assert np.load(output / 'embeddings.npy').shape == (15, 9)
    manifest = json.loads((output / 'export_manifest.json').read_text())
    assert manifest['encoder_forward_count'] == 3 and not manifest['promoted']


def test_cached_reports_and_vectors_protect_content(tmp_path):
    context = {'output': tmp_path, 'signature': 'sig'}
    assert experiment.cached_report(context, 'r.json', lambda: {'x': 1}) == {'x': 1}
    assert experiment.cached_report(context, 'r.json', lambda: pytest.fail('Recomputed')) == {'x': 1}
    x = json.loads((tmp_path / 'r.json').read_text()); x['result']['x'] = 2
    (tmp_path / 'r.json').write_text(json.dumps(x))
    with pytest.raises(ValueError, match='Changed cached report'):
        experiment.cached_report(context, 'r.json', lambda: None)
    rows = [{'image_id': 'a'}, {'image_id': 'b'}]
    first = experiment.cached_vectors(context, 'v', rows, lambda: np.ones((2, 3)))
    second = experiment.cached_vectors(context, 'v', rows, lambda: pytest.fail('Recomputed'))
    np.testing.assert_array_equal(first['a'], second['a'])
    (tmp_path / 'cache/v.npz').write_bytes(b'broken')
    with pytest.raises(ValueError, match='Changed embedding cache'):
        experiment.cached_vectors(context, 'v', rows, lambda: None)


def test_equal_draw_then_seed_average_no_best_seed():
    matrix = {str(seed): {f'regular_{d}': {'legacy': {'ranking': {'mAP@10': seed / 10 + d / 100}}}
                          for d in range(3)} for seed in (1, 2, 3)}
    result = experiment.summarize_matrix(matrix, 'legacy')
    assert result['mean'] == pytest.approx(.21)
    assert result['std'] == pytest.approx(.1)
    ensemble = experiment.summarize_matrix({'equal3': matrix['1']}, 'legacy')
    assert ensemble['std'] is None and 'ensemble' in ensemble['unit']


def test_selection_tie_keeps_legacy():
    matrix = {str(s): {'regular_1': {p: {'ranking': {'mAP@10': .8}} for p in policy.POLICIES}} for s in (1, 2, 3)}
    result = experiment.select_policies({name: matrix for name in experiment.SYSTEMS})
    assert all(x['policy'] == 'legacy' and x['step'] == 800 for x in result.values())


def test_csv_resume_after_export_before_report(protocol, tmp_path):
    query, gallery, qv, gv = protocol
    ranked = policy.rank_vectors(qv, gv, 'raw')
    output = tmp_path / 'csv'
    first = experiment.verify_or_export_csv(output, query, gallery, ranked, .5, 'raw_top1')
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert experiment.verify_or_export_csv(output, query, gallery, ranked, .5, 'raw_top1') == first
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    (output / 'submission.csv').write_text('bad,bad\n')
    with pytest.raises(ValueError, match='differs'):
        experiment.verify_or_export_csv(output, query, gallery, ranked, .5, 'raw_top1')


def test_inner_freezes_before_alternate_and_never_calls_final(protocol, tmp_path, monkeypatch):
    query, gallery, qv, gv = protocol
    rows = query + gallery
    encoded = dict(zip([r['image_id'] for r in rows], np.concatenate([qv, gv])))
    draw = {'query_ids': [r['image_id'] for r in query], 'gallery_ids': [r['image_id'] for r in gallery], 'selection_eligible': True}
    context = {'output': tmp_path, 'signature': 's', 'rows': rows, 'seeds': (1, 2, 3),
               'manifest': {'draws': {fold: {'regular_1': draw} for fold in ('primary', 'alternate')}},
               'confirmation': {'selection': {}, 'alternate_summary': {}}}
    def encode(ctx, fold, *_args):
        if fold == 'alternate':
            assert (tmp_path / 'selection.json').exists()
        return encoded
    monkeypatch.setattr(experiment, 'inner_vectors', encode)
    monkeypatch.setattr(experiment, 'final_vectors', lambda *_a: pytest.fail('No outer'))
    monkeypatch.setattr(experiment.review, 'fit', lambda *_a: pytest.fail('No training'))
    result = experiment.run_inner(context)
    assert not result['outer_evaluated'] and result['training_updates'] == 0
    assert set(result['summary']) == {'primary', 'alternate'}
    assert 'R1_equal3' in experiment.report_text('inner', result)


def test_final_requires_explicit_outer_permission():
    with pytest.raises(ValueError, match='ALLOW_OUTER'):
        experiment.run({}, 'final')


def test_final_freezes_all_calibration_before_outer_and_official_csvs(protocol, tmp_path, monkeypatch):
    query, gallery, qv, gv = protocol
    encoded = dict(zip([r['image_id'] for r in query + gallery], np.concatenate([qv, gv])))
    selection = {name: {'policy': 'legacy' if name == 'B0_control' else 'less_graph'} for name in experiment.SYSTEMS}
    experiment.write_json(tmp_path / 'selection.json', selection)
    experiment.write_json(tmp_path / 'inner_complete.json', {'signature': 'sig', 'artifacts': {
        'selection.json': experiment.sha256(tmp_path / 'selection.json')}})
    source_paths = []
    for i in range(7):
        path = tmp_path / f'source{i}.json'; path.write_text(json.dumps({'source': i})); source_paths.append(path)
    systems, paths = {}, {}
    for i, (name, seed) in enumerate((n, s) for n in experiment.seeds.NAMES for s in (1, 2, 3)):
        key = f'{name}_{seed}'; systems[key] = encoded; paths[key] = [source_paths[i]]
    systems['R1_equal3'] = encoded; paths['R1_equal3'] = source_paths[3:6]
    context = {'output': tmp_path, 'signature': 'sig', 'source': {},
               'manifest': {'protocols': {'calibration': {'synthetic': True}}}}
    order = []
    def final_systems(ctx, split):
        order.append(split)
        if split == 'validation':
            saved = json.loads((tmp_path / 'final_selection.json').read_text())
            assert len(saved) == 22
            assert all((tmp_path / 'final' / s['case'] / 'bundle.json').exists() for s in saved)
        return systems, paths
    monkeypatch.setattr(experiment, 'final_systems', final_systems)
    monkeypatch.setattr(experiment.review.old, 'protocol_rows', lambda *_: (query, gallery))
    monkeypatch.setattr(experiment, 'ARTIFACTS', tmp_path)
    experiment.write_json(tmp_path / 'baseline_metrics.json', {})
    monkeypatch.setattr(experiment.review, 'fit', lambda *_: pytest.fail('No training'))
    result = experiment.run_final(context)
    assert order == ['calibration', 'validation'] and result['outer_evaluated']
    assert len(result['cases']) == 22 and not result['promoted']
    for key in systems:
        cases = [x for x in result['cases'] if x['key'] == key and x['ranking_policy'] == 'legacy']
        assert cases[0]['ranking'] == cases[1]['ranking']
        assert {x['candidate'] for x in cases} == set(policy.CANDIDATES)
    text = experiment.report_text('final', result)
    assert 'C = 0.7 F1' in text and 'R1_equal3' in text
    assert experiment.run_final(context) == result


def test_phase_resume_and_receipt_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'VARIANT', tmp_path / 'variant')
    monkeypatch.setattr(experiment.seeds, 'runtime', lambda *_: {'fake': True})
    checks = []
    monkeypatch.setattr(experiment.review, 'check_inputs', lambda *_a, **_k: checks.append('verified'))
    monkeypatch.setattr(experiment.review, 'check_other_runs', lambda *_: None)
    monkeypatch.setattr(experiment, 'run_inner', lambda _: {'outer_evaluated': False, 'training_updates': 0})
    monkeypatch.setattr(experiment, 'report_text', lambda *_: 'synthetic')
    context = {'output': tmp_path / 'variant/runs/test', 'signature': 'sig',
               'device': 'cpu', 'manifest': {'runtime': {'fake': True}}}
    first = experiment.run(context)
    monkeypatch.setattr(experiment, 'run_inner', lambda _: pytest.fail('Already completed'))
    assert experiment.run(context) == first
    (context['output'] / 'inner.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        experiment.run(context)
    assert len(checks) == 4


def test_checkpoint_diagnostic_never_replaces_main_selection(tmp_path, monkeypatch):
    context = {'output': tmp_path, 'seeds': (1, 2, 3), 'confirmation': {'primary': [
        {'variant': n, 'seed': s, 'checkpoints': {'200': {}, '800': {}}}
        for n in experiment.seeds.NAMES for s in (1, 2, 3)]}}
    selection = {'untouched': True}
    experiment.write_json(tmp_path / 'selection.json', selection)
    monkeypatch.setattr(experiment, 'inner_vectors', lambda *_: {})
    def evaluate(ctx, fold, name, seed, encoded, settings, step):
        assert fold == 'primary'
        return {'regular_1': {p: {'ranking': {'mAP@10': .9 if step == 200 else .8}} for p in settings}}
    monkeypatch.setattr(experiment, 'evaluate_inner', evaluate)
    result = experiment.run_checkpoints(context)
    assert all(v == ('200', 'legacy') for v in result['primary_proposals_only'].values())
    assert not result['changes_final_recipe']
    assert json.loads((tmp_path / 'selection.json').read_text()) == selection
