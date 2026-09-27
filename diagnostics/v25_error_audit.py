"""Read-only v25 error decomposition. Label-aware oracles are NOT inference methods."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['ORT_DISABLE_TELEMETRY'] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from backend.core import normalize, read_rows
from training import dual_role_inference as dual, map_inference as v25

RUN = ROOT/'OSNet-AIN-x1.0/variant_29_family_finalists/runs/family_finalists_v1'
KS = (1, 5, 10, 20, 50, 100)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ap10(order, positive, junk):
    # The submission has ONLY ten IDs: never backfill after junk removal.
    selected = order[:10]
    rel = positive[selected[~junk[selected]]]
    return float((np.cumsum(rel)/(np.arange(len(rel))+1)*rel).sum()/min(int(positive.sum()), 10))


def query_stats(order, positive, junk):
    order = np.asarray(order)
    positive, junk = np.asarray(positive, dtype=bool), np.asarray(junk, dtype=bool)
    if not positive.any() or np.any(positive & junk):
        raise ValueError('Need at least one valid positive and disjoint junk')
    first = int(np.flatnonzero(positive[order])[0])+1
    oracle = {str(k): min(int(positive[order[:k]].sum()), 10)/min(int(positive.sum()), 10)
              for k in (10, 20, 50, 100)}
    ap = ap10(order, positive, junk)
    clean = order[~junk[order]]
    short_clean = order[:10][~junk[order[:10]]]
    group = ('perfect' if ap >= 1-1e-12 else 'positive_in_top10' if first <= 10
             else 'first_11_20' if first <= 20 else 'first_21_50' if first <= 50
             else 'first_51_100' if first <= 100 else 'first_above_100')
    return {'ap': ap, 'first_positive_position': first, 'group': group,
            'rank1': bool(positive[short_clean[:1]].any()),
            'rank5': bool(positive[short_clean[:5]].any()),
            'hit': {str(k): bool(positive[order[:k]].any()) for k in KS},
            'oracle': oracle, 'junk_in_top10': int(junk[order[:10]].sum()),
            'ap_if_full_list_were_submitted': ap10(clean, positive, junk),
            'loss_order_inside_10': oracle['10']-ap,
            'loss_missing_from_10_present_50': oracle['50']-oracle['10'],
            'loss_missing_from_50': 1-oracle['50']}


def summarize(entries):
    n = len(entries)
    result = {'known_queries': n, 'mAP@10': float(np.mean([e['ap'] for e in entries])),
              'Rank-1': float(np.mean([e['rank1'] for e in entries])),
              'Rank-5': float(np.mean([e['rank5'] for e in entries])),
              'hit_counts': {str(k): sum(e['hit'][str(k)] for e in entries) for k in KS},
              'oracle_mAP': {str(k): float(np.mean([e['oracle'][str(k)] for e in entries]))
                             for k in (10, 20, 50, 100)},
              'groups': {}}
    for group in ('perfect', 'positive_in_top10', 'first_11_20', 'first_21_50',
                  'first_51_100', 'first_above_100'):
        subset = [e for e in entries if e['group'] == group]
        result['groups'][group] = {'count': len(subset), 'lost_mAP': sum(1-e['ap'] for e in subset)/n}
    for key in ('loss_order_inside_10', 'loss_missing_from_10_present_50', 'loss_missing_from_50'):
        result[key] = float(np.mean([e[key] for e in entries]))
    result['queries_with_junk_in_top10'] = sum(e['junk_in_top10'] > 0 for e in entries)
    result['full_list_junk_backfill_mAP_gap'] = float(np.mean([
        e['ap_if_full_list_were_submitted']-e['ap'] for e in entries]))
    return result


def paired(before, after):
    delta = np.array([b['ap']-a['ap'] for a, b in zip(before, after)])
    return {'mAP_delta': float(delta.mean()), 'better': int((delta > 1e-12).sum()),
            'worse': int((delta < -1e-12).sum()), 'equal': int((abs(delta) <= 1e-12).sum()),
            'rank1_fixed': sum(not a['rank1'] and b['rank1'] for a, b in zip(before, after)),
            'rank1_broken': sum(a['rank1'] and not b['rank1'] for a, b in zip(before, after))}


def audit_split(manifest, split, inputs):
    protocol = manifest['protocols'][split]
    ids = protocol['query_ids']+protocol['gallery_ids']
    by_id = {r['image_id']: r for r in read_rows(ROOT/'dataset/train.csv')}
    q, g = ([by_id[i] for i in protocol[key]] for key in ('query_ids', 'gallery_ids'))
    source22 = Path(manifest['dual_role_plan']['source_directory'])
    blocks = []
    for component in ('mvp', 'full_v18'):
        path = source22/'tasks'/f'features_{split}_{component}'/'features.npz'
        assert sha(path) == inputs[str(path)]
        with np.load(path, allow_pickle=False) as cache:
            assert cache['ids'].tolist() == ids
            blocks.append(cache['vectors'])
    values = dual.pack(*blocks)
    mixed = normalize(np.concatenate([normalize(b)*np.float32(np.sqrt(.5)) for b in blocks], axis=1))
    ranked = v25.rank(values[:len(q)], values[len(q):])
    orders = {name: dual.policy.rank_vectors(v[:len(q)], v[len(q):], 'raw')['order']
              for name, v in zip(('MVP_raw', 'R1_raw', 'fusion_raw'), [*blocks, mixed])}
    orders['v25'] = ranked['order']
    qf, gf = dual.policy.frames(q, g)
    source28 = Path(manifest['finalists_plan']['source_directory'])
    expected = read(source28/'tasks'/f'{split}_V25_control'/'result.json')
    threshold = manifest['dual_role_plan']['profile']['threshold']
    actual = dual.policy.evaluate(q, g, ranked, threshold, 'raw_top1')
    assert actual['ranking'] == expected['ranking'] and actual['candidates'] == expected['candidates']
    if split == 'validation':
        path = RUN/'tasks/validation_V25_control/export'
        assert np.array_equal(values, np.load(path/'embeddings.npy', allow_pickle=False))
        with (path/'submission.csv').open() as stream:
            saved = {r[0]: r[1:] for r in csv.reader(stream)}
        assert dual.policy.predictions(q, g, ranked, threshold, 'raw_top1')[0] == saved
    vids, cams = (np.array([r[key] for r in g]) for key in ('vehicle_id', 'camera_id'))
    records = []
    for i, query in enumerate(q):
        positive = (vids == query['vehicle_id']) & (cams != query['camera_id'])
        junk = (vids == query['vehicle_id']) & (cams == query['camera_id'])
        if not positive.any():
            continue
        systems = {name: query_stats(order[i], positive, junk) for name, order in orders.items()}
        for name, order in orders.items():
            # Cross-check the independent AP calculation against the organizer evaluator.
            ordered = {query['image_id']: [g[int(j)]['image_id'] for j in order[i, :10]]}
            official = dual.policy.official.ranking_metrics(qf.loc[[query['image_id']]], gf, ordered)
            assert systems[name]['ap'] == official['mAP@10']
        union = np.union1d(orders['MVP_raw'][i, :50], orders['R1_raw'][i, :50])
        best_positive = int(next(j for j in orders['v25'][i] if positive[j]))
        records.append({'query_id': query['image_id'], 'vehicle_id': query['vehicle_id'],
                        'valid_positives': int(positive.sum()), 'systems': systems,
                        'query_bbox': {k: query[k] for k in ('x', 'y', 'w', 'h')},
                        'v25_top1_id': g[int(orders['v25'][i, 0])]['image_id'],
                        'v25_first_positive_id': g[best_positive]['image_id'],
                        'component_union50_hit': bool(positive[union].any()),
                        'component_union50_oracle': min(int(positive[union].sum()), 10)/min(int(positive.sum()), 10)})
    summaries = {name: summarize([r['systems'][name] for r in records]) for name in orders}
    for key in ('mAP@10', 'Rank-1', 'Rank-5'):
        assert summaries['v25'][key] == actual['ranking'][key]
    changes = paired([r['systems']['fusion_raw'] for r in records], [r['systems']['v25'] for r in records])
    worst = sorted(records, key=lambda r: r['systems']['v25']['ap'])
    total_loss = sum(1-r['systems']['v25']['ap'] for r in records)
    return {'query_count': len(q), 'gallery_count': len(g), 'known_queries': len(records),
            'unknown_queries_excluded': len(q)-len(records), 'systems': summaries,
            'reranker_vs_same_fusion_raw': changes,
            'component_union_top50': {'hit_count': sum(r['component_union50_hit'] for r in records),
                                      'oracle_mAP': float(np.mean([r['component_union50_oracle'] for r in records])),
                                      'note': 'Union has up to 100 images, not a 50-candidate method'},
            'worst_queries_loss_share': {str(k): sum(1-r['systems']['v25']['ap'] for r in worst[:k])/total_loss
                                         for k in (10, 20, 30)},
            'per_query': records}


def main(output):
    manifest = read(RUN/'manifest.json')
    inputs = dict(manifest['protected'])
    inputs.update({str(ROOT/p): h for p, h in manifest['source_sha256'].items()})
    for p in (RUN/'manifest.json', RUN/'results.json', RUN/'frozen_comparison.json', Path(__file__)):
        inputs[str(p)] = sha(p)
    result = read(RUN/'results.json')
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert result['status'] == 'complete' and result['signature'] == signature
    for receipt in RUN.glob('tasks/*/complete.json'):
        value = read(receipt)
        assert value['signature'] == signature
        inputs[str(receipt)] = sha(receipt)
        inputs.update({str(RUN/p): h for p, h in value['artifacts'].items()})
    def verify():
        for path, expected in inputs.items():
            assert sha(Path(path)) == expected, path
    verify()
    results = {split: audit_split(manifest, split, inputs) for split in ('calibration', 'validation')}
    verify()
    report = {'scope': 'Post-hoc diagnosis on already observed development splits; no tuning or promotion',
              'oracle_warning': 'Uses true identities ONLY for upper bounds; NOT a usable inference score',
              'loss_identity': '(1-AP) = (oracle10-AP) + (oracle50-oracle10) + (1-oracle50)',
              'pool_semantics': 'Prefixes of unfiltered full ranking; junk stripped only inside submitted top10',
              'threshold': manifest['dual_role_plan']['profile']['threshold'], 'splits': results,
              'source_signature': signature, 'verified_sha256': inputs,
              'protected_unchanged': True, 'encoder_forwards': 0, 'training_updates': 0, 'promoted': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for split, data in results.items():
        print(split, json.dumps({k: v for k, v in data.items() if k != 'per_query'}, ensure_ascii=False))
    print('Verified files:', len(inputs), '| Report:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; existing files are never overwritten')
    main(parser.parse_args().output)
