import numpy as np
import pytest

from diagnostics.v25_error_audit import paired, query_stats, summarize


def test_oracle_counts_all_valid_positives_not_just_one_hit():
    positive = np.zeros(60, bool)
    positive[[4, 19, 49, 59]] = True
    s = query_stats(np.arange(60), positive, np.zeros(60, bool))
    assert s['ap'] == .05
    assert s['oracle'] == {'10': .25, '20': .5, '50': .75, '100': 1.}
    assert s['group'] == 'positive_in_top10'
    assert (s['loss_order_inside_10']+s['loss_missing_from_10_present_50']+s['loss_missing_from_50']) == 1-s['ap']


def test_junk_does_not_backfill_an_eleventh_image_into_submission():
    positive = np.zeros(20, bool); positive[10] = True
    junk = np.zeros(20, bool); junk[0] = True
    s = query_stats(np.arange(20), positive, junk)
    assert s['ap'] == 0 and s['oracle']['10'] == 0
    assert s['ap_if_full_list_were_submitted'] == .1
    assert s['group'] == 'first_11_20'


def test_more_than_ten_positives_have_capped_oracle_denominator():
    s = query_stats(np.arange(30), np.arange(30) < 15, np.zeros(30, bool))
    assert s['ap'] == 1 and all(v == 1 for v in s['oracle'].values())
    assert s['group'] == 'perfect'


def test_unknown_and_overlapping_junk_are_rejected():
    for positive, junk in (([False, False], [False, False]), ([True, False], [True, False])):
        with pytest.raises(ValueError): query_stats(np.arange(2), positive, junk)


def test_group_losses_and_paired_changes_use_known_query_denominator():
    positive = np.zeros(60, bool); positive[0] = True
    junk = np.zeros(60, bool)
    good = query_stats(np.arange(60), positive, junk)
    bad = query_stats(np.roll(np.arange(60), -1), positive, junk)
    r = summarize([good, bad])
    assert r['mAP@10'] == .5 and r['groups']['first_51_100']['lost_mAP'] == .5
    p = paired([good, bad], [good, good])
    assert p == {'mAP_delta': .5, 'better': 1, 'worse': 0, 'equal': 1, 'rank1_fixed': 1, 'rank1_broken': 0}
