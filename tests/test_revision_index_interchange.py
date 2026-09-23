import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from revision_index_interchange_common import (case_pairs,assistant_prefix,ordinary_correct,
                                              score_suffix,transfer,capability_gate)


def test_pair_splits_do_not_reuse_clean_contexts():
    pairs=case_pairs();assert len(pairs)==64
    sides=[(p[s]['symbol'],p[s]['start']) for p in pairs for s in ['recipient','donor']]
    assert len(set(sides))==128
    assert sum(p['split']=='validation' for p in pairs)==16
    assert all(p['recipient']['symbol']!=p['donor']['symbol'] and p['recipient']['start']!=p['donor']['start'] for p in pairs)


def test_target_and_non_target_are_separate_free_output_checks():
    r={'symbol':'a','start':1};d={'symbol':'b','start':4}
    assert ordinary_correct('a_1,a_2, a_3,a_4,a_5,a_6,a_7',r)
    good=score_suffix('8, a_9, a_10',r,d)
    assert good['joint_success'] and not good['ordinary_recipient_correct']
    changed_symbol=score_suffix('8, b_9, b_10',r,d)
    assert changed_symbol['target_counter_sequence'] and not changed_symbol['new_symbols_preserved']
    ordinary=score_suffix('5, a_6, a_7',r,d)
    assert ordinary['new_symbols_preserved'] and not ordinary['target_counter_sequence']
    assert not score_suffix('8, a_9, a_10. explanation',r,d)['format_valid']


def test_projection_transfers_donor_coordinates_and_keeps_recipient_complement():
    x=np.array([[1.,2.,3.],[9.,5.,6.]])
    d=np.array([[7.,8.,9.],[1.,2.,3.]])
    centers=np.array([[0.,0.,0.],[10.,5.,6.]])
    local=np.array([[[1.,0.,0.]],[[0.,1.,0.]]]);shared=np.array([[0.,0.,1.]])
    np.testing.assert_array_equal(transfer(x,d,centers,local,shared,'local8'),[[7,2,3],[9,2,6]])
    np.testing.assert_array_equal(transfer(x,d,centers,local,shared,'shared8'),[[1,2,9],[9,5,3]])
    np.testing.assert_array_equal(transfer(x,d,centers,local,shared,'full_donor'),d)
    np.testing.assert_array_equal(transfer(x,x,centers,local,shared,'local8'),x)


def test_capability_threshold_fails_closed():
    rows=[{'case_id':p['id'],'aligned':True,'recipient':{'free_correct':True,'prefixed_correct':True},
           'donor':{'free_correct':True,'prefixed_correct':True}} for p in case_pairs() if p['split']=='validation']
    assert capability_gate(rows)['passed']
    for r in rows[:4]:r['recipient']['free_correct']=False
    assert not capability_gate(rows)['passed']
    with pytest.raises(AssertionError):capability_gate(rows[:-1])
