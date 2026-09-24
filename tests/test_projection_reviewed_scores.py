import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from report_projection_reviewed_scores import add_note, identity, validate_packet


def test_exact_text_reuse_and_conflicting_verdicts():
    known={};aid=identity('question','7','answer 7')
    add_note(known,aid,True,'Correct','first')
    add_note(known,aid,True,'Correct','second')
    assert known[aid]['provenance']==['first','second']
    assert aid != identity('other question','7','answer 7')
    assert aid != identity('question','8','answer 7')
    assert aid != identity('question','7','answer 7 ')
    with pytest.raises(ValueError): add_note(known,aid,False,'Wrong','third')
    with pytest.raises(ValueError): add_note(known,aid,1,'Not boolean','third')


def test_missing_reading_and_tampered_full_text_rejected():
    aid=identity('question','7','answer 7')
    packet={'answers':[dict(answer_id=aid,prompt='question',reference='7',response='answer 7')]}
    notes=dict(complete=True,packet_sha256='frozen',independent_human_review=False,
               notes={aid:dict(full_answer_read=True,packet_index=0,final_answer_correct=True,note='Correct')})
    assert aid in validate_packet(packet,notes,'frozen')
    with pytest.raises(AssertionError):validate_packet(packet,notes,'different')
    packet['answers'][0]['response']='answer 8'
    with pytest.raises(AssertionError):validate_packet(packet,notes,'frozen')
    packet['answers'][0]['response']='answer 7';notes['notes']={}
    with pytest.raises(AssertionError):validate_packet(packet,notes,'frozen')
