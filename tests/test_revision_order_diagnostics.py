from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_revision_token_order import code_identity_diagnostics


def test_position_specific_states_already_encode_order():
    codes=np.tile(np.arange(4),(8,1));tokens=codes+100
    train=np.arange(8)<4;test=~train
    d=code_identity_diagnostics(codes,tokens,train,test,4)
    assert d['position_from_state_accuracy']['estimate']==1
    assert d['position_entropy_fraction_in_state']==1
    assert d['state_position_mutual_information_bits']==2
    assert d['token_from_state_accuracy']['estimate']==1
    assert d['token_accuracy_state_minus_position']['estimate']==0


def test_one_state_does_not_encode_position():
    codes=np.zeros((8,4),dtype=int);tokens=np.tile(np.arange(4),(8,1))
    train=np.arange(8)<4;test=~train
    d=code_identity_diagnostics(codes,tokens,train,test,1)
    assert d['position_from_state_accuracy']['estimate']==.25
    assert d['position_entropy_fraction_in_state']==0
