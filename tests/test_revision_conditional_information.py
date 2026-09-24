from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from revision_conditional_information import fit_readout,question_text


def test_question_only_lexical_baseline_strips_format_instruction():
    question='What is 1+2?'
    prompt='Question: '+question+'\nPlease reason step by step, and put your final answer within \\boxed{}.'
    assert question_text(prompt)==question
    with pytest.raises(ValueError):
        question_text(question)


def test_readout_selection_does_not_accept_test_outcomes(tmp_path):
    import inspect
    assert 'y_all' not in inspect.signature(fit_readout).parameters
    rng=np.random.default_rng(8);x=rng.normal(size=(120,5));y=(x[:,0]+rng.normal(size=120)*.3>0).astype(int)
    p,selection=fit_readout(x[:60],y[:60],x[60:90],y[60:90],x,tmp_path,[.01,.1,1],500,42)
    assert p.shape==(120,) and selection['converged']
    assert len(list(tmp_path.glob('C_*.npz')))==3
    assert (tmp_path/'selection.json').exists()
