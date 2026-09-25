"""Scientific contracts: endpoint, dev separation, paired unit, hook placement."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sink_direction_common import boxed_answer, select_alpha, paired_binary, GeneratedTokenShift


def test_boxed_endpoint_not_last_number_fallback():
    assert boxed_answer(r'Therefore 17, but no final answer.') is None
    assert boxed_answer(r'\boxed{}') is None
    assert boxed_answer(r'\boxed{\frac{1}{3}}') == r'\frac{1}{3}'
    assert boxed_answer(r'\boxed{\frac{1}{3}') is None
    assert boxed_answer(r'\boxed{2} then \boxed{3}') == '3'
    assert boxed_answer(r'\boxed{\{1,2\}}') == r'\{1,2\}'


def test_dev_selection_and_tie_do_not_use_correctness():
    rows=[{'split':'dev','sample_id':'a','condition':c,'complete_boxed':True,'correct':c=='hss_1'}
          for c in ['zero','hss_0.1','hss_0.3','hss_1']]
    assert select_alpha(rows,[.1,.3,1.])['alpha']==.1
    with pytest.raises(ValueError):select_alpha([dict(r,split='sink_test') for r in rows],[.1,.3,1.])
    with pytest.raises(ValueError):select_alpha(rows[:-1],[.1,.3,1.])


def test_paired_inference_uses_discordant_questions():
    r=paired_binary([1,1,1,1,1,0],[0,0,0,0,0,0],bootstrap=100)
    assert r['wins']==5 and r['losses']==0 and r['one_sided_exact_p']==1/32
    assert r['difference']==5/6


def test_hook_skips_prompt_and_changes_all_forwarded_tokens():
    torch=pytest.importorskip('torch')
    v=torch.tensor([.5,-.25]);patch=GeneratedTokenShift(3,v)
    prompt=torch.ones(1,3,2);original=prompt.clone()
    assert patch(None,(),prompt) is None
    assert torch.equal(prompt,original)
    for _ in range(3):
        token=torch.ones(1,1,2);out=patch(None,(),(token,'other'))
        assert torch.equal(out[0],token+v) and out[1]=='other'
        assert torch.equal(token,torch.ones_like(token))
    assert patch.calls==4 and len(patch.steps)==3
    np.testing.assert_allclose(patch.arrays()['mean_after']-patch.arrays()['mean_before'],v.numpy())
    zero=GeneratedTokenShift(3,torch.zeros(2));zero(None,(),prompt)
    assert zero(None,(),torch.ones(1,1,2)) is None
