"""Scientific contracts for the reviewed sink follow-up."""
import ast
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import hss_followup_tools as t
from hss_followup_common import sentence_boundaries, boxed_token_indices, gmm_assign, token_text


def test_original_synthetic_branches():
    screen=t._synthetic_screen(offsets_ans={'C1':.5})
    decision=t.step2_decision(screen,True)
    assert decision['chosen']=='C1' and decision['step3']=='3A'
    with pytest.raises(ValueError):t.step2_decision(screen.iloc[1:],True)
    bad=screen.copy();bad.loc[0,'m_ans']=np.inf
    with pytest.raises(ValueError):t.step2_decision(bad,True)


def test_votes_and_degenerate_vectors():
    assert t.majority_vote(['bad','ok','bad'],lambda s:'' if s=='bad' else s)=='ok'
    with pytest.raises(ValueError):t.chart_vectors(np.ones((2,4,1)),np.zeros(4))
    with pytest.raises(ValueError):t.manifold_controls(np.ones((4,5)),k=2)
    with pytest.raises(ValueError):t.bootstrap_mean_ci([1,np.inf])


def test_current_repository_segmenter_exact():
    # Load only the function AST, avoiding unrelated sklearn imports in GPU env.
    source=Path(__file__).parents[1]/'src/hss/data/openact.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sentence_boundaries')
    scope={'np':np,'re':__import__('re')};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    text='A. Next\n中文。 Last!';offsets=np.array([(i,i+1) for i in range(len(text))]+[(0,0)])
    assert sentence_boundaries(text,offsets,len(offsets))==scope['sentence_boundaries'](text,offsets,len(offsets))


def test_boxed_span_includes_closing_brace_and_no_prompt():
    text=r'old \boxed{2}, final \boxed{\frac{1}{3}}!'
    offsets=np.array([(i,i+1) for i in range(len(text))]+[(0,0)])
    ix=boxed_token_indices(text,offsets)
    assert ''.join(text[i] for i in ix)==r'\frac{1}{3}}'
    assert not len(boxed_token_indices(r'\boxed{}',np.array([(i,i+1) for i in range(8)])))


def test_gmm_posterior_includes_variance_and_prior():
    means=np.array([[0.,0.],[0.,0.]])
    assert gmm_assign([0,0],means,np.array([[1.,1.],[100.,100.]]),np.array([.5,.5]))[0]==0
    assert gmm_assign([30,30],means,np.array([[1.,1.],[100.,100.]]),np.array([.5,.5]))[0]==1


def test_torch_scientific_contracts():
    torch=pytest.importorskip('torch')
    from run_hss_followup import GeneratedPositions
    h=torch.arange(30).reshape(1,5,6).float()
    hook=GeneratedPositions(3,lambda x:x+1)
    result=hook(None,(),(h,'cache'))
    assert torch.equal(result[0][:,:3],h[:,:3]) and torch.equal(result[0][:,3:],h[:,3:]+1)
    assert result[1]=='cache' and hook.n==2
    assert max(t.check_torch_equivalence(device='cuda' if torch.cuda.is_available() else 'cpu').values())<1e-4
