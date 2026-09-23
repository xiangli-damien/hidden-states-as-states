from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from revision_locality_common import radial_control,gram_control,derangement,replacement,conditions,key


def decoder():
    return {'centers':np.array([[0.,0.,0.],[10.,0.,0.]]),
        'local_basis':np.array([[[0.,1.,0.]],[[0.,0.,1.]]]),
        'shared_basis':np.array([[0.,1.,0.]]),
        'local_empirical_centers':np.array([[1.,0.,0.],[11.,0.,0.]]),
        'local_empirical_basis':np.array([[[0.,1.,0.]],[[0.,0.,1.]]]),
        'shared_empirical_basis':np.array([[0.,1.,0.]]),'permutation_42':np.array([1,0])}


def test_common_anchor_and_wrong_basis_only_changes_directions():
    d=decoder();x=np.array([[2.,3.,4.],[12.,3.,4.]])
    local,_=replacement(x,d,{'method':'local','rank':1},'x')
    shared,_=replacement(x,d,{'method':'shared','rank':1},'x')
    wrong,_=replacement(x,d,{'method':'wrong_local','rank':1,'seed':42},'x')
    np.testing.assert_allclose(local,[[0,3,0],[10,0,4]])
    np.testing.assert_allclose(shared,[[0,3,0],[10,3,0]])
    np.testing.assert_allclose(wrong,[[0,0,4],[10,3,0]])
    for seed in [42,137,271]:assert np.all(derangement(64,seed)!=np.arange(64))


def test_radial_control_preserves_all_three_invariants_including_zero_activation():
    rng=np.random.default_rng(2);x=rng.normal(size=(8,30));x[0]=0;delta=rng.normal(size=x.shape)
    control=radial_control(x,delta,42)
    np.testing.assert_allclose((control**2).sum(1),(delta**2).sum(1),atol=1e-12)
    np.testing.assert_allclose((x*control).sum(1),(x*delta).sum(1),atol=1e-12)
    np.testing.assert_allclose(((x+control)**2).sum(1),((x+delta)**2).sum(1),atol=1e-12)
    assert not np.allclose(control,delta)


def test_error_gram_rotation_preserves_correlation_even_when_rank_deficient():
    rng=np.random.default_rng(4);delta=rng.normal(size=(6,24));delta[2]=delta[1];delta[3]=0
    control=gram_control(delta,42)
    np.testing.assert_allclose(control@control.T,delta@delta.T,rtol=1e-12,atol=1e-12)
    assert not np.allclose(control,delta)


def test_clean_ablation_and_reconstruction_are_different_complements():
    d=decoder();x=np.array([[2.,3.,4.],[12.,3.,4.]])
    reconstruct,_=replacement(x,d,{'method':'local','rank':1},'x')
    remove,_=replacement(x,d,{'method':'remove_local','rank':1,'alpha':1.},'x')
    other,_=replacement(x,d,{'method':'remove_complement','rank':1,'alpha':1.},'x')
    np.testing.assert_allclose(other,reconstruct)
    np.testing.assert_allclose(remove,[[2,0,4],[12,3,0]])
    a,_=replacement(x,d,{'method':'remove_local_radial_random','rank':1,'alpha':.25,'seed':42},'x')
    b,_=replacement(x,d,{'method':'remove_local_radial_random','rank':1,'alpha':1.,'seed':42},'x')
    np.testing.assert_allclose(a-x,.25*(b-x),atol=1e-12)


def test_predeclared_conditions_are_unique():
    cfg={'ranks':[4,8,16,32],'shared_ranks':[4,8,16,32,64,128,256,512],'seeds':[42,137,271],'alphas':[.25,.5,1.]}
    for full,expected in [(True,48),(False,14)]:
        values=conditions(cfg,full)
        assert len(values)==expected and len({key(c) for c in values})==expected


def test_full_reference_and_early_loss_match_direct_tiny_qwen():
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from evaluate_revision_locality import measure
    torch.manual_seed(42)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).eval()
    prefix=[2,3,4,5];reference=[6,7,8,9,10]
    metric,logp,losses=measure(model,prefix,reference,1,[1,2,3],lambda h:h)
    with torch.inference_mode():
        logits=model(torch.tensor([prefix+reference[:-1]]),use_cache=False).logits[0,len(prefix)-1:]
        expected=torch.nn.functional.cross_entropy(logits,torch.tensor(reference),reduction='none').numpy()
    np.testing.assert_allclose(losses,expected,atol=1e-6)
    assert metric['patch_calls']==1 and metric['actual_patch_energy']==0
    assert metric['nll']==pytest.approx(float(expected.mean()),abs=1e-6)
    assert metric['first_token_nll']==pytest.approx(float(expected[0]),abs=1e-6)
