from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from delta_online_common import route_features,RouteBayes,prefix_controls,posterior_region,replacement,FUNCTIONAL_METHODS


def test_prefix_controls_cannot_use_future_or_next_target():
    rng=np.random.default_rng(1);conf=rng.normal(size=(20,4));kinds=rng.integers(0,6,20)
    original=prefix_controls(conf,77,kinds,8)
    conf[9:]=1e10;conf[8,3]=1e10;kinds[8:]=5
    np.testing.assert_array_equal(original,prefix_controls(conf,77,kinds,8))
    conf[7,3]+=1
    assert prefix_controls(conf,77,kinds,8)[3]==pytest.approx(original[3]+1/8)


def test_route_counts_and_layer_numbering_invariance():
    a=np.array([[0,1],[0,0],[1,0],[1,1]])
    x,shape=route_features(a,[2,2],True,True)
    np.testing.assert_allclose(x[:4],[.5,.5,.5,.5]);np.testing.assert_allclose(x[4:8],[.25]*4)
    np.testing.assert_allclose(x[8:12],[1/3,1/3,0,1/3])
    b,_=route_features(a[[0,2,1,3]],[2,2],True,True)
    np.testing.assert_array_equal(x[:8],b[:8]);assert not np.array_equal(x[8:],b[8:])
    seq=[a,np.roll(a,1,axis=0),a[::-1],a[[0,2,1,3]]];y=np.array([0,1,0,1])
    xx=np.stack([route_features(v,[2,2],True,True)[0] for v in seq]);flipped=np.stack([route_features(v*np.array([-1,1])+np.array([1,0]),[2,2],True,True)[0] for v in seq])
    np.testing.assert_allclose(RouteBayes(shape).fit(xx,y).decision_function(xx),RouteBayes(shape).fit(flipped,y).decision_function(flipped))


def decoder():
    rng=np.random.default_rng(17);d={'centers':rng.normal(size=(3,12))*.1,'variances':rng.uniform(.2,1,(3,12)),'weights':np.array([.2,.5,.3]),'train_mean':rng.normal(size=12)*.01}
    d['shared_basis']=np.linalg.qr(rng.normal(size=(12,8)))[0].T
    d['local_basis']=np.stack([np.linalg.qr(rng.normal(size=(12,8)))[0].T for _ in range(3)]);d['permutation']=np.array([1,2,0]);return d


def test_posterior_matches_diagonal_gmm_and_update_keeps_input():
    pytest.importorskip('sklearn')
    from sklearn.mixture import GaussianMixture
    d=decoder();gm=GaussianMixture(3,covariance_type='diag');gm.weights_=d['weights'];gm.means_=d['centers'];gm.covariances_=d['variances'];gm.precisions_cholesky_=1/np.sqrt(d['variances'])
    x=np.random.default_rng(9).normal(size=(18,12));np.testing.assert_array_equal(gm.predict(x),posterior_region(x,d))
    np.testing.assert_array_equal(replacement(x,x+1,d,d,'zero_update'),x)
    np.testing.assert_allclose(replacement(x,x+1,d,d,'global_update'),x+d['train_mean'])
    for method in FUNCTIONAL_METHODS:
        assert replacement(x,x+1,d,d,method).shape==x.shape


def test_real_tiny_qwen_delta_patch_cached_matches_full_replay():
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from revision_common import OncePatch
    torch.manual_seed(44);model=Qwen2ForCausalLM(Qwen2Config(vocab_size=41,hidden_size=12,intermediate_size=24,num_hidden_layers=3,num_attention_heads=3,num_key_value_heads=3)).eval();d=decoder()
    def run(method,ids,positions,use_cache):
        cap={}
        def pre(module,args,kwargs):
            h=args[0] if args else kwargs['hidden_states'];cap['before']=h[0,positions].detach().numpy()
        def trans(h):return torch.tensor(replacement(cap['before'],h.detach().numpy(),d,d,method),dtype=h.dtype)
        a=model.model.layers[1].register_forward_pre_hook(pre,with_kwargs=True);patch=OncePatch(positions,len(ids),trans);b=model.model.layers[1].register_forward_hook(patch)
        try:
            with torch.no_grad():out=model(torch.tensor([ids]),use_cache=use_cache)
        finally:a.remove();b.remove()
        assert patch.calls==1;return out
    for width in [1,4]:
        positions=list(range(6-width,6))
        for method in FUNCTIONAL_METHODS:
            out=run(method,[1,2,3,4,5,6],positions,True)
            with torch.no_grad():cached=model(torch.tensor([[7]]),past_key_values=out.past_key_values,use_cache=True).logits[0,-1]
            full=run(method,[1,2,3,4,5,6,7],positions,False).logits[0,-1]
            torch.testing.assert_close(cached,full,rtol=1e-5,atol=1e-6)
        identity=run('identity',[1,2,3,4,5,6],positions,False)
        with torch.no_grad():plain=model(torch.tensor([[1,2,3,4,5,6]]),use_cache=False)
        torch.testing.assert_close(identity.logits,plain.logits,rtol=0,atol=0)


def test_future_tokens_do_not_change_prefix_hidden_states():
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(4);model=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=12,intermediate_size=24,num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=3)).eval()
    with torch.no_grad():
        a=model(torch.tensor([[1,2,3,4,5]]),output_hidden_states=True,use_cache=False)
        b=model(torch.tensor([[1,2,3,9,8]]),output_hidden_states=True,use_cache=False)
    for x,y in zip(a.hidden_states,b.hidden_states):torch.testing.assert_close(x[:,:3],y[:,:3],rtol=0,atol=0)
