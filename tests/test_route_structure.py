import numpy as np
from scipy.special import logsumexp
from hss.route.structure import RouteEncoder, BackoffMarkov, MixtureMarkov, LayerHMM


def test_encoder_unknown_and_id_invariance():
    z=np.array([[3,9],[4,8],[3,8]])
    e=RouteEncoder().fit(z)
    assert e.transform(np.array([[99,8]])).tolist()==[[2,0]]
    assert np.array_equal(e.onehot(e.transform(z)).sum(1),np.full(3,2))
    a=BackoffMarkov(e.sizes).fit(e.transform(z)).losses(e.transform(z))
    zz=z+np.array([100,300]);ee=RouteEncoder().fit(zz)
    b=BackoffMarkov(ee.sizes).fit(ee.transform(zz)).losses(ee.transform(zz))
    np.testing.assert_allclose(a,b)


def test_markov_conditional_distribution_and_no_test_fit():
    z=np.array([[0,0],[1,1]]*10);m=BackoffMarkov([3,3],order=1).fit(z)
    queries=np.array([[0,0],[0,1],[0,2]])
    np.testing.assert_allclose(np.exp(-m.losses(queries)[:,1]).sum(),1.)
    assert m.losses(np.array([[0,1]])).mean()>m.losses(np.array([[0,0]])).mean()
    assert len(m.counts[1,1])==2


def test_mixture_chain_factorization():
    z=np.random.default_rng(3).integers(0,3,(100,6))
    m=MixtureMarkov([3]*6,components=2,max_iter=10).fit(z)
    joint=logsumexp(m.component_logprob(z)+np.log(m.weights),axis=1)
    np.testing.assert_allclose(-m.losses(z).sum(1),joint)


def test_hmm_forward_matches_enumeration():
    from itertools import product
    z=np.array([[0,1,0],[1,0,1]])
    m=LayerHMM([2]*3,components=2,max_iter=4).fit(z)
    for i,row in enumerate(z):
        total=0.
        for h in product(range(2),repeat=3):
            p=m.initial[h[0]]*m.emissions[0][h[0],row[0]]
            for l in range(1,3):p*=m.transitions[l-1,h[l-1],h[l]]*m.emissions[l][h[l],row[l]]
            total+=p
        np.testing.assert_allclose(np.exp(-m.losses(z)[i].sum()),total)
    initial=float(-m.losses(z).sum(1).mean())
    m.fit(z,warm_start=True)
    np.testing.assert_allclose(m.trace[0],initial)


def test_causal_no_future_leakage_and_suffix_preservation():
    import pytest
    torch=pytest.importorskip('torch')
    from hss.route.neural import RouteNet,suffix_negatives
    z=torch.randint(0,3,(40,6));other=z.clone();other[:,3:]=(other[:,3:]+1)%3
    for kind in ['gru','causal_transformer']:
        model=RouteNet([3]*6,kind,16).eval()
        with torch.no_grad():a=model.sequence_logits(z);b=model.sequence_logits(other)
        torch.testing.assert_close(a[:,:4],b[:,:4])
    swapped,_=suffix_negatives(z)
    for l in range(5):
        assert torch.equal(torch.bincount(z[:,l]*3+z[:,l+1],minlength=9),torch.bincount(swapped[:,l]*3+swapped[:,l+1],minlength=9))


def test_saved_route_inference_does_not_require_labels(tmp_path):
    import joblib,json
    from hss.route.inference import score_new_routes
    z=np.array([[10,30,50],[20,40,60],[10,30,50]]*10)
    encoder=RouteEncoder().fit(z);zz=encoder.transform(z)
    model=BackoffMarkov(encoder.sizes).fit(zz)
    folder=tmp_path/'models'/'mfa'/'markov';folder.mkdir(parents=True)
    joblib.dump(encoder,folder.parent/'encoder.joblib');joblib.dump(model,folder/'model.joblib')
    (tmp_path/'selected.json').write_text(json.dumps([dict(map='mfa',method='markov',candidates=['models/mfa/markov'])]))
    out=score_new_routes(tmp_path,'mfa',z)
    np.testing.assert_allclose(out['markov__mean'],model.losses(zz).mean(1))
    assert np.isfinite(score_new_routes(tmp_path,'mfa',np.array([[99,99,99]]))['markov__mean']).all()


def test_vectorized_bootstrap_matches_tied_auc():
    import sys
    from pathlib import Path
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from evaluate_route_unsupervised import auc_bootstrap
    rng=np.random.default_rng(5);y=np.r_[np.zeros(50),np.ones(50)]
    scores=rng.integers(0,8,100);idx=rng.integers(100,size=(231,100))
    np.testing.assert_allclose(auc_bootstrap(y,scores,idx),[roc_auc_score(y[a],scores[a]) for a in idx],atol=1e-12)
