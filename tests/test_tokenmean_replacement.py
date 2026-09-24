import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tokenmean_replacement_common import apply, posterior, condition_names


def fixture():
    rng = np.random.default_rng(2)
    bases = np.stack([np.linalg.qr(rng.normal(size=(12, 8)))[0].T for _ in range(3)])
    d = {'centers': rng.normal(size=(3, 12)), 'variances': np.ones((3, 12)),
         'weights': np.array([.2, .3, .5]), 'local_basis': bases,
         'shared_basis': bases[0]}
    for seed in [42, 137, 271]: d['permutation_' + str(seed)] = np.array([1, 2, 0])
    return rng.normal(size=(16, 12)), d


def test_shift_preserves_all_token_differences_and_reconstructs_mean():
    x, d = fixture()
    for name in condition_names():
        if not name.startswith('mean_shift_'): continue
        z, meta = apply(x, d, d, name)
        np.testing.assert_allclose(z - z.mean(0), x - x.mean(0), atol=1e-14)
        direct, _ = apply(x.mean(0)[None], d, d, name.replace('mean_shift', 'token_project'))
        np.testing.assert_allclose(z.mean(0), direct[0], atol=1e-14)


def test_projection_and_shift_share_mean_but_not_token_variation():
    x, d = fixture()
    a, _ = apply(x, d, d, 'mean_shift_local8')
    b, _ = apply(x, d, d, 'token_project_local8')
    np.testing.assert_allclose(a.mean(0), b.mean(0), atol=1e-14)
    assert np.linalg.norm(a - b) > 1
    b2, _ = apply(x[::-1], d, d, 'token_project_local8')
    np.testing.assert_allclose(b2[::-1], b, atol=1e-14)


def test_centroid_and_identity_are_distinct_and_wrong_basis_keeps_anchor():
    x, d = fixture()
    identity, _ = apply(x, d, d, 'identity')
    np.testing.assert_array_equal(identity, x)
    z, m = apply(x, d, d, 'token_project_centroid')
    np.testing.assert_array_equal(z, np.broadcast_to(d['centers'][m['mean_region']], x.shape))
    anchor = d['centers'][m['mean_region']]
    # Manually check wrong basis still uses the recipient center.
    w, _ = apply(x, d, d, 'token_project_wrong8_42')
    basis = d['local_basis'][d['permutation_42'][m['mean_region']]]
    np.testing.assert_allclose(w, anchor + (x-anchor) @ basis.T @ basis)


def test_posterior_matches_direct_gaussian_density():
    x, d = fixture()
    p, logp = posterior(x, d)
    density = np.stack([d['weights'][k] * np.exp(-.5 * ((x-d['centers'][k])**2).sum(1))
                        / (2*np.pi)**6 for k in range(3)], axis=1)
    np.testing.assert_allclose(p, density/density.sum(1, keepdims=True))
    np.testing.assert_allclose(logp, np.log(density.sum(1)))
    assert len(condition_names()) == 16


def test_nonfinite_is_rejected():
    x, d = fixture(); x[0, 0] = np.nan
    with pytest.raises(ValueError): apply(x, d, d, 'identity')


def test_independent_audit_agrees_for_all_operators():
    from audit_tokenmean_replacement import expected_patch
    x,d=fixture()
    for name in condition_names():
        actual,meta=apply(x,d,d,name)
        expected,region=expected_patch(x,d,d,name)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        assert region==meta['mean_region']


def test_real_tiny_qwen_cached_and_full_replay():
    torch=pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from revision_common import OncePatch
    torch.manual_seed(5)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=12,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=3)).eval()
    ids=torch.tensor([[1,2,3,4,5,6]])
    _,d=fixture()
    for name in ['identity','mean_shift_local8','token_project_local8']:
        def transform(h):
            z,_=apply(h.detach().numpy(),d,d,name)
            return torch.tensor(z,dtype=h.dtype)
        patch=OncePatch([2,3,4,5],6,transform)
        handle=model.model.layers[0].register_forward_hook(patch)
        with torch.no_grad():first=model(ids,use_cache=True)
        handle.remove();assert patch.calls==1
        with torch.no_grad():continued=model(torch.tensor([[7]]),past_key_values=first.past_key_values,use_cache=True).logits
        fullpatch=OncePatch([2,3,4,5],7,transform)
        handle=model.model.layers[0].register_forward_hook(fullpatch)
        with torch.no_grad():full=model(torch.tensor([[1,2,3,4,5,6,7]]),use_cache=False).logits[:,-1:]
        handle.remove();assert fullpatch.calls==1
        torch.testing.assert_close(continued,full,rtol=1e-5,atol=1e-6)
