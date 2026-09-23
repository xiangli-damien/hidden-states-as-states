import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from revision_common import OncePatch, nearest, reconstruct, nmse_rows, paired_ratio_ci


def test_reconstruction_keeps_actual_tokens_distinct_and_uses_training_anchor():
    d={'centers': np.array([[0.,0.],[10.,0.]],np.float32),
       'train_mean': np.array([5.,0.],np.float32),
       'global_basis':np.array([[0.,1.]],np.float32),
       'local_basis':np.array([[[0.,1.]],[[0.,1.]]],np.float32)}
    x=np.array([[1.,3.],[11.,-2.]],np.float32)
    np.testing.assert_array_equal(nearest(x,d['centers']),[0,1])
    np.testing.assert_allclose(reconstruct(x,d,'local_pca_1'),[[0,3],[10,-2]])
    np.testing.assert_allclose(reconstruct(x,d,'global_pca_1'),[[5,3],[5,-2]])
    a,b=nmse_rows(x,reconstruct(x,d,'centroid'),d['train_mean'])
    np.testing.assert_allclose(a,[10,5]);np.testing.assert_allclose(b,[25,40])
    assert paired_ratio_ci(a,b,boot=20)['estimate']==pytest.approx(15/65)


def test_nmse_denominator_is_not_test_centered():
    x=np.array([[10.,0.],[10.,0.]])
    a,b=nmse_rows(x,np.zeros_like(x),np.zeros(2))
    assert paired_ratio_ci(a,b,boot=20)['estimate']==1
    assert paired_ratio_ci([0,0],[0,0],boot=20)['estimate'] is None


def test_patch_identity_position_and_decode_cache_contract():
    torch=pytest.importorskip('torch')
    h=torch.arange(30).reshape(1,5,6).float()
    patch=OncePatch([1,3],5,lambda z:z+2)
    out=patch(None,None,h)
    assert torch.equal(h[0,0],out[0,0])
    assert torch.equal(out[0,[1,3]],h[0,[1,3]]+2)
    assert patch.calls==1 and patch.energy==48
    assert patch(None,None,torch.zeros(1,1,6)) is None
    identity=OncePatch([0,1,2,3,4],5,lambda z:z)
    assert torch.equal(identity(None,None,(h,'cache'))[0],h)
    assert identity.energy==0
    with pytest.raises(ValueError,match='fresh'):
        OncePatch([0],5,lambda z:z)(None,None,torch.zeros(1,1,6))


def test_basis_rank_and_geometric_center_are_explicit():
    pytest.importorskip('sklearn')
    from fit_revision_geometry import basis
    x=np.array([[10.,0.],[11.,0.],[12.,0.]])
    b=basis(x,4,np.zeros(2))
    assert b.shape==(4,2)
    np.testing.assert_allclose(b[2:],0)
    np.testing.assert_allclose(((x@b.T)@b),x,atol=1e-5)
    assert not basis(np.zeros((0,2)),4,np.zeros(2)).any()


def test_teacher_forced_nll_includes_every_reference_token_and_hook_is_once():
    torch=pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from evaluate_revision_patches import teacher_force
    torch.manual_seed(7)
    cfg=Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=24,
                   num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)
    model=Qwen2ForCausalLM(cfg).eval()
    prefix=[3,4,5,6];reference=[7,8,9,10,11]
    result,logp=teacher_force(model,prefix,reference,1,[1,2,3],lambda x:x)
    with torch.inference_mode():
        value=model(torch.tensor([prefix+reference[:-1]]),use_cache=False).logits[0]
        expected=torch.nn.functional.cross_entropy(value[len(prefix)-1:],torch.tensor(reference),reduction='sum')
    assert result['nll_sum']==pytest.approx(float(expected),abs=1e-5)
    assert result['reference_tokens']==5 and result['patch_calls']==1 and result['actual_patch_energy']==0
    np.testing.assert_allclose(logp,value[len(prefix)-1].log_softmax(-1).numpy(),atol=1e-6)


def test_window_energy_control_and_beta_endpoints():
    torch=pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from evaluate_revision_patches import replacement
    x=torch.tensor([[1.,2.],[2.,3.],[4.,3.]])
    decoder={'centers':np.zeros((1,2),np.float32)}
    whole=replacement(decoder,'centroid_energy1',42)(x)
    one=replacement(decoder,'centroid_energy1',42)(x[-1:])
    assert float((whole-x).square().sum())==pytest.approx(float((one-x[-1:]).square().sum()))
    rnd=replacement(decoder,'matched_random_energy1',42)(x)
    assert float((whole-x).square().sum())==pytest.approx(float((rnd-x).square().sum()))
    assert torch.equal(replacement(decoder,'beta_1',42)(x),x)
    assert not replacement(decoder,'beta_0',42)(x).any()


def test_full_small_geometry_stage_with_actual_split_names(tmp_path):
    pytest.importorskip('sklearn')
    import pandas as pd
    import json
    from fit_revision_geometry import fit_one
    from revision_common import write_npz,write_json
    rng=np.random.default_rng(42);n=200;d=8
    y=np.arange(n)%2
    x=(rng.normal(size=(n,d))+y[:,None]*5).astype(np.float32)
    rows=pd.DataFrame({'sample_id':[f'q{i}' for i in range(n)],
        'split':['train']*120+['validation']*40+['test']*40,'label':y,
        'category':['algebra']*n,'level':[1]*n,'n_prompt_tokens':rng.integers(30,80,n)})
    folder=tmp_path/'prefixes/shard_00000_00200';folder.mkdir(parents=True)
    rows.to_parquet(folder/'rows.parquet',index=False);write_json(folder/'_SUCCESS.json',{})
    write_npz(folder/'prefix_0.npz',valid=np.ones(n,bool),window=np.repeat(x[:,None,None,:],16,axis=2))
    confidence=tmp_path/'confidence';confidence.mkdir()
    pd.DataFrame({'sample_id':rows.sample_id,'next_token_entropy':rng.random(n),
                  'next_token_logit_margin':rng.random(n)}).to_parquet(confidence/'prefix_0.parquet',index=False)
    cfg={'output':str(tmp_path),'layers':[7],'k_grid':[1,2],'seeds':[42],'pca_ranks':[1],'blas_threads':1}
    out=fit_one((cfg,0,7,'last'))
    assert out['train_questions']==120 and out['val_questions']==40 and out['test_questions']==40
    assert out['selected']['converged'] and out['normalization']=='none'
    results=pd.read_parquet(tmp_path/'geometry/p0_l7_last/prediction_per_question.parquet')
    np.testing.assert_array_equal(results.failure,1-y)
    assert out['prediction']['state_nb']['test_auroc']>.9
    assert (tmp_path/'geometry/p0_l7_last/_SUCCESS.json').exists()
