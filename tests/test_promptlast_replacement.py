import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from promptlast_replacement_common import apply,conditions


def fixture():
    rng=np.random.default_rng(7);bases=np.stack([np.linalg.qr(rng.normal(size=(12,8)))[0].T for _ in range(3)])
    d={'centers':rng.normal(size=(3,12)), 'train_mean':rng.normal(size=12),'local_basis':bases,'shared_basis':bases[0]}
    for seed in [42,137,271]:d['permutation_'+str(seed)]=np.array([1,2,0])
    return rng.normal(size=(1,12)),d


def test_raw_nearest_and_independent_projections():
    x,d=fixture();s=int(np.argmin(((d['centers']-x)**2).sum(1)))
    for name in conditions():
        z,meta=apply(x,d,name);assert meta['region']==s
        if name=='identity':expected=x
        elif name=='global_mean':expected=d['train_mean'][None]
        elif name=='centroid':expected=d['centers'][s][None]
        else:
            b=d['shared_basis'] if name=='shared8' else d['local_basis'][s if name=='local8' else d['permutation_'+name.rsplit('_',1)[1]][s]]
            expected=d['centers'][s].copy()
            for axis in b:expected+=np.dot(x[0]-d['centers'][s],axis)*axis
            expected=expected[None]
        np.testing.assert_allclose(z,expected,atol=1e-12)


def test_rejects_mean_window_or_nonfinite():
    x,d=fixture()
    with pytest.raises(ValueError):apply(np.repeat(x,16,axis=0),d,'local8')
    x[0,0]=np.nan
    with pytest.raises(ValueError):apply(x,d,'identity')


def test_real_tiny_qwen_single_prompt_position_and_cached_continuation():
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from revision_common import OncePatch
    torch.manual_seed(9);model=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=12,
        intermediate_size=24,num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=3)).eval()
    _,d=fixture();ids=torch.tensor([[1,2,3,4]])
    for name in conditions():
        def transform(h):return torch.tensor(apply(h.detach().numpy(),d,name)[0],dtype=h.dtype)
        patch=OncePatch([3],4,transform);handle=model.model.layers[0].register_forward_hook(patch)
        with torch.no_grad():out=model(ids,use_cache=True)
        handle.remove();assert patch.calls==1
        with torch.no_grad():cached=model(torch.tensor([[5]]),past_key_values=out.past_key_values,use_cache=True).logits
        replay=OncePatch([3],5,transform);handle=model.model.layers[0].register_forward_hook(replay)
        with torch.no_grad():full=model(torch.tensor([[1,2,3,4,5]]),use_cache=False).logits[:,-1:]
        handle.remove();assert replay.calls==1
        torch.testing.assert_close(cached,full,rtol=1e-5,atol=1e-6)
