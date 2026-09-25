import copy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from promptlast_tokenwise_common import selected_cases, names, trajectory, metrics


def test_selection_and_metrics():
    cases=[{'dataset':ds,'sample_id':str(i)+ds} for i in range(5) for ds in ['math','gsm8k']]
    chosen=selected_cases(cases,3)
    assert [r['sample_id'] for r in chosen]==['0math','1math','2math','0gsm8k','1gsm8k','2gsm8k']
    lp=np.log([[.2,.8],[.5,.5]])
    d={'centers':np.array([[0.,0.],[2.,2.]]),'train_mean':np.zeros(2)}
    r=metrics(lp,np.array([2.,1.]),np.array([[2.,1.],[2.,2.]]),1,d)
    np.testing.assert_allclose(r['kl'],[0,.2*np.log(.4)+.8*np.log(1.6)])
    assert r['region']==1 and r['nmse']==[0,.2]


def test_tiny_qwen_independent_and_cumulative_match_full_replay():
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from promptlast_replacement_common import apply
    torch.manual_seed(29)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=37,hidden_size=12,intermediate_size=24,
        num_hidden_layers=3,num_attention_heads=3,num_key_value_heads=3,attn_implementation='eager')).eval()
    rng=np.random.default_rng(33)
    d={'centers':rng.normal(size=(3,12))*.2,'train_mean':np.zeros(12),
       'local_basis':np.stack([np.linalg.qr(rng.normal(size=(12,8)))[0].T for _ in range(3)])}
    d['shared_basis']=np.linalg.qr(rng.normal(size=(12,8)))[0].T
    prompt=[1,2,3,4];response=[5,6,7,8];methods=['centroid','shared8','local8'];order=names(methods)
    outputs=list(trajectory(model,prompt,response,d,1,methods))
    changed=False
    for p,r in enumerate(outputs):
        assert r['target_id']==response[p] and r['input_id']==(prompt[-1] if p==0 else response[p-1])
        seq=prompt+response[:p]
        for j,name in enumerate(order):
            positions=[] if name=='identity' else ([len(seq)-1] if name.startswith('independent') else list(range(len(prompt)-1,len(seq))))
            def hook(module,args,out):
                h=out[0] if isinstance(out,tuple) else out;value=h.clone()
                for pos in positions:
                    z,_=apply(h[0,pos:pos+1].detach().numpy(),d,name.split('_',1)[1])
                    value[0,pos]=torch.tensor(z[0],dtype=h.dtype)
                return (value,)+out[1:] if isinstance(out,tuple) else value
            handle=model.model.layers[0].register_forward_hook(hook)
            with torch.inference_mode():full=model(torch.tensor([seq]),use_cache=False).logits[0,-1].float().log_softmax(-1).numpy()
            handle.remove()
            np.testing.assert_allclose(r['logp'][j],full,rtol=2e-6,atol=2e-6)
        if p>0:changed |= not np.allclose(r['logp'][1],r['logp'][4],atol=1e-6,rtol=0)
    assert changed, 'Cumulative history must actually affect later logits.'
