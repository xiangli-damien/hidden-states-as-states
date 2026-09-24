"""CPU tiny-Qwen execution check for every v2 operator; no scientific outcomes."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from evaluate_revision_completion import continuation
from projection_v2_common import project,catalog,derangement
from projection_v2_audit_math import independent_projection
from revision_common import sha,write_json


def run(root):
    torch.set_num_threads(2);torch.manual_seed(9242026)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=59,hidden_size=80,intermediate_size=128,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,eos_token_id=None)).eval()
    gen=copy.deepcopy(model.generation_config);gen.do_sample=False;gen.repetition_penalty=1.;gen.pad_token_id=0
    rng=np.random.default_rng(24);bases=np.stack([np.linalg.qr(rng.normal(size=(80,80)))[0].T for _ in range(3)])
    decoder=dict(centers=rng.normal(size=(3,80))*.01,local_basis=bases,shared_basis=bases[0]);perm=derangement(3,24)
    completed=[];max_logit_error=0.
    for c in catalog():
        ids=[1,2,3,4]+[5+i%40 for i in range(c['prefix'])];positions=list(range(len(ids)-c['width'],len(ids)))
        base,_=continuation(model,None,ids,3,gen)
        no_op,captured=continuation(model,None,ids,3,gen,2,positions,lambda x:(x.copy(),{}))
        assert no_op==base
        changed,saved=continuation(model,None,ids,3,gen,2,positions,lambda x:project(x,decoder,c,perm))
        expected,_,_=independent_projection(saved['before'],decoder['centers'],bases,bases[0],c,perm,1e-12)
        np.testing.assert_allclose(saved['ideal'],expected,rtol=1e-10,atol=1e-9)
        assert not model.model.layers[1]._forward_hooks
        def patch(module,args,out):
            result=out.clone();result[0,positions]=torch.tensor(saved['actual'],dtype=result.dtype);return result
        handle=model.model.layers[1].register_forward_hook(patch)
        with torch.inference_mode():
            first=model(torch.tensor([ids]),use_cache=True);handle.remove()
            assert int(first.logits[0,-1].argmax())==changed[0]
            cached=model(torch.tensor([[changed[0]]]),past_key_values=first.past_key_values,use_cache=True).logits[0,-1]
            handle=model.model.layers[1].register_forward_hook(patch)
            full=model(torch.tensor([ids+[changed[0]]]),use_cache=False).logits[0,-1];handle.remove()
        torch.testing.assert_close(cached,full,rtol=2e-5,atol=2e-6)
        max_logit_error=max(max_logit_error,float((cached-full).abs().max()));completed.append(c['name'])
    result=dict(complete=True,scope='Random tiny-Qwen CPU execution only; not evidence of correctness benefit',
        operators=completed,identity_exact=True,cache_matches_full_replay=True,max_cache_logit_error=max_logit_error,
        hook_removal_verified=True,source_sha256=sha(Path(__file__)))
    if root is not None:write_json(root/'preflight_SUCCESS.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);run(p.parse_args().root)
