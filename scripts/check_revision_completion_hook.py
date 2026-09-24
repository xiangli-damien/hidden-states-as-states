"""Actual tinyQwen cache test, not a scientific outcome or generation benchmark."""
import copy,json
import numpy as np
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from evaluate_revision_completion import continuation


def run():
    torch.set_num_threads(2);torch.manual_seed(42)
    m=Qwen2ForCausalLM(Qwen2Config(vocab_size=43,hidden_size=16,intermediate_size=32,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,eos_token_id=42)).eval()
    gen=copy.deepcopy(m.generation_config);gen.do_sample=False;gen.repetition_penalty=1.;gen.pad_token_id=0
    ids=list(range(1,21));pos=[16,17,18,19]
    identity=lambda x:(x.copy(),{})
    base,_=continuation(m,None,ids,12,gen)
    same,capture=continuation(m,None,ids,12,gen,2,pos,identity)
    assert base==same
    def operation(x):
        z=x.copy();z[:,::2]*=0
        return z,{}
    changed,saved=continuation(m,None,ids,12,gen,2,pos,operation)
    assert np.any(saved['before']!=saved['actual']) and not m.model.layers[1]._forward_hooks
    # Independently obtain the first and second patched logits by full forward
    # with the identical prefix positions; compare against fresh cached decoding.
    h=[]
    def hook(module,args,out):
        z=out.clone();z[0,pos]=torch.tensor(saved['actual'],dtype=z.dtype);return z
    handle=m.model.layers[1].register_forward_hook(hook)
    with torch.inference_mode():
        first=m(torch.tensor([ids]),use_cache=True)
        assert int(first.logits[0,-1].argmax())==changed[0]
        handle.remove()
        cached=m(torch.tensor([[changed[0]]]),past_key_values=first.past_key_values,use_cache=True).logits[0,-1]
        handle=m.model.layers[1].register_forward_hook(hook)
        full=m(torch.tensor([ids+[changed[0]]]),use_cache=False).logits[0,-1]
        handle.remove()
    torch.testing.assert_close(cached,full,rtol=2e-5,atol=2e-6)
    if len(changed)>1:assert int(cached.argmax())==changed[1]
    return {'complete':True,'random_small_model_only':True,'identity_exact':True,'nonidentity_patch':True,
            'hook_removed_after_prefill':True,'modified_cache_matches_full_replay':True,'max_cache_logit_error':float((cached-full).abs().max())}


if __name__=='__main__':print(json.dumps(run()))
