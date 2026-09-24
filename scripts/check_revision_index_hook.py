"""CPU-only actual Qwen generate/capture/hook contract; random small model."""
import json
import numpy as np
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from evaluate_revision_index_interchange import capture,generate
from revision_index_interchange_common import transfer


class TokenizerStub:
    pad_token_id=0
    eos_token_id=30
    def decode(self,ids,**kwargs):return ' '.join(map(str,ids))


def run():
    torch.set_num_threads(2);torch.manual_seed(23)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=24,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,
        pad_token_id=0,eos_token_id=30)).eval()
    model.generation_config.eos_token_id=30
    recipient=list(range(1,19));donor=list(range(2,20));pos=list(range(2,18));tok=TokenizerStub()
    x=capture(model,recipient,pos,2);y=capture(model,donor,pos,2)
    baseline=generate(model,tok,recipient)
    identity=generate(model,tok,recipient,pos,lambda h:h,2)
    assert baseline['generated_ids']==identity['generated_ids'] and identity['patch_calls']==1
    centers=np.zeros((1,16));basis=np.eye(16)[:8];calls=[]
    def transform(h):
        np.testing.assert_array_equal(h.detach().numpy(),x)
        z=transfer(x,y,centers,basis[None],basis,'local8')
        np.testing.assert_allclose(z[:,:8],y[:,:8],atol=1e-8)
        np.testing.assert_array_equal(z[:,8:],x[:,8:])
        calls.append(True);return torch.tensor(z,dtype=h.dtype)
    changed=generate(model,tok,recipient,pos,transform,2)
    assert len(calls)==1 and changed['patch_calls']==1 and changed['actual_patch_energy']>0
    return {'passed':True,'device':'cpu','random_small_model_only':True,
            'identity_generation_exact':True,'capture_matches_generate_prefill':True,
            'nonidentity_projected_donor_patch_once':True,'fresh_cache':True}


if __name__=='__main__':print(json.dumps(run()))
