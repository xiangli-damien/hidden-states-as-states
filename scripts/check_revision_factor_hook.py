"""CPU-only nonidentity hook check in OpenAct's pinned model runtime.

No scipy, sklearn, pytest or threadpoolctl dependency and no model download.
"""
import numpy as np
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from evaluate_revision_locality import measure
from revision_common import OncePatch
from revision_factor_common import FactorDecoder


def run():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=24,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2)).eval()
    decoder=FactorDecoder([1.],np.zeros((1,16)),np.random.default_rng(4).normal(size=(1,16,3)),np.ones((1,16)))
    def transform(h):
        z=decoder.assigned(h.detach().float().numpy(),np.zeros(len(h),int))
        return torch.from_numpy(z).to(h.dtype)
    prefix=[2,3,4,5];reference=[6,7,8,9,10];positions=[2,3]
    metric,logp,loss=measure(model,prefix,reference,2,positions,transform)
    ids=prefix+reference[:-1];patch=OncePatch(positions,len(ids),transform)
    hook=model.model.layers[1].register_forward_hook(patch)
    try:
        with torch.inference_mode():
            logits=model(torch.tensor([ids]),use_cache=False).logits[0,len(prefix)-1:]
            expected=torch.nn.functional.cross_entropy(logits,torch.tensor(reference),reduction='none').numpy()
            expected_logp=logits[0].log_softmax(-1).numpy()
    finally:hook.remove()
    assert patch.calls==metric['patch_calls']==1 and metric['actual_patch_energy']>0
    np.testing.assert_allclose(loss,expected,atol=1e-6)
    np.testing.assert_allclose(logp,expected_logp,atol=1e-6)
    return {'passed':True,'device':'cpu','nonidentity_factor_patch':True,
            'cached_vs_full_forward_max_nll_error':float(np.max(np.abs(loss-expected)))}


if __name__=='__main__':
    import json
    print(json.dumps(run()))
