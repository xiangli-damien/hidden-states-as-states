"""Diagnose an embedding/input positive-control mismatch without relaxing it."""
import json
from pathlib import Path
import numpy as np
import torch
from extract_revision_prefixes import load_model
from revision_common import OncePatch, write_json, sha
from revision_counterfactual_gate import encode, full_transform, generate


@torch.inference_mode()
def run():
    root=Path('/lambda/nfs/dami/hss/revision-counterfactual-gate-20260923')
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    done={p.stem for p in (root/'cases').glob('*.json')}
    case=next(c for c in cfg['cases'] if c['id'] not in done)
    model,tokenizer=load_model(cfg)
    recipient,donor=case['recipient'],case['donor']
    edited={**recipient,'a':donor['a'],'b':donor['b'],'u':donor['u']}
    ids,_,_=encode(tokenizer,recipient,case['mode']);eids,_,_=encode(tokenizer,edited,case['mode'])
    changed=[i for i,(a,b) in enumerate(zip(ids,eids)) if a!=b]
    embedding=model.model.embed_tokens(torch.tensor([eids[i] for i in changed],device=model.device)).float().cpu().numpy()
    true=model(torch.tensor([eids],device=model.device),use_cache=False,logits_to_keep=1).logits.float().cpu().numpy()
    patch=OncePatch(changed,len(ids),full_transform(embedding));hook=model.model.embed_tokens.register_forward_hook(patch)
    try:swapped=model(torch.tensor([ids],device=model.device),use_cache=False,logits_to_keep=1).logits.float().cpu().numpy()
    finally:hook.remove()
    results=[]
    original_penalty=model.generation_config.repetition_penalty
    for penalty in [original_penalty,1.0]:
        model.generation_config.repetition_penalty=penalty
        cf=generate(model,tokenizer,eids)
        swap=generate(model,tokenizer,ids,model.model.embed_tokens,changed,full_transform(embedding))
        results.append({'repetition_penalty':penalty,'true_input':cf,'embedding_swap':swap,
                        'generated_ids_exact':cf['generated_ids']==swap['generated_ids']})
    result={'case':case,'input_ids':ids,'edited_input_ids':eids,'changed_positions':changed,
            'raw_next_logits_exact':bool(np.array_equal(true,swapped)),
            'raw_next_logits_max_abs_difference':float(np.abs(true-swapped).max()),
            'generation_comparisons':results,'pinned_original_generation_config_repetition_penalty':original_penalty,
            'old_plan_sha256':sha(root/'plan.json'),'diagnostic_code_sha256':sha(Path(__file__))}
    write_json(root/'embedding_failure_diagnostic.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
