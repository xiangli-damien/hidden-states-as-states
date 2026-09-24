"""Independent raw patch, token, label and pairing audit for a completed v2 batch."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer
from openact_eval.evaluators.registry import auto_select_evaluator
from revision_common import sha,write_json


from projection_v2_audit_math import independent_projection


def run(root,batch):
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config'];tier=json.loads((root/'tier.json').read_text())
    rc=json.loads((root/f'{batch}_SUCCESS.json').read_text())
    assert rc['complete'] and rc['plan_sha256']==sha(root/'plan.json') and rc['tier_sha256']==sha(root/'tier.json')
    for p,h in plan['files'].items():assert sha(p)==h,p
    assert rc['execution_plan_sha256']==sha(root/(batch+'_execution.json'))
    for p,h in json.loads((root/(batch+'_execution.json')).read_text())['files'].items():assert sha(p)==h
    assert rc['generation_config_sha256']==sha(root/'generation_config.json')
    generation=json.loads((root/'generation_config.json').read_text())['resolved']
    assert generation['do_sample'] is False and generation['repetition_penalty']==1 and generation['max_new_tokens']==2048
    eos=generation['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
    tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
    cohort='transfer' if batch=='transfer' else 'validation'
    expected_rows=[r for r in json.loads((root/'inputs.json').read_text()) if r['cohort']==cohort]
    if cohort=='transfer':expected_rows=expected_rows[:tier['transfer_n']]
    source={r['sample_id']:r for r in expected_rows};reuse=json.loads((root/'reuse.json').read_text())
    if batch=='transfer':conditions=[c for c in plan['transfer_conditions'] if c['name'] in tier['transfer_conditions']]
    else:
        c=next(c for c in plan['conditions'] if c['name']==batch);conditions=[c]
        if c['prefix']!=16:conditions.insert(0,dict(c,name='baseline',operator='baseline'))
    assert conditions==rc['conditions']
    ev=auto_select_evaluator('gsm8k' if cohort=='transfer' else 'math')
    with np.load(cfg['decoder']) as f:centers=f['centers'].astype(float);local=f['local_basis'].astype(float);shared=f['shared_basis'].astype(float)
    seen=set();max_error=0.;fallbacks=0
    for rel,h in rc['records'].items():
        path=root/rel;assert sha(path)==h
        r=json.loads(path.read_text());sid=r['sample_id'];row=source[sid];c=r['condition']
        assert c in conditions and (sid,c['name']) not in seen;seen.add((sid,c['name']))
        assert r['plan_sha256']==sha(root/'plan.json') and r['ground_truth']==row['ground_truth'] and r['prompt_text']==row['prompt_text']
        assert json.loads(path.with_suffix('.receipt.json').read_text())['record_sha256']==h
        for p,d in r['files'].items():assert sha(path.parent/p)==d
        pre=json.loads((path.parent/'prefix.json').read_text());p=pre['generated_prefix'];ids=r['generated_ids']
        assert pre['prompt_ids']==row['prompt_ids'] and pre['plan_sha256']==sha(root/'plan.json')
        if pre['reused_source']:
            orig=pre['reused_source'];assert sha(orig['path'])==orig['sha256']
            assert json.loads(Path(orig['path']).read_text())['generated_prefix']==p
        rendered=tok.apply_chat_template([dict(role='user',content=row['prompt_text'])],tokenize=False,add_generation_prompt=True)
        assert tok(rendered,add_special_tokens=False).input_ids==row['prompt_ids']
        assert ids[:len(p)]==p and 0<len(ids)<=2048 and not eos.intersection(ids[:-1])
        assert len(p)<=c['prefix'] and not eos.intersection(p[:-1])
        active=len(p)==c['prefix'] and not eos.intersection(p)
        assert bool(active)==r['prefix_active']==pre['active']
        positions=list(range(len(row['prompt_ids'])+len(p)-c['width'],len(row['prompt_ids'])+len(p))) if active else []
        assert r['positions']==positions and r['hook_calls']==int(active) and r['fresh_cache'] and r['hook_removed_after_prefill']
        assert r['length']==len(ids) and r['finish_reason']==('eos' if ids[-1] in eos else 'length')
        if r['finish_reason']=='length':assert len(ids)==2048
        text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False);assert text==r['response_text']
        score=ev.evaluate_sample(SimpleNamespace(sample_idx=row['source_sample_idx'],response_text=text,ground_truth=row['ground_truth'],meta={'sample_id':sid}))
        assert not score.error and bool(score.is_correct)==r['correct'] and bool(score.meta['parse_failed'])==r['parse_failed']
        assert score.extracted_answer==r['parsed_answer'] and score.normalized_answer==r['normalized_answer']
        if active:
            with np.load(path.with_suffix('.npz')) as f:x=f['before'].astype(float);ideal=f['ideal'];actual=f['actual']
            assert x.shape==(c['width'],centers.shape[1]) and np.isfinite(actual).all()
            z,labels,fallback=independent_projection(x,centers,local,shared,c,plan['permutation'],cfg['near_zero_norm'])
            np.testing.assert_allclose(ideal,z,rtol=1e-10,atol=1e-9)
            np.testing.assert_array_equal(actual,torch.tensor(ideal).to(torch.bfloat16).float().numpy())
            max_error=max(max_error,float(np.abs(z-ideal).max()));fallbacks+=int(fallback.sum())
            meta=r['geometry']['operator_metadata'];assert meta['current']==labels.tolist() and meta['norm_fallback']==fallback.tolist()
            donors=np.asarray(plan['permutation'])[labels] if c['operator']=='wrong' else labels
            assert meta['basis_region']==donors.tolist()
            after=np.square(actual.astype(float)[:,None]-centers[None]).sum(-1).argmin(1)
            assert r['geometry']['region_after']==after.tolist() and r['geometry']['region_retained']==(after==labels).tolist()
            delta=actual.astype(float)-x
            np.testing.assert_allclose(r['geometry']['energy'],np.square(delta).sum(),rtol=1e-12,atol=1e-10)
            assert r['geometry']['modified_tokens']==int(np.any(delta!=0,axis=1).sum())
            if c['operator']=='baseline':np.testing.assert_array_equal(actual,x)
            if cohort=='validation' and c['prefix']==16:
                with np.load(Path(reuse[sid]['baseline']).with_suffix('.npz')) as f:
                    n=min(c['width'],4);np.testing.assert_array_equal(x[-n:],f['before'][-n:])
            else:
                with np.load(path.parent/'baseline.npz') as f:np.testing.assert_array_equal(x,f['before'])
        else:
            assert r['geometry']['energy']==0
            if cohort=='validation' and c['prefix']==16:base=json.loads(Path(reuse[sid]['baseline']).read_text())
            else:base=json.loads((path.parent/'baseline.json').read_text())
            assert ids==base['generated_ids']
    assert seen=={(sid,c['name']) for sid in source for c in conditions}
    write_json(root/f'{batch}_audit.json',dict(complete=True,batch=batch,questions=len(source),records=len(seen),
        independently_rescored=True,independent_patch_max_abs_error=max_error,norm_fallbacks=fallbacks,
        source_receipt_sha256=sha(root/f'{batch}_SUCCESS.json'),code_sha256=sha(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--batch',required=True);a=p.parse_args();run(a.root,a.batch)
