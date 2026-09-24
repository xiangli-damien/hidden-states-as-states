"""Independently check generated text/labels, recorded patch arithmetic and inputs."""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer
from openact_eval.evaluators.registry import auto_select_evaluator
from revision_common import sha,write_json


def run(root,stage):
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    success=json.loads((root/f'{stage}_SUCCESS.json').read_text())
    assert success['complete'] and success['plan_sha256']==sha(root/'plan.json')
    for field,name in [('execution_plan_sha256','execution_plan.json'),('generation_config_sha256','generation_config.json')]:
        assert success[field]==sha(root/name)
    for p,h in json.loads((root/'execution_plan.json').read_text())['files'].items():assert sha(p)==h
    tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
    ev=auto_select_evaluator('math');inputs={r['sample_id']:r for r in json.loads((root/'inputs.json').read_text())}
    generation=json.loads((root/'generation_config.json').read_text())['resolved']
    assert generation['do_sample'] is False and generation['repetition_penalty']==1.
    eos=generation['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
    with np.load(cfg['decoder']) as f:
        centers=f['centers'].astype(float);local=f['local_basis'][:,:8].astype(float);shared=f['shared_basis'][:8].astype(float)
    support=json.loads((root/'support.json').read_text());counts=np.asarray(support['question_count'])
    probabilities=np.asarray(support['shrunk_correctness']);seen=set();max_error=0.;identity=0
    expected_cases=[r['sample_id'] for r in inputs.values() if r['split']==('test' if stage=='test' else 'validation')]
    if stage=='smoke':expected_cases=expected_cases[:cfg['smoke_questions']]
    if stage=='test':expected_cases=expected_cases[:json.loads((root/'selection.json').read_text())['test_questions']]
    for name,digest in success['records'].items():
        path=root/name;assert sha(path)==digest
        r=json.loads(path.read_text());sid=r['sample_id'];source=inputs[sid];cond=r['condition'];family=cond['family']
        assert r['split']==source['split'] and r['ground_truth']==source['ground_truth']
        assert (sid,cond['name']) not in seen;seen.add((sid,cond['name']))
        for rel,h in r['files'].items():assert sha(path.parent/rel)==h
        prefix=json.loads((path.parent/'prefix.json').read_text())
        assert prefix['prompt_ids']==source['prompt_ids'] and prefix['plan_sha256']==sha(root/'plan.json')
        ids=r['generated_ids'];p=prefix['generated_prefix']
        assert ids[:len(p)]==p and 0<len(ids)<=cfg['max_new_tokens'] and not eos.intersection(ids[:-1])
        assert r['length']==len(ids)
        reason='eos' if ids[-1] in eos else 'length';assert r['finish_reason']==reason
        if reason=='length':assert len(ids)==cfg['max_new_tokens']
        active=len(p)==16 and not eos.intersection(p);assert r['prefix_active']==bool(active)
        expected_pos=list(range(len(source['prompt_ids'])+12,len(source['prompt_ids'])+16)) if active else []
        assert r['positions']==expected_pos and r['hook_calls']==int(active)
        assert r['fresh_cache'] and r['hook_removed_after_prefill']
        text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False);assert text==r['response_text']
        score=ev.evaluate_sample(SimpleNamespace(sample_idx=source['source_sample_idx'],response_text=text,
            ground_truth=source['ground_truth'],meta={'sample_id':sid}))
        assert not score.error and bool(score.is_correct)==r['correct']
        assert score.extracted_answer==r['parsed_answer'] and score.normalized_answer==r['normalized_answer']
        assert bool(score.meta['parse_failed'])==r['parse_failed']
        baseline=json.loads((path.parent/'baseline.json').read_text())
        if active:
            with np.load(path.with_suffix('.npz')) as f:x=f['before'].astype(float);z=f['ideal'];actual=f['actual']
            with np.load(path.parent/'baseline.npz') as f:np.testing.assert_array_equal(x,f['before'])
            assert x.shape==(4,3584) and np.isfinite(actual).all()
            distance=((x[:,None]-centers[None])**2).sum(-1)
            ranked=np.argsort(distance,axis=1,kind='stable');original=ranked[:,0];target=original.copy()
            mask=np.ones(4,dtype=bool)
            if family in ('baseline','identity'):mask[:]=False
            elif cond.get('target_family',family)=='c2':
                mask[:]=False
                for i in range(4):
                    for k in ranked[i,:3]:
                        if counts[k]>=20 and probabilities[k]-probabilities[original[i]]>=.10:
                            target[i]=k;mask[i]=True;break
            expected=x.copy()
            for i in np.flatnonzero(mask):
                b=shared if family=='shared' else local[target[i]]
                residual=x[i]-centers[target[i]]
                projected=centers[target[i]]+b.T@(b@residual)
                expected[i]=x[i]+cond['alpha']*(projected-x[i])
            if family=='random':
                seed=int(hashlib.sha256(f'completion-random/{sid}/{cond["seed"]}'.encode()).hexdigest()[:8],16)
                delta=expected-x
                parallel=x*((x*delta).sum(1)/np.square(x).sum(1))[:,None]
                rnd=np.random.default_rng(seed).standard_normal(x.shape)
                rnd-=x*((x*rnd).sum(1)/np.square(x).sum(1))[:,None]
                expected=x+parallel+rnd*(np.linalg.norm(delta-parallel,axis=1)/np.linalg.norm(rnd,axis=1))[:,None]
                expected[~mask]=x[~mask]
            np.testing.assert_allclose(z,expected,rtol=1e-10,atol=1e-9)
            # The saved ideal is the exact input to the dtype conversion. This
            # conversion is independently replayed, without redoing GPU algebra.
            np.testing.assert_array_equal(actual,torch.tensor(z).to(torch.bfloat16).float().numpy())
            max_error=max(max_error,float(np.max(np.abs(z-expected))))
            meta=r['geometry']['regions'];assert meta=={'current':original.tolist(),'target':target.tolist(),'mask':mask.tolist()}
            delta=actual.astype(float)-x
            np.testing.assert_allclose(r['geometry']['energy'],np.square(delta).sum(),rtol=1e-12)
            assert r['geometry']['modified_tokens']==int(np.any(delta!=0,axis=1).sum())
            np.testing.assert_array_equal(actual[~mask],x[~mask])
        else:assert ids==baseline['generated_ids'] and r['geometry']['energy']==0
        if family=='identity':assert ids==baseline['generated_ids'];identity+=1
    assert seen=={(sid,c['name']) for sid in expected_cases for c in success['conditions']}
    result={'complete':True,'phase':stage,'questions':len(expected_cases),'conditions':len(seen),
        'identity_full_generations_exact':identity,'independent_patch_max_abs_error':max_error,
        'labels_independently_rescored':True,'scope':'Artifact arithmetic/IDs/labels; fresh-cache execution validated separately with real tinyQwen check',
        'stage_receipt_sha256':sha(root/f'{stage}_SUCCESS.json'),'audit_code_sha256':sha(Path(__file__))}
    write_json(root/f'{stage}_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--stage',choices=['smoke','validation','test'],required=True);a=ap.parse_args();run(a.root,a.stage)
