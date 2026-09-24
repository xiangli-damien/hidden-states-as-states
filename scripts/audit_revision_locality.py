"""Independent execution audit from frozen inputs, per-token losses and logits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GenerationConfig
from evaluate_revision_patches import load_questions
from evaluate_revision_locality import plan_conditions
from revision_common import sha,write_json
from revision_locality_common import replacement,geometry_metrics


def distribution(values):
    a=np.asarray(values,float)
    return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())} if len(a) else {'n':0}


def assert_geometry(got,expected):
    assert set(got)==set(expected)
    for field in got:np.testing.assert_allclose(got[field],expected[field],rtol=1e-10,atol=1e-6)


def run(root,smoke=False,partial=False):
    dest=root/('smoke' if smoke else 'functional');plan=json.loads((dest/'plan.json').read_text());cfg=plan['config']
    for path,digest in plan['files'].items():assert sha(path)==digest,path
    fitplan=json.loads((root/'plan.json').read_text());assert fitplan['config']==cfg
    for path,digest in fitplan['files'].items():assert sha(path)==digest,path
    selected=fitplan['selected_sample_ids'][:1] if smoke else fitplan['selected_sample_ids']
    assert not set(selected)&set(fitplan['excluded_prior_intervention_ids'])
    frame,tokens=load_questions({'output':cfg['foundation']});splits=frame.set_index('sample_id').split.to_dict()
    gen=GenerationConfig.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    eos=set(gen.eos_token_id if isinstance(gen.eos_token_id,list) else [gen.eos_token_id])
    tasks,exclusions=plan_conditions(cfg,selected,tokens,eos)
    assert tasks==plan['expected'] and exclusions==plan['exclusions']
    expected={t['name']:t for t in tasks};paths=sorted((dest/'samples').glob('*.json'))
    if not partial:
        success=json.loads((dest/'_SUCCESS.json').read_text())
        assert success['plan_sha256']==sha(dest/'plan.json')
        assert len(paths)==success['conditions']==len(tasks) and {p.stem for p in paths}==set(expected)
    decoders={};captures={};identities={};hashes={};control_errors={};records=[]
    for p,l,role in cfg['views']:
        name=f'p{p}_l{l}_{role}';folder=root/'decoders'/name
        with np.load(folder/'decoder.npz') as f:decoders[name]={k:f[k].copy() for k in f.files}
        with np.load(folder/'pilot_activations.npz') as f:captures[name]=dict(zip(f['sample_ids'].tolist(),f['x']))
    for path in paths:
        r=json.loads(path.read_text());task=r['task'];sid=task['sample_id'];assert task==expected[path.stem]
        assert sha(path.with_suffix('.npz'))==r['arrays_sha256'];hashes[path.name]=sha(path)
        if task['condition']['method']=='identity':
            with np.load(path.with_suffix('.npz')) as f:logp,losses=f['logp'].copy(),f['reference_nll'].copy()
            ikey=(sid,task['prefix_tokens'])
            if ikey in identities:
                np.testing.assert_array_equal(logp,identities[ikey][0]);np.testing.assert_array_equal(losses,identities[ikey][1])
            identities[ikey]=(logp,losses)
        records.append(r)
    for r in records:
        task=r['task'];sid=task['sample_id'];prefix=task['prefix_tokens'];name=f'p{prefix}_l{task["layer"]}_{task["role"]}'
        c=task['condition'];x=captures[name][sid][-task['width']:];decoder=decoders[name]
        context=f'{sid}/{prefix}/{task["layer"]}/{task["role"]}/{task["width"]}'
        z,_=replacement(x,decoder,c,context)
        rounded=torch.from_numpy(z).to(torch.bfloat16).float().numpy()
        ideal=geometry_metrics(x,z,decoder['centers']);actual=geometry_metrics(x,rounded,decoder['centers'])
        assert_geometry(r['geometry']['ideal'],ideal);assert_geometry(r['geometry']['actual'],actual)
        np.testing.assert_allclose(sum(actual['token_delta_energy']),r['actual_patch_energy'],rtol=2e-6,atol=1e-5)
        assert r['patch_calls']==1 and r['split']==splits[sid]
        with np.load(dest/'samples'/(task['name']+'.npz')) as f:logp,losses=f['logp'],f['reference_nll']
        assert len(losses)==len(tokens[sid]['response_ids'])-prefix==r['reference_tokens']
        assert np.isfinite(logp).all() and np.isfinite(losses).all() and (losses>=0).all()
        np.testing.assert_allclose(np.exp(logp.astype(np.float64)).sum(),1,atol=2e-6)
        for field,value in [('nll_sum',losses.sum()),('nll',losses.mean()),('first_token_nll',losses[0]),('first16_nll',losses[:16].mean())]:
            np.testing.assert_allclose(r[field],value,rtol=0,atol=1e-12)
        if (sid,prefix) not in identities:
            assert partial;continue
        baseline=identities[sid,prefix][0].astype(np.float64)
        kl=float((np.exp(baseline)*(baseline-logp)).sum())
        np.testing.assert_allclose(r['next_token_kl'],kl,rtol=0,atol=1e-12)
        assert r['next_token_argmax_agreement']==bool(baseline.argmax()==logp.argmax())
        if c['method']=='identity':assert r['actual_patch_energy']==0 and kl==0
        target=None
        if c['method'] in ('centroid_random','centroid_radial_random','centroid_gram_random'):
            target={'method':'centroid'}
        elif c['method']=='remove_local_radial_random':
            target={'method':'remove_local','rank':c['rank'],'alpha':c['alpha']}
        elif c['method']=='complement_energy_matched':
            target={'method':'remove_local','rank':c['rank'],'alpha':c['alpha']}
        if target:
            tz,_=replacement(x,decoder,target,context)
            ti=geometry_metrics(x,tz,decoder['centers'])
            ta=geometry_metrics(x,torch.from_numpy(tz).to(torch.bfloat16).float().numpy(),decoder['centers'])
            np.testing.assert_allclose(ideal['token_delta_energy'],ti['token_delta_energy'],rtol=1e-10,atol=1e-6)
            if 'radial_random' in c['method']:
                for field in ['token_radial_dot','token_final_norm']:
                    np.testing.assert_allclose(ideal[field],ti[field],rtol=1e-10,atol=1e-6)
            if c['method']=='centroid_gram_random':
                np.testing.assert_allclose(ideal['error_gram'],ti['error_gram'],rtol=1e-10,atol=1e-6)
            energy=np.asarray(ta['token_delta_energy']);observed=np.asarray(actual['token_delta_energy'])
            errors=control_errors.setdefault(c['method'],{'relative_token_energy_error':[],
                'normalized_radial_dot_error':[],'relative_final_norm_error':[],'relative_gram_error':[]})
            errors['relative_token_energy_error'].extend((np.abs(observed-energy)/np.maximum(energy,1e-20)).tolist())
            normalizer=np.asarray(ta['token_original_norm'])*np.sqrt(np.maximum(energy,1e-20))
            errors['normalized_radial_dot_error'].extend((np.abs(np.asarray(actual['token_radial_dot'])-ta['token_radial_dot'])/np.maximum(normalizer,1e-20)).tolist())
            errors['relative_final_norm_error'].extend((np.abs(np.asarray(actual['token_final_norm'])-ta['token_final_norm'])/np.maximum(ta['token_final_norm'],1e-20)).tolist())
            errors['relative_gram_error'].append(float(np.linalg.norm(np.asarray(actual['error_gram'])-ta['error_gram'])/max(np.linalg.norm(ta['error_gram']),1e-20)))
    result={'complete':not partial,'smoke':smoke,'conditions':len(records),'expected':len(expected),
        'questions':len({r['task']['sample_id'] for r in records}),
        'all_conditions_input_SHA_positions_exact_replacement_reference_losses_logits_and_controls_verified':True,
        'control_actual_bf16_deviations':{k:{m:distribution(v) for m,v in values.items()} for k,values in control_errors.items()},
        'notes':'Only energy is matched for ordinary random/complement; Gram control also preserves ideal error Gram; radial control also preserves ideal radial dot/final norm. Other deviations are expected.',
        'plan_sha256':sha(dest/'plan.json'),'audit_code_sha256':sha(Path(__file__))}
    write_json(dest/('audit_partial.json' if partial else 'audit.json'),result)
    write_json(dest/('audit_inputs_partial.json' if partial else 'audit_inputs.json'),hashes)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--partial',action='store_true');a=p.parse_args()
    run(a.root,a.smoke,a.partial)
