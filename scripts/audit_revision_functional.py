"""Audit intervention execution contracts, optionally on a running snapshot.

This does not interpret partial NLL effects. A snapshot is never a completed
scientific result. Expected conditions come from the immutable stage plan and
the original token/validity manifests, not from observed successful outputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from revision_common import sha,write_json


def distribution(values):
    x=np.asarray(values,dtype=float)
    return {'n':len(x),'min':float(x.min()),'p05':float(np.quantile(x,.05)),
            'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),
            'max':float(x.max())} if len(x) else {'n':0}


def run(root,partial=False):
    dest=root/'functional';success=dest/'_SUCCESS.json'
    completed=json.loads(success.read_text()) if success.exists() else None
    if completed is None and not partial:raise ValueError('Completed functional stage required; use --partial for execution-only audit')
    plan=json.loads((dest/'plan.json').read_text());cfg=plan['config'];selected=set(plan['sample_ids'])
    tokens={};available={};splits={}
    for marker in (root/'prefixes').glob('shard_*/_SUCCESS.json'):
        rows=pd.read_parquet(marker.parent/'rows.parquet',columns=['sample_id','split'])
        entries=json.loads((marker.parent/'tokens.json').read_text())
        np.testing.assert_array_equal(rows.sample_id,[r['sample_id'] for r in entries])
        for row in entries:
            if row['sample_id'] in selected:tokens[row['sample_id']]=row
        for prefix in cfg['functional_prefixes']:
            with np.load(marker.parent/f'prefix_{prefix}.npz') as z:
                for i,sid in enumerate(rows.sample_id):
                    if sid in selected:
                        available[sid,prefix]=(bool(z['valid'][i]),bool(z['question_valid'][i]))
        splits.update(dict(zip(rows.sample_id,rows.split)))
    assert set(tokens)==selected
    expected={}
    for sid,row in tokens.items():
        for prefix in cfg['functional_prefixes']:
            valid,qvalid=available[sid,prefix]
            if not valid:continue
            length=len(row['prompt_ids'])+prefix
            for role in ('tokens','question_tokens') if prefix==0 else ('tokens',):
                if role=='question_tokens' and not qvalid:continue
                eligible=row['question_positions'] if role=='question_tokens' else list(range(length))
                if len(eligible)<16:continue
                for layer in plan['layers']:
                    for width in cfg['functional_widths']:
                        for method in plan['methods']:
                            name=f'{sid}_p{prefix}_l{layer}_{role}_w{width}_{method}.json'
                            expected[name]={'sample_id':sid,'split':splits[sid],'prefix_tokens':prefix,
                                'layer':layer,'role':role,'width':width,'method':method,'positions':eligible[-width:],
                                'reference_tokens':len(row['response_ids'])-prefix}
    paths=list((dest/'samples').glob('*.json'));records=[];lookup={};identities={};logits={}
    for path in paths:
        r=json.loads(path.read_text());assert path.name in expected
        for k,v in expected[path.name].items():
            if r[k]!=v:raise AssertionError((path.name,k,r[k],v))
        assert r['patch_calls']==1 and r['reference_tokens']>0 and r['actual_patch_energy']>=0
        assert all(np.isfinite(r[k]) for k in ('nll_sum','nll','actual_patch_energy','next_token_kl'))
        np.testing.assert_allclose(r['nll_sum']/r['reference_tokens'],r['nll'],rtol=0,atol=1e-12)
        key=tuple(r[k] for k in ('sample_id','prefix_tokens','layer','role','width','method'));lookup[key]=r
        if r['method']=='identity':
            assert r['actual_patch_energy']==0 and r['next_token_kl']==0 and r['next_token_argmax_agreement']
            key=(r['sample_id'],r['prefix_tokens'])
            with np.load(path.with_suffix('.npz')) as z:logp=z['logp'].copy()
            if key in logits:
                np.testing.assert_array_equal(logits[key],logp)
                np.testing.assert_allclose(identities[key]['nll_sum'],r['nll_sum'],rtol=0,atol=1e-10)
            else:logits[key]=logp;identities[key]=r
        records.append(r)
    if completed is not None:
        assert {p.name for p in paths}==set(expected)
        assert len(paths)==completed['conditions']==completed['expected_conditions']
    architecture_null=0
    energy={'centroid_energy1_vs_width1':[],'random_energy1_vs_centroid_energy1':[],'matched_random_vs_centroid':[]}
    for r in records:
        # Qwen block28 past outputs cannot influence a future token: no later
        # block consumes them and layer28 KV was formed before its output hook.
        row=tokens[r['sample_id']];length=len(row['prompt_ids'])+r['prefix_tokens']
        if r['layer']==28 and max(r['positions'])<length-1:
            base=identities[r['sample_id'],r['prefix_tokens']]
            np.testing.assert_allclose(r['nll_sum'],base['nll_sum'],rtol=0,atol=1e-10)
            assert abs(r['next_token_kl'])<=1e-7 and r['next_token_argmax_agreement']
            architecture_null+=1
        prefix=tuple(r[k] for k in ('sample_id','prefix_tokens','layer','role'))
        comparisons=[]
        if r['method']=='centroid_energy1':comparisons.append(('centroid_energy1_vs_width1',prefix+(1,'centroid_energy1')))
        if r['method']=='matched_random_energy1':comparisons.append(('random_energy1_vs_centroid_energy1',prefix+(r['width'],'centroid_energy1')))
        if r['method']=='matched_random':comparisons.append(('matched_random_vs_centroid',prefix+(r['width'],'centroid')))
        for name,key in comparisons:
            if key in lookup and lookup[key]['actual_patch_energy']>0:
                energy[name].append(r['actual_patch_energy']/lookup[key]['actual_patch_energy'])
    result={'stage_complete':completed is not None,'snapshot_conditions':len(paths),'expected_conditions':len(expected),
        'selected_questions':len(selected),'identity_prefixes':len(identities),'final_block_past_position_null_conditions':architecture_null,
        'position_reference_length_single_hook_identity_and_null_checks_passed':True,
        'actual_bf16_energy_ratios':{k:distribution(v) for k,v in energy.items()},
        'energy_caution':'Ideal equal energy is implemented before bf16 rounding. These are actual post-rounding energies, not assumed exact equality.',
        'scope':'Execution integrity only; no partial scientific effect claim',
        'plan_sha256':sha(dest/'plan.json'),'audit_code_sha256':sha(Path(__file__))}
    write_json(dest/('execution_audit.json' if completed is not None else 'execution_audit_partial.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();run(a.root,a.partial)
