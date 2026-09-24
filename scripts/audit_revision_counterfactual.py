"""Check complete controlled-task outputs; execution success is not capability."""
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from revision_common import sha,write_json
from revision_counterfactual_gate import ROOT,encode,parse,summarize


def run():
    plan=json.loads((ROOT/'plan.json').read_text());cfg=plan['config']
    for filename,digest in plan['files'].items():assert sha(filename)==digest
    marker=json.loads((ROOT/'_SUCCESS.json').read_text())
    assert sha(ROOT/'summary.json')==marker['summary_sha256']
    expected={c['id']:c for c in cfg['cases']};paths=sorted((ROOT/'cases').glob('*.json'))
    assert {p.stem for p in paths}==set(expected) and len(paths)==80==marker['cases']
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    conditions={f'{m}_l{l}_{s}' for m in ['donor','matched_random'] for l in [7,14] for s in ['w1','w4','w16','memory_region']}
    records=[];hashes={};ratios=[];donor_mismatch=[];errors=[]
    for path in paths:
        r=json.loads(path.read_text());assert r['case']==expected[path.stem];c=r['case'];rc,dc=c['recipient'],c['donor']
        edited={**rc,'a':dc['a'],'b':dc['b'],'u':dc['u']}
        for item,idkey,textkey in [(rc,'input_ids','prompt'),(dc,'donor_input_ids','donor_prompt'),(edited,'edited_input_ids','edited_prompt')]:
            ids,pos,text=encode(tokenizer,item,c['mode']);assert ids==r[idkey] and pos==r['positions'] and text==r[textkey]
            _,region,_=encode(tokenizer,item,c['mode'],'memory_region');assert region==r['memory_region_positions']
        assert r['positions']==r['memory_region_positions'][-16:]
        assert len(r['patches'])==16 and {p['condition'] for p in r['patches']}==conditions
        outputs=[r[k] for k in ['baseline','edited_baseline','embedding_control','final_block_null','identity']]+r['patches']
        for out in outputs:
            assert out['text']==tokenizer.decode(out['generated_ids'],skip_special_tokens=True)
            digit,tag=parse(out['text']);assert (digit,tag)==(out['digit'],out['tag']) and out['parse_failed']==(digit is None)
            assert 0<len(out['generated_ids'])<=16
        assert r['embedding_control']['generated_ids']==r['edited_baseline']['generated_ids']
        assert r['identity']['generated_ids']==r['final_block_null']['generated_ids']==r['baseline']['generated_ids']
        assert r['baseline_correct']==(r['baseline']['digit']==(rc['u']+rc['c'])%10 and r['baseline']['tag']==rc['tag'])
        assert r['edited_correct']==(r['edited_baseline']['digit']==c['cf_digit'] and r['edited_baseline']['tag']==rc['tag'])
        activation=ROOT/'activations'/(path.stem+'.npz');assert sha(activation)==r['activation_sha256']
        lookup={p['condition']:p for p in r['patches']}
        with np.load(activation) as z:
            for layer in cfg['layers']:
                a,b=z[f'recipient_{layer}'],z[f'donor_{layer}']
                assert a.shape==b.shape==(len(r['memory_region_positions']),3584)
                assert np.isfinite(a).all() and np.isfinite(b).all()
                if layer==28:continue
                for scope in ['w1','w4','w16','memory_region']:
                    da=b-a;delta=da if scope=='memory_region' else da[-int(scope[1:]):]
                    energy=float(np.square(delta).sum());actual=lookup[f'donor_l{layer}_{scope}']['actual_patch_energy']
                    donor_mismatch.append(abs(actual-energy)/max(energy,1e-20))
                    random=lookup[f'matched_random_l{layer}_{scope}']['actual_patch_energy']
                    if actual>0:ratios.append(random/actual)
                    else:assert random==0
        if not r['baseline_correct']:
            errors.append({'case':c['id'],'mode':c['mode'],'pair_type':c['pair_type'],'u':rc['u'],'c':rc['c'],
                'gold':f'{(rc["u"]+rc["c"])%10}|{rc["tag"]}','actual':r['baseline']['text'],
                'returned_u_without_offset':r['baseline']['digit']==rc['u']})
        records.append(r);hashes[str(path)]=sha(path);hashes[str(activation)]=sha(activation)
    # Stable summation is integer counts; all input-order variants must agree.
    summary=summarize(records);assert summary==json.loads((ROOT/'summary.json').read_text())
    result={'cases':80,'conditions_per_case':16,'summary_recomputed':True,
        'prompt_decode_controls_labels_activation_SHA_and_shapes_verified':True,
        'max_donor_energy_relative_difference':max(donor_mismatch),
        'random_donor_actual_energy_ratio':{'min':float(min(ratios)),'median':float(np.median(ratios)),'max':float(max(ratios))},
        'capability':summary['capability'],'state_only_stage_authorized_by_gate':False,
        'reason':'Both task modes fail the predeclared90% original/edited capability requirement. No state-only inference.',
        'plan_sha256':sha(ROOT/'plan.json'),'audit_code_sha256':sha(Path(__file__))}
    assert not any(c['passed'] for c in summary['capability'])
    write_json(ROOT/'audit.json',result);write_json(ROOT/'audit_input_hashes.json',hashes);write_json(ROOT/'capability_errors.json',errors)
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
