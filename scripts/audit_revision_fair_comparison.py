"""Recompute coverage, saved replacements, logits and reference losses on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from revision_common import sha,write_json
from revision_factor_common import FairDecoders
from revision_locality_common import geometry_metrics
from prepare_revision_fair_comparison import load_arrays
from measure_revision_fair_geometry import verify_inputs
from evaluate_revision_fair_comparison import tasks,inputs


def close_dict(actual,expected):
    assert set(actual)==set(expected)
    for k in actual:np.testing.assert_allclose(actual[k],expected[k],rtol=1e-10,atol=1e-6)


def run(root,smoke=False):
    input_plan=verify_inputs(root);cfg=input_plan['config'];dest=root/('smoke' if smoke else 'functional')
    plan=json.loads((dest/'plan.json').read_text());success=json.loads((dest/'_SUCCESS.json').read_text())
    assert sha(dest/'plan.json')==success['plan_sha256'] and plan['input_plan_sha256']==sha(root/'plan.json')
    for filename,digest in plan['files'].items():assert sha(filename)==digest,filename
    expected=tasks(input_plan,smoke);assert expected==plan['expected']
    assert {p.stem for p in (dest/'samples').glob('*.json')}=={t['name'] for t in expected}
    assert len(expected)==success['conditions']
    datasets=inputs(root,input_plan);models=FairDecoders(load_arrays(root/'decoders.npz'))
    baseline={};hashes={};largest_patch_error=0.
    for task in expected:
        path=dest/'samples'/(task['name']+'.json');r=json.loads(path.read_text());assert r['task']==task
        assert sha(path.with_suffix('.npz'))==r['arrays_sha256'];hashes[path.name]=sha(path)
        arrays=load_arrays(path.with_suffix('.npz'));logp,loss,z=arrays['logp'],arrays['reference_nll'],arrays['replacement']
        sid=task['sample_id'];frame,tokens,captures=datasets[task['dataset']];x=captures[sid]
        item=tokens[sid];prefix=item['prompt_ids']+item['response_ids'][:16];reference=item['response_ids'][16:]
        assert r['positions']==list(range(len(prefix)-16,len(prefix)))
        assert r['split']==str(frame.loc[sid,'split']) and r['patch_calls']==1 and r['capture_match_exact']
        expected_z,coding=models.replace(x,task['condition'])
        np.testing.assert_array_equal(z,torch.from_numpy(expected_z).to(torch.bfloat16).float().numpy())
        close_dict(r['geometry']['coding'],coding)
        close_dict(r['geometry']['ideal'],geometry_metrics(x,expected_z,models.arrays['gmm_centers']))
        actual=geometry_metrics(x,z,models.arrays['gmm_centers']);close_dict(r['geometry']['actual'],actual)
        energy=sum(actual['token_delta_energy'])
        np.testing.assert_allclose(r['actual_patch_energy'],energy,rtol=2e-6,atol=1e-5)
        largest_patch_error=max(largest_patch_error,abs(r['actual_patch_energy']-energy)/max(energy,1e-20))
        assert len(loss)==len(reference)==r['reference_tokens'] and np.isfinite(logp).all() and np.isfinite(loss).all() and (loss>=0).all()
        np.testing.assert_allclose(np.exp(logp.astype(float)).sum(),1,atol=2e-6)
        np.testing.assert_allclose(loss[0],-logp[reference[0]],rtol=0,atol=1e-12)
        for metric,value in [('nll',loss.mean()),('nll_sum',loss.sum()),('first_token_nll',loss[0]),('first16_nll',loss[:16].mean())]:
            np.testing.assert_allclose(r[metric],value,rtol=0,atol=1e-12)
        bkey=(task['dataset'],sid)
        if task['condition']['method']=='identity':
            baseline[bkey]=logp.astype(float)
            assert r['actual_patch_energy']==0 and r['next_token_kl']==0
        original=baseline[bkey]
        np.testing.assert_allclose(r['next_token_kl'],(np.exp(original)*(original-logp)).sum(),rtol=0,atol=1e-12)
        assert r['next_token_argmax_agreement']==bool(original.argmax()==logp.argmax())
    write_json(dest/'audit_inputs.json',hashes)
    write_json(dest/'audit.json',{'complete':True,'conditions':len(expected),'questions':len(baseline),
        'all_saved_bf16_patches_recomputed':True,'positions_reference_loss_logits_and_coverage_verified':True,
        'largest_patch_energy_relative_arithmetic_error':largest_patch_error,'plan_sha256':sha(dest/'plan.json'),
        'audit_code_sha256':sha(Path(__file__)),'scope':'Execution/data audit; numerical decoder formulas independently tested against dense conditional Gaussians.'})
    print(json.dumps({'complete':True,'conditions':len(expected),'questions':len(baseline)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    run(a.root,a.smoke)
