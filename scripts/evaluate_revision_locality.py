"""Fixed-protocol locality, radial-control and clean-state ablation pilot."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from evaluate_revision_patches import load_questions
from extract_revision_prefixes import load_model
from revision_common import config,freeze,provenance,sha,write_json,write_npz,status,OncePatch
from revision_locality_common import conditions,key,replacement,geometry_metrics


@torch.inference_mode()
def measure(model,prefix,reference,layer,positions,transform):
    if not reference:raise ValueError('Reference must be nonempty')
    patch=OncePatch(positions,len(prefix),transform)
    handle=model.model.layers[layer-1].register_forward_hook(patch)
    try:
        out=model(input_ids=torch.tensor([prefix],device=model.device),use_cache=True,logits_to_keep=1)
        logp=out.logits[0,-1].float().log_softmax(-1)
        first=logp.cpu().numpy();losses=[np.asarray([-float(logp[reference[0]])])]
        cache=out.past_key_values;del out,logp
        for start in range(0,len(reference)-1,128):
            end=min(start+128,len(reference)-1)
            out=model(input_ids=torch.tensor([reference[start:end]],device=model.device),past_key_values=cache,use_cache=True)
            loss=torch.nn.functional.cross_entropy(out.logits[0].float(),torch.tensor(reference[start+1:end+1],device=model.device),reduction='none')
            losses.append(loss.cpu().numpy());cache=out.past_key_values;del out,loss
        losses=np.concatenate(losses).astype(np.float64)
        assert len(losses)==len(reference) and patch.calls==1
        return {'nll_sum':float(losses.sum()),'nll':float(losses.mean()),
            'first16_nll':float(losses[:16].mean()),'first_token_nll':float(losses[0]),
            'reference_tokens':len(reference),'patch_calls':patch.calls,'actual_patch_energy':patch.energy},first,losses
    finally:handle.remove()


def plan_conditions(cfg,selected,tokens,eos):
    expected=[];excluded=[]
    for sid in selected:
        item=tokens[sid]
        for prefixn,layer,role in cfg['views']:
            if len(item['response_ids'])<=prefixn or eos.intersection(item['response_ids'][:prefixn]):
                excluded.append({'sample_id':sid,'prefix':prefixn,'layer':layer,'role':role,'reason':'prefix_unavailable'});continue
            prefix=item['prompt_ids']+item['response_ids'][:prefixn]
            eligible=item['question_positions'] if role=='question_tokens' else list(range(len(prefix)))
            if len(eligible)<16:
                excluded.append({'sample_id':sid,'prefix':prefixn,'layer':layer,'role':role,'reason':'window_unavailable'});continue
            primary=(prefixn,layer,role)==(16,14,'tokens')
            for width in [1,4,16] if primary else [16]:
                for condition in conditions(cfg,primary and width==16):
                    expected.append({'sample_id':sid,'prefix_tokens':prefixn,'layer':layer,'role':role,'width':width,
                        'positions':eligible[-width:],'condition':condition,
                        'name':f'{sid}_p{prefixn}_l{layer}_{role}_w{width}_{key(condition)}'})
    return expected,excluded


def run(cfg,smoke=False):
    root=Path(cfg['output']);dest=root/('smoke' if smoke else 'functional');dest.mkdir(parents=True,exist_ok=True)
    assert (root/'fit_SUCCESS.json').exists()
    fitplan=json.loads((root/'plan.json').read_text());assert fitplan['config']==cfg
    frame,tokens=load_questions({'output':cfg['foundation']});index=frame.set_index('sample_id')
    files=[Path(__file__),Path(__file__).with_name('revision_locality_common.py'),
        Path(__file__).with_name('revision_common.py'),Path(__file__).with_name('extract_revision_prefixes.py'),
        Path(__file__).with_name('evaluate_revision_patches.py'),root/'plan.json',root/'fit_SUCCESS.json']
    decoders={};captures={}
    for prefixn,layer,role in cfg['views']:
        name=f'p{prefixn}_l{layer}_{role}';folder=root/'decoders'/name
        receipt=json.loads((folder/'_SUCCESS.json').read_text())
        for filename,digest in [('decoder.npz',receipt['decoder_sha256']),('audit.json',receipt['audit_sha256']),('pilot_activations.npz',receipt['pilot_activations_sha256'])]:
            assert sha(folder/filename)==digest;files.append(folder/filename)
        with np.load(folder/'decoder.npz') as f:decoders[name]={k:f[k].copy() for k in f.files}
        with np.load(folder/'pilot_activations.npz') as f:captures[name]=dict(zip(f['sample_ids'].tolist(),f['x']))
    # Freeze plan before loading/running any model outcome.
    selected=fitplan['selected_sample_ids'][:1] if smoke else fitplan['selected_sample_ids']
    from transformers import GenerationConfig
    generation=GenerationConfig.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    eos=set(generation.eos_token_id if isinstance(generation.eos_token_id,list) else [generation.eos_token_id])
    expected,excluded=plan_conditions(cfg,selected,tokens,eos)
    plan=provenance(cfg,files);plan.update(expected=expected,exclusions=excluded,
        logits='raw forward distributions; no repetition penalty or sampling processor',
        reference='complete original reference suffix including EOS; also first-token and first16 NLL',
        smoke=smoke,primary='local_8 minus shared_8 at generated16/block14/width16, validation/test separate')
    freeze(dest/'plan.json',plan)
    model,_=load_model(cfg);baseline={};done=0;started=time.monotonic()
    status(root,'smoke' if smoke else 'functional',state='running',completed=0,expected=len(expected))
    for task in expected:
        sid=task['sample_id'];prefixn=task['prefix_tokens'];layer=task['layer'];role=task['role'];width=task['width']
        path=dest/'samples'/(task['name']+'.json');arraypath=path.with_suffix('.npz')
        bkey=(sid,prefixn,layer,role)
        if path.exists():
            record=json.loads(path.read_text());assert record['task']==task and sha(arraypath)==record['arrays_sha256']
            if task['condition']['method']=='identity':
                with np.load(arraypath) as f:baseline[bkey]=(f['logp'].copy(),f['reference_nll'].copy())
            done+=1;continue
        item=tokens[sid];prefix=item['prompt_ids']+item['response_ids'][:prefixn];reference=item['response_ids'][prefixn:]
        view=f'p{prefixn}_l{layer}_{role}';decoder=decoders[view]
        saved=captures[view][sid][-width:];context=f'{sid}/{prefixn}/{layer}/{role}/{width}'
        data={}
        def transform(h):
            x=h.float().cpu().numpy();np.testing.assert_array_equal(x,saved)
            z,_=replacement(x,decoder,task['condition'],context)
            out=torch.from_numpy(z).to(device=h.device,dtype=h.dtype)
            rounded=out.float().cpu().numpy()
            data['ideal']=geometry_metrics(x,z,decoder['centers'])
            data['actual']=geometry_metrics(x,rounded,decoder['centers'])
            return out
        metrics,logp,losses=measure(model,prefix,reference,layer,task['positions'],transform)
        if task['condition']['method']=='identity':
            if bkey in baseline:
                np.testing.assert_array_equal(logp,baseline[bkey][0]);np.testing.assert_array_equal(losses,baseline[bkey][1])
            baseline[bkey]=(logp.copy(),losses.copy())
        original=baseline[bkey][0].astype(np.float64)
        metrics.update(next_token_kl=float((np.exp(original)*(original-logp)).sum()),
            next_token_argmax_agreement=bool(original.argmax()==logp.argmax()))
        write_npz(arraypath,logp=logp,reference_nll=losses)
        record={'task':task,'split':str(index.loc[sid,'split']),**metrics,
            'geometry':data,'arrays_sha256':sha(arraypath)}
        write_json(path,record);done+=1
        if done%20==0:
            status(root,'smoke' if smoke else 'functional',state='running',completed=done,expected=len(expected),seconds=time.monotonic()-started)
    paths=list((dest/'samples').glob('*.json'))
    assert {p.stem for p in paths}=={t['name'] for t in expected}
    write_json(dest/'_SUCCESS.json',{'conditions':len(paths),'questions':len(selected),'plan_sha256':sha(dest/'plan.json'),'seconds':time.monotonic()-started})
    status(root,'smoke' if smoke else 'functional',state='complete',completed=len(paths),expected=len(expected),seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=config(a.config)
    try:run(cfg,a.smoke)
    except BaseException:
        status(cfg['output'],'smoke' if a.smoke else 'functional',state='failed',traceback=traceback.format_exc());raise
