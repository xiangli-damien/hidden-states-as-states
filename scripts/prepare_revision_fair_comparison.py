"""Verify fitted parameters and freeze both datasets for the fair comparison.

CPU only. No fitting, generation, target selection or score-based adaptation.
Copies compact immutable inputs; original fitting and locality studies stay intact.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from revision_common import config,freeze,provenance,sha,write_json,write_npz,status
from revision_factor_common import FairDecoders,conditions,key,budget


def load_arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def verified_model(folder,files,joint=False):
    receipt=json.loads((folder/'receipt.json').read_text())
    for name in ['model.npz','summary.json']+(['plan.json'] if joint else []):
        assert sha(folder/name)==receipt[name.split('.')[0]+'_sha256'],folder/name
        files.append(folder/name)
    files.append(folder/'receipt.json')
    return load_arrays(folder/'model.npz'),json.loads((folder/'summary.json').read_text())


def pack_models(cfg,files):
    root=Path(cfg['fit_root']); training=json.loads((root/'training_identity.json').read_text())
    plan=json.loads((root/'plan.json').read_text());staging=Path(plan['config']['staging'])
    for path in [root/'source_decoder.npz',root/'training_identity.json',staging/'empirical_means.npy']:
        assert sha(path)==plan['files'][str(path)],path;files.append(path)
    files.extend([root/'plan.json',root/'fit_summary.json',root/'joint/_SUCCESS.json'])
    old=load_arrays(root/'source_decoder.npz');means=np.load(staging/'empirical_means.npy')
    counts=np.array(training['counts']);assert len(counts)==cfg['k']
    np.testing.assert_array_equal(means.astype(np.float32),old['local_empirical_centers'])
    arrays={'gmm_centers':old['centers'],'old_anchor':old['local_empirical_centers'],
            'old_pca':old['local_empirical_basis'][:,:8], 'common_anchor':means,
            'fixed_weights':counts/counts.sum()}
    summaries={}
    for rank in cfg['ranks']:
        components=[];records=[]
        for k in range(cfg['k']):
            part,record=verified_model(root/f'rank_{rank}/component_{k:03d}',files)
            np.testing.assert_array_equal(part['common_anchor'],means[k])
            assert record['tokens']==int(counts[k]) and record['rank']==rank and record['component']==k
            np.testing.assert_allclose(part['means'][0],means[k],rtol=0,atol=1e-7)
            np.testing.assert_allclose(part['pca_basis']@part['pca_basis'].T,np.eye(rank),rtol=0,atol=2e-6)
            components.append(part);records.append(record)
        arrays[f'fixed_w_{rank}']=np.concatenate([r['loadings'] for r in components])
        arrays[f'fixed_noise_{rank}']=np.concatenate([r['noise'] for r in components])
        arrays[f'fixed_fit_means_{rank}']=np.concatenate([r['means'] for r in components])
        arrays[f'pca_{rank}']=np.stack([r['pca_basis'] for r in components])
        model,record=verified_model(root/f'joint/rank_{rank}',files,joint=True)
        for name in ['weights','means','loadings','noise']:arrays[f'joint_{name}_{rank}']=model[name]
        summaries[str(rank)]={'fixed_converged':sum(r['converged'] for r in records),
            'components':len(records),'joint_converged':record['converged'],
            'fixed_common_anchor_max_abs_change':float(np.max(np.abs(arrays[f'fixed_fit_means_{rank}']-means))),
            'training_assigned_component_mean_log_density':float(np.average([r['mean_log_likelihood'] for r in records],weights=counts)),
            'training_joint_mean_log_density':record['mean_training_log_likelihood'],
            'density_warning':'Fixed FA assigned-component score is not mixture likelihood; do not directly compare it to joint mixture likelihood.',
            'fixed_summary_paths':[str(root/f'rank_{rank}/component_{k:03d}/summary.json') for k in range(cfg['k'])],
            'joint_summary_path':str(root/f'joint/rank_{rank}/summary.json')}
    # Construct actual decoder caches now, catching malformed parameters on CPU.
    FairDecoders(arrays)
    return arrays,summaries,training


def prepare_dataset(name,source,root,cfg,training,files):
    plan=json.loads((source/'plan.json').read_text());ids=plan['selected_sample_ids']
    assert len(ids)==len(set(ids))==64
    assert not set(ids)&set(training['question_ids'])
    foundation=Path(plan['config']['foundation'])
    rows=[];tokens={}
    for marker in sorted((foundation/'prefixes').glob('shard_*/_SUCCESS.json')):
        frame=pd.read_parquet(marker.parent/'rows.parquet')
        keep=frame.sample_id.isin(ids)
        if not keep.any():continue
        saved=json.loads(marker.read_text())
        for filename in ['rows.parquet','tokens.json']:
            assert sha(marker.parent/filename)==saved['sha256'][filename],marker.parent/filename
        rows.append(frame.loc[keep].copy());files.extend([marker,marker.parent/'rows.parquet',marker.parent/'tokens.json'])
        for item in json.loads((marker.parent/'tokens.json').read_text()):
            if item['sample_id'] in ids:
                assert item['sample_id'] not in tokens
                tokens[item['sample_id']]=item
    frame=pd.concat(rows,ignore_index=True).set_index('sample_id',drop=False).loc[ids].copy()
    assert frame.index.is_unique and set(tokens)==set(ids)
    if name=='math':assert frame.split.value_counts().to_dict()=={'validation':32,'test':32}
    else:
        assert plan['target_fit_performed'] is False and set(frame.split)=={'test'}
        frame['split']='confirmation'
    folder=source/'decoders/p16_l14_tokens';receipt=json.loads((folder/'_SUCCESS.json').read_text())
    assert sha(folder/'pilot_activations.npz')==receipt['pilot_activations_sha256']
    files.extend([source/'plan.json',folder/'_SUCCESS.json',folder/'pilot_activations.npz'])
    captures=load_arrays(folder/'pilot_activations.npz')
    mapping=dict(zip(captures['sample_ids'].tolist(),captures['x']))
    selected=[];excluded=[];x=[]
    for sid in ids:
        t=tokens[sid]
        valid=len(t['response_ids'])>16 and not set(cfg['eos_ids']).intersection(t['response_ids'][:16])
        if not valid:
            excluded.append({'sample_id':sid,'reason':'prefix_unavailable'});continue
        if sid not in mapping:raise ValueError(f'Missing capture for available prefix: {sid}')
        assert mapping[sid].shape==(16,3584) and np.isfinite(mapping[sid]).all()
        selected.append(sid);x.append(mapping[sid])
    dest=root/'inputs'/name;dest.mkdir(parents=True,exist_ok=True)
    frame['dataset']=name
    frame.to_parquet(dest/'rows.parquet',index=False)
    write_json(dest/'tokens.json',[tokens[sid] for sid in ids])
    write_npz(dest/'captures.npz',sample_ids=np.array(selected),x=np.stack(x))
    outputs=[dest/f for f in ['rows.parquet','tokens.json','captures.npz']]
    files.extend(outputs)
    return {'dataset':name,'selected_ids':selected,'all_requested_ids':ids,'excluded':excluded,
            'splits':frame.split.value_counts().to_dict(),'files':{str(p):sha(p) for p in outputs}}


def run(cfg):
    assert cfg['ranks']==[8,4,16] and cfg['layer']==14 and cfg['width']==cfg['prefix_tokens']==16 and cfg['k']==64
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'prepare_SUCCESS.json').exists():
        plan=json.loads((root/'plan.json').read_text());assert plan['config']==cfg
        assert sha(root/'plan.json')==json.loads((root/'prepare_SUCCESS.json').read_text())['plan_sha256']
        for name,digest in plan['files'].items():assert sha(name)==digest,name
        print('Completed preparation verified; no refit or target selection.');return
    if (root/'plan.json').exists():raise ValueError('Incomplete preparation retained; inspect before resuming')
    started=time.monotonic();files=[Path(__file__),Path(__file__).with_name('revision_factor_common.py'),
        Path(__file__).with_name('revision_common.py'),Path(__file__).resolve().parents[1]/'docs/revision-fair-factors-plan-20260923.zh-CN.md']
    status(root,'prepare',state='running')
    arrays,summaries,training=pack_models(cfg,files)
    write_npz(root/'decoders.npz',**arrays);files.append(root/'decoders.npz')
    write_json(root/'fitting_summary.json',summaries);files.append(root/'fitting_summary.json')
    sources={'math':Path(cfg['math_locality']),'gsm8k':Path(cfg['confirmation'])}
    datasets={name:prepare_dataset(name,path,root,cfg,training,files) for name,path in sources.items()}
    methods=conditions();budgets={key(c):budget(c,cfg['k'],3584) for c in methods}
    write_json(root/'budgets.json',budgets);files.append(root/'budgets.json')
    plan=provenance(cfg,files)
    plan.update(datasets=datasets,conditions=methods,expected_conditions=sum(len(d['selected_ids']) for d in datasets.values())*len(methods),
        target_fit_performed=False,training_questions=training['questions'],training_tokens=training['tokens'],
        common_anchor_policy='Both fixed PCA and FA use exact saved float64 empirical anchors; fitted FA numerical mean drift (<1e-7) is saved and quantified, not used as a different anchor.',
        scope='Prepared decoder/input contracts only; no functional outcome measured here.')
    freeze(root/'plan.json',plan)
    write_json(root/'prepare_SUCCESS.json',{'complete':True,'plan_sha256':sha(root/'plan.json'),
        'questions':sum(len(d['selected_ids']) for d in datasets.values()),'conditions':plan['expected_conditions'],'seconds':time.monotonic()-started})
    status(root,'prepare',state='complete',questions=sum(len(d['selected_ids']) for d in datasets.values()),
           conditions=plan['expected_conditions'],seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=config(a.config)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=cfg['cpu_threads']):run(cfg)
