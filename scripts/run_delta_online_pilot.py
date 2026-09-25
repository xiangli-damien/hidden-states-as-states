"""Prepare and execute the frozen five-layer Qwen MATH delta pilot."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from revision_common import config,freeze,provenance,sha,write_json,write_npz,status
from delta_online_common import FUNCTIONAL_METHODS,posterior_region,replacement


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    for f,h in plan['files'].items():
        if sha(f)!=h:raise ValueError('Frozen source changed: '+f)
    return plan


def prepare(cfg):
    import joblib,pandas as pd,zarr
    from threadpoolctl import threadpool_limits
    from fit_revision_geometry import basis
    from revision_locality_common import derangement
    r=Path(cfg['output']);r.mkdir(parents=True,exist_ok=True)
    if (r/'plan.json').exists():return verify(r)
    src=Path(cfg['source']);cache=Path(cfg['cache']);oa=Path(cfg['openact'])
    rows=pd.read_parquet(src/'rows.parquet');splits=pd.read_parquet(src/'splits.parquet')
    np.testing.assert_array_equal(rows.sample_id,splits.sample_id);assert rows.question_group.is_unique and len(rows)==5000
    rows['split']=splits.split;rows.to_parquet(r/'rows.parquet',index=False)
    files=[src/'rows.parquet',src/'splits.parquet',src/'protocol.json',src/'PREPARED.json',src/'ASSIGNED.json']
    cards={};gm={}
    for rep in ['delta','state']:
        cards[rep]=[]
        for l in cfg['layers']:
            folder=src/'fits'/f'token_{rep}_L{l:02d}'
            selection=json.loads((folder/'selection.json').read_text());model=joblib.load(folder/'selected.joblib')
            assert model.converged_ and model.n_components==selection['k']==32
            cards[rep].append(model.n_components);gm[rep,l]=model
            files.extend([folder/'selection.json',folder/'selected.joblib'])
    nmax=max(cfg['prefixes']);seq={rep:np.full((len(rows),nmax,len(cfg['layers'])),-1,np.int16) for rep in cards}
    kinds=np.full((len(rows),nmax),-1,np.int8);lengths=np.zeros(len(rows),int);tokens={};source_records=[]
    for rec in json.loads((src/'PREPARED.json').read_text())['shards']:
        name=rec['shard'];ap=src/'assignments'/(name+'.npz');receipt=src/'assignments'/(name+'.json')
        assert sha(ap)==json.loads(receipt.read_text())['sha256'];files.extend([ap,receipt])
        layout_path=cache/'shards'/name/'layout.npz';files.append(layout_path)
        source=oa/name;manifest=source/'manifest.json';data=source/'data.parquet';files.extend([manifest,data,source/'_COPY_VERIFIED.json',source/'tensors.zarr/.zmetadata'])
        md=json.loads(manifest.read_text());special=md['custom']['special_token_ids'];frame=pd.read_parquet(data)
        z=zarr.open_consolidated(str(source/'tensors.zarr'),mode='r');allids=np.asarray(z['tokens/ids']);ptr=np.asarray(z['tokens/sample_ptr'])
        with np.load(ap) as a,np.load(layout_path) as layout:
            np.testing.assert_array_equal(ptr,a['token_ptr']);np.testing.assert_array_equal(layout['question_index'],a['question_index'])
            for j,i in enumerate(a['question_index']):
                i=int(i);row=frame.iloc[j];assert row.sample_id==rows.sample_id.iloc[i]
                lo,hi=map(int,ptr[j:j+2]);nn=min(nmax,hi-lo);lengths[i]=hi-lo
                for rep in cards:seq[rep][i,:nn]=a['selected_token_'+rep][lo:lo+nn]
                kinds[i,:nn]=layout['token_kind'][lo:lo+nn]
                tokens[row.sample_id]={'sample_id':row.sample_id,'row_index':i,'shard':name,'sample_idx':j,
                    'prompt_ids':json.loads(row.prompt_token_ids_json),'response_ids':allids[lo:hi].tolist()}
        source_records.append({'shard':name,'rows':len(frame),'special_ids':special})
    np.testing.assert_array_equal(lengths,rows.n_tokens.to_numpy());assert np.all(lengths>0)
    write_npz(r/'prefix_sequences.npz',**seq,kinds=kinds,lengths=lengths)
    write_json(r/'tokens.json',tokens);write_json(r/'source_records.json',source_records)
    # Local axes are fitted only to the same original training-token pool.
    fit_audit={}
    for rep in ['delta','state']:
        model=gm[rep,cfg['functional_layer']];path=cache/'views'/f'token_{rep}_L{cfg["functional_layer"]:02d}.npy'
        sel=json.loads((src/'fits'/f'token_{rep}_L{cfg["functional_layer"]:02d}'/'selection.json').read_text())
        assert sha(path)==sel['feature_sha256'];files.append(path)
        x=np.load(path,mmap_mode='r');train=np.repeat(rows.split.eq('train').to_numpy(),4);x=np.asarray(x[train],dtype=np.float64)
        assert len(x)==12044
        d={'centers':model.means_,'variances':model.covariances_,'weights':model.weights_,'train_mean':x.mean(0)}
        with threadpool_limits(cfg['threads']):
            codes=model.predict(x);np.testing.assert_array_equal(codes,posterior_region(x,d))
            d['shared_basis']=basis(x-d['centers'][codes],cfg['rank'],np.zeros(x.shape[1]),cfg['seed'])
            d['local_basis']=np.stack([basis(x[codes==j],cfg['rank'],d['centers'][j],cfg['seed']+j) for j in range(model.n_components)])
        counts=np.bincount(codes,minlength=model.n_components);d['local_counts']=counts
        d['permutation']=derangement(model.n_components,cfg['seed'])
        write_npz(r/(rep+'_decoder.npz'),**d)
        fit_audit[rep]={'training_vectors':len(x),'local_counts':counts.tolist(),'effective_ranks':np.minimum(8,np.maximum(0,counts-1)).tolist(),
                        'posterior_matches_sklearn':True,'k':32,'gmm_refit':False}
    test_ids=rows.loc[rows.split.eq('test'),'sample_id'].tolist()
    chosen=sorted(test_ids,key=lambda s:hashlib.sha256(f'{cfg["seed"]}:{s}'.encode()).hexdigest())[:cfg['functional_questions']]
    tasks=[];excluded=[]
    for sid in chosen:
        c=tokens[sid]
        for p in cfg['functional_prefixes']:
            if len(c['response_ids'])<=p:excluded.append({'sample_id':sid,'prefix':p,'reason':'already_ended'});continue
            for width in cfg['functional_widths']:
                for method in FUNCTIONAL_METHODS:tasks.append({'sample_id':sid,'prefix':p,'width':width,'method':method,'key':f'{sid}_p{p}_w{width}_{method}'})
    write_json(r/'functional_tasks.json',{'tasks':tasks,'excluded':excluded,'selected_ids':chosen});write_json(r/'fit_audit.json',fit_audit)
    files.extend(r/f for f in ['rows.parquet','prefix_sequences.npz','tokens.json','source_records.json','delta_decoder.npz','state_decoder.npz','functional_tasks.json','fit_audit.json'])
    files.extend(Path(__file__).with_name(f) for f in ['run_delta_online_pilot.py','delta_online_common.py','evaluate_delta_online_pilot.py','audit_report_delta_online_pilot.py','revision_common.py','extract_revision_prefixes.py','evaluate_revision_locality.py','fit_revision_geometry.py','revision_locality_common.py','evaluate_change_bayes.py'])
    files.extend(Path(__file__).resolve().parents[1]/f for f in ['src/hss/analysis/change_bayes.py'])
    plan=provenance(cfg,files);plan.update(cards=cards,expected_functional=len(tasks),smoke_ids=chosen[:2],source_scope='Frozen original MAP GMM assignments; no refit or outcome-selected examples.')
    freeze(r/'plan.json',plan);status(r,'prepare',state='complete',functional_records=len(tasks))


def gpu_guard(root):
    lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert shutil.disk_usage('/home/ubuntu').free>50*1024**3
    return lock


def confidence(cfg):
    import torch,zarr,pandas as pd
    from extract_revision_prefixes import load_model
    r=Path(cfg['output']);verify(r);out=r/'confidence';out.mkdir(exist_ok=True)
    if (out/'_SUCCESS.json').exists():return
    lock=gpu_guard(r);torch.set_num_threads(cfg['threads']);model,_=load_model(cfg)
    tokens=json.loads((r/'tokens.json').read_text());allrows=[];files={};started=time.monotonic();maxp=max(cfg['prefixes']);audit=[]
    for record in json.loads((r/'source_records.json').read_text()):
        shard=record['shard'];path=out/(shard+'.npz');receipt=path.with_suffix('.json')
        if receipt.exists():
            a=json.loads(receipt.read_text());assert a['plan_sha256']==sha(r/'plan.json') and a['arrays_sha256']==sha(path)
            allrows.extend(a['rows']);files[path.name]=sha(path);files[receipt.name]=sha(receipt);continue
        source=Path(cfg['openact'])/shard;frame=pd.read_parquet(source/'data.parquet');z=zarr.open_consolidated(str(source/'tensors.zarr'),mode='r')
        ptr=np.asarray(z['tokens/sample_ptr']);selection=np.concatenate([np.arange(int(lo),min(int(hi),int(lo)+maxp)) for lo,hi in zip(ptr[:-1],ptr[1:])])
        post=np.asarray(z['final_norm/post/per_token'].oindex[selection,:],dtype=np.float32)
        prompt=np.asarray(z['final_norm/post/prompt_last'],dtype=np.float32)
        arrays=np.full((len(frame),maxp+1,4),np.nan,np.float32);offset=0;ids=[];rows=[]
        for j,row in enumerate(frame.itertuples()):
            c=tokens[row.sample_id];n=min(maxp,len(c['response_ids']));h=np.concatenate([prompt[j:j+1],post[offset:offset+n]]);offset+=n;vals=[]
            with torch.inference_mode():
                for a in range(0,len(h),cfg['head_batch']):
                    b=min(a+cfg['head_batch'],len(h));x=torch.tensor(h[a:b],device=model.device,dtype=torch.bfloat16)
                    lp=model.lm_head(x).float().log_softmax(-1);top=lp.topk(2,dim=-1).values
                    targets=c['response_ids'][a:b];nll=np.full(b-a,np.nan,np.float32)
                    if targets:nll[:len(targets)]=-lp[torch.arange(len(targets),device=model.device),torch.tensor(targets,device=model.device)].cpu().numpy()
                    vals.append(np.column_stack([(-(lp.exp()*lp).sum(-1)).cpu().numpy(),(top[:,0]-top[:,1]).cpu().numpy(),top[:,0].exp().cpu().numpy(),nll]))
            values=np.concatenate(vals);arrays[j,:len(values)]=values;ids.append(row.sample_id)
            rows.append({'sample_id':row.sample_id,'row_index':c['row_index'],'available':n})
            # Real-prefix source check before proceeding through all shards.
            if len(audit)<2:
                p=16;full=c['prompt_ids']+c['response_ids'][:p]
                with torch.inference_mode():
                    fresh=model(input_ids=torch.tensor([full],device=model.device),use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1)
                    cached=model.lm_head(torch.tensor(h[p],device=model.device,dtype=torch.bfloat16)).float().log_softmax(-1)
                    kl=float((fresh.exp()*(fresh-cached)).sum());fe=float(-(fresh.exp()*fresh).sum())
                assert abs(kl)<.01
                audit.append({'sample_id':row.sample_id,'prefix':p,'fresh_vs_stored_kl':kl,'fresh_entropy':fe,'stored_batched_entropy':float(values[p,0])})
        write_npz(path,sample_ids=np.array(ids),values=arrays)
        write_json(receipt,{'rows':rows,'arrays_sha256':sha(path),'plan_sha256':sha(r/'plan.json')});allrows.extend(rows)
        files[path.name]=sha(path);files[receipt.name]=sha(receipt)
        write_json(out/'readout_audit.json',audit)
        status(r,'confidence',state='running',questions=len(allrows),expected=5000,seconds=time.monotonic()-started)
        print(json.dumps({'confidence_questions':len(allrows),'seconds':time.monotonic()-started}),flush=True)
    assert len(allrows)==5000
    write_json(out/'_SUCCESS.json',{'files':files,'questions':5000,'seconds':time.monotonic()-started,'scope':'Stored post-RMS bf16 states through pinned bf16 lm_head, raw forward distribution; tiny batch-rounding differences possible. No decoding processors.'})


def functional(cfg,smoke=False):
    import torch
    from extract_revision_prefixes import load_model
    from evaluate_revision_locality import measure
    r=Path(cfg['output']);plan=verify(r);out=r/('smoke' if smoke else 'functional');out.mkdir(exist_ok=True)
    if (out/'_SUCCESS.json').exists():return
    if not smoke:assert json.loads((r/'smoke/audit.json').read_text())['complete']
    tasks=json.loads((r/'functional_tasks.json').read_text())['tasks']
    if smoke:tasks=[t for t in tasks if t['sample_id'] in plan['smoke_ids'] and t['prefix']==16 and t['width']==4]
    tokens=json.loads((r/'tokens.json').read_text());d={}
    for rep in ['delta','state']:
        with np.load(r/(rep+'_decoder.npz')) as a:d[rep]={k:a[k].copy() for k in a.files}
    lock=gpu_guard(r);torch.set_num_threads(cfg['threads']);model,_=load_model(cfg);start=time.monotonic();done=0;baseline={}
    for task in tasks:
        path=out/(task['key']+'.json');ap=path.with_suffix('.npz');c=tokens[task['sample_id']];p=task['prefix'];width=task['width'];key=(c['sample_id'],p,width)
        if path.exists():
            saved=json.loads(path.read_text());assert saved['task']==task and saved['plan_sha256']==sha(r/'plan.json') and saved['arrays_sha256']==sha(ap)
            if task['method']=='identity':
                with np.load(ap) as z:baseline[key]=z['logp'].astype(float)
            done+=1;continue
        prefix=c['prompt_ids']+c['response_ids'][:p];ref=c['response_ids'][p:p+cfg['reference_tokens']];positions=list(range(len(prefix)-width,len(prefix)));cap={}
        def prehook(module,args,kwargs):
            h=args[0] if args else kwargs['hidden_states']
            if not cap and h.shape[1]==len(prefix):cap['before']=h[0,positions].float().cpu().numpy().copy()
        def transform(h):
            after=h.float().cpu().numpy();ideal=replacement(cap['before'],after,d['delta'],d['state'],task['method'])
            actual=torch.as_tensor(ideal,device=h.device,dtype=h.dtype)
            cap.update(after=after,ideal=ideal,actual=actual.float().cpu().numpy());return actual
        handle=model.model.layers[cfg['functional_layer']-1].register_forward_pre_hook(prehook,with_kwargs=True)
        try:metrics,lp,nll=measure(model,prefix,ref,cfg['functional_layer'],positions,transform)
        finally:handle.remove()
        if task['method']=='identity':
            baseline[key]=lp.astype(float)
            if smoke:
                with torch.inference_mode():plain=model(input_ids=torch.tensor([prefix],device=model.device),use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1).cpu().numpy()
                np.testing.assert_array_equal(plain,lp)
        metrics['kl']=float(np.dot(np.exp(baseline[key]),baseline[key]-lp.astype(float)))
        write_npz(ap,**cap,logp=lp,reference_nll=nll)
        write_json(path,{'task':task,'positions':positions,'metrics':metrics,'arrays_sha256':sha(ap),'plan_sha256':sha(r/'plan.json')})
        done+=1;status(r,'smoke' if smoke else 'functional',state='running',completed=done,expected=len(tasks),seconds=time.monotonic()-start)
    write_json(out/'_SUCCESS.json',{'records':done,'seconds':time.monotonic()-start,'plan_sha256':sha(r/'plan.json')})


def queue(cfg,path):
    r=Path(cfg['output']);r.mkdir(parents=True,exist_ok=True);lock=(r/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpu=Path(__file__).resolve().parents[1]/'.venv/bin/python';gpu=Path('/lambda/nfs/dami/openact/.venv/bin/python')
    jobs=[('prepare',cpu,'run_delta_online_pilot.py',['--stage','prepare']),('smoke',gpu,'run_delta_online_pilot.py',['--stage','smoke']),
          ('smoke_audit',gpu,'audit_report_delta_online_pilot.py',['--stage','smoke']),('confidence',gpu,'run_delta_online_pilot.py',['--stage','confidence']),
          ('risk',cpu,'evaluate_delta_online_pilot.py',[]),('risk_audit',cpu,'audit_report_delta_online_pilot.py',['--stage','risk']),
          ('functional',gpu,'run_delta_online_pilot.py',['--stage','functional']),
          ('audit',gpu,'audit_report_delta_online_pilot.py',['--stage','audit']),('report',cpu,'audit_report_delta_online_pilot.py',['--stage','report'])]
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]=str(cfg['threads'])
    state={'state':'running','pid':os.getpid(),'started_unix':time.time(),'stages':{}}
    for name,py,script,extra in jobs:
        state['phase']=name
        with (r/(name+'.log')).open('a') as log:
            child=subprocess.Popen([str(py),str(Path(__file__).with_name(script)),'--config',str(Path(path).resolve()),*extra],stdout=log,stderr=log,env=env)
            state['stages'][name]={'state':'running','pid':child.pid,'started_unix':time.time()};write_json(r/'queue_status.json',state);rc=child.wait()
        state['stages'][name].update(state='complete' if rc==0 else 'failed',returncode=rc,ended_unix=time.time())
        if rc:state['state']='failed';write_json(r/'queue_status.json',state);raise RuntimeError(name+' failed')
    state.update(state='complete',finished_unix=time.time(),delivery='visual_review_pending');write_json(r/'queue_status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',required=True,choices=['prepare','confidence','smoke','functional','queue']);a=p.parse_args();cfg=config(a.config)
    if a.stage=='queue':queue(cfg,a.config)
    elif a.stage=='prepare':prepare(cfg)
    elif a.stage=='confidence':confidence(cfg)
    else:functional(cfg,a.stage=='smoke')
