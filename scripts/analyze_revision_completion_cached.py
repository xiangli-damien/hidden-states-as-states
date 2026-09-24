"""A/B/D: reuse audited reconstruction and frozen prompt-last maps, no new fits."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logsumexp
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from revision_common import sha,write_json,write_npz,nearest
from revision_completion_common import ordered


def features(path):
    rows=[];values=[];hashes={}
    for marker in sorted((path/'prefixes').glob('shard_*/_SUCCESS.json')):
        rc=json.loads(marker.read_text());folder=marker.parent
        for name in ['prefix_0.npz','rows.parquet']:
            assert sha(folder/name)==rc['sha256'][name]
            hashes[str(folder/name)]=rc['sha256'][name]
        frame=pd.read_parquet(folder/'rows.parquet')
        with np.load(folder/'prefix_0.npz') as f:
            assert f['valid'].all();x=f['window'][:,:,-1].copy()
        assert len(x)==len(frame);rows.append(frame);values.append(x)
    frame=pd.concat(rows,ignore_index=True)
    assert not frame.sample_id.duplicated().any()
    return frame,np.concatenate(values),hashes


def metrics(y,p,draws=2000):
    assert len(np.unique(y))==2
    rng=np.random.default_rng(42);boot=[]
    for _ in range(draws):
        ix=rng.integers(len(y),size=len(y))
        if len(np.unique(y[ix]))==2:boot.append(roc_auc_score(y[ix],p[ix]))
    return {'n':len(y),'failures':int(y.sum()),'correct':int(len(y)-y.sum()),
        'auroc':float(roc_auc_score(y,p)),'auroc_ci95':np.quantile(boot,[.025,.975]).tolist(),
        'accuracy_at_fixed_0.5':float(np.mean((p>.5)==y)),'brier':float(brier_score_loss(y,p)),
        'log_loss':float(log_loss(y,p,labels=[0,1])),'bootstrap_valid':len(boot)}


def run(root):
    started=time.time();plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    dest=root/'cached';dest.mkdir(exist_ok=True)
    if (dest/'_SUCCESS.json').exists():raise RuntimeError('Cached analysis already complete')
    sources={};atable=[]
    for name,base in [('MATH historical test32',Path(cfg['locality'])),('GSM8K confirmation64',Path(cfg['confirmation']))]:
        report=base/'report';rc=json.loads((report/'_SUCCESS.json').read_text());audit=json.loads((report/'statistics_audit.json').read_text())
        assert audit['complete'] and audit['report_receipt_sha256']==sha(report/'_SUCCESS.json')
        assert sha(report/'summary.parquet')==rc['files']['summary.parquet']
        sub=pd.read_parquet(report/'summary.parquet')
        sub=sub.loc[sub.split.eq('test')&sub.prefix_tokens.eq(16)&sub.layer.eq(14)&sub.width.eq(16)&sub.role.eq('tokens')]
        wanted=['centroid','shared_8','local_8','shared_64','remove_local_8_1.0','remove_local_radial_random_8_1.0']
        for row in sub.loc[sub.method.isin(wanted)].to_dict('records'):
            atable.append({'dataset':name,'method':row['method'],'n':row['n'],
                'KL':row['next_token_kl_estimate'],'KL_low':row['next_token_kl_low'],'KL_high':row['next_token_kl_high'],
                'reference_delta_NLL':row['delta_nll_estimate'],'NLL_low':row['delta_nll_low'],'NLL_high':row['delta_nll_high']})
        for file in ['_SUCCESS.json','statistics_audit.json','summary.parquet']:sources[str(report/file)]=sha(report/file)
    assert len(atable)==12
    pd.DataFrame(atable).to_parquet(dest/'reconstruction.parquet',index=False)
    source=Path(cfg['foundation']);target=Path(cfg['transfer'])
    oldaudit=json.loads((target/'audit.json').read_text())
    assert oldaudit['selection_training_ids_predictions_metrics_CIs_geometry_coverage_and_joint_prior_verified']
    hashes=json.loads((target/'report/input_hashes.json').read_text())
    assert sha(target/'report/input_hashes.json')==oldaudit['input_hashes_sha256']
    sf,sx,sh=features(source);tf,tx,th=features(target);sources.update(sh);sources.update(th)
    assert len(sf)==5000 and len(tf)==1319 and sx.shape[1:]==tx.shape[1:]==(4,3584)
    st=sf.split.eq('train').to_numpy();sv=sf.split.eq('validation').to_numpy();se=sf.split.eq('test').to_numpy()
    te=tf.split.eq('test').to_numpy();y=1-tf.label.to_numpy(int);sy=1-sf.label.to_numpy(int)
    adapt=ordered(tf.loc[tf.split.eq('train'),'sample_id'].tolist(),'completion-adapt128-v1')[:128]
    ad=tf.sample_id.isin(adapt).to_numpy();assert ad.sum()==128 and not np.any(ad&te)
    frozen_terms=[];adapt_terms=[];assignments=[];models=[];support_rows=[];ids=tf.sample_id.tolist()
    predictions=pd.DataFrame({'sample_id':ids,'split':tf.split,'failure':y,'adapt128':ad})
    for li,layer in enumerate([7,14,21,28]):
        origin=source/'geometry'/f'p0_l{layer}_last';saved=target/'transfer'/f'p0_l{layer}_last'
        model=np.load(origin/'decoder.npz');readout=np.load(origin/'readout.npz');center=model['centers'].astype(float)
        source_codes=nearest(sx[:,li],center);codes=nearest(tx[:,li],center);assignments.append((source_codes,codes));models.append(center)
        with np.load(origin/'assignments.npz') as a:
            np.testing.assert_array_equal(a['sample_id'],sf.sample_id.to_numpy(str));np.testing.assert_array_equal(a['nearest'],source_codes)
        counts=np.stack([np.bincount(source_codes[st&(sy==c)],minlength=len(center))+1 for c in (0,1)])
        likelihood=counts/counts.sum(1,keepdims=True);prior=(np.bincount(sy[st],minlength=2)+1)/(st.sum()+2)
        np.testing.assert_array_equal(readout['nb_likelihood'],likelihood);np.testing.assert_array_equal(readout['nb_prior'],prior)
        assert sha(saved/'predictions.parquet')==hashes[str(saved/'predictions.parquet')]
        old=pd.read_parquet(saved/'predictions.parquet').set_index('sample_id').loc[ids]
        logterm=np.log(likelihood[:,codes].T);frozen_terms.append(logterm)
        p=expit(logterm[:,1]-logterm[:,0]+np.log(prior[1]/prior[0]))
        np.testing.assert_allclose(p,old.frozen_source_full,rtol=1e-12,atol=1e-12)
        count128=np.stack([np.bincount(codes[ad&(y==c)],minlength=len(center))+1 for c in (0,1)])
        like128=count128/count128.sum(1,keepdims=True);adapt_terms.append(np.log(like128[:,codes].T))
        predictions[f'state_l{layer}']=codes
        dist=np.linalg.norm(tx[:,li]-center[codes],axis=1)
        sd=np.linalg.norm(sx[:,li]-center[source_codes],axis=1)
        threshold=float(np.quantile(sd[sv],.95));predictions[f'distance_l{layer}']=dist
        support_rows.append({'block':layer,'K':len(center),'source_validation_n':int(sv.sum()),'source_test_n':int(se.sum()),
            'source_validation_q95':threshold,'source_test_coverage':float((sd[se]<=threshold).mean()),
            'target_all_coverage':float((dist<=threshold).mean()),'target_confirmation_coverage':float((dist[te]<=threshold).mean()),
            'source_test_occupancy':(np.bincount(source_codes[se],minlength=len(center))/se.sum()).tolist(),
            'target_confirmation_occupancy':(np.bincount(codes[te],minlength=len(center))/te.sum()).tolist()})
        write_npz(dest/f'block{layer}.npz',source_code=source_codes,target_code=codes,source_distance=sd,target_distance=dist,
            likelihood=likelihood,prior=prior,adapt128_likelihood=like128)
        if layer==28:
            linear=expit(((tx[:,li]-readout['scaler_mean'])/readout['scaler_scale'])@readout['linear_coef'].ravel()+readout['linear_intercept'][0])
            np.testing.assert_allclose(linear,old.frozen_source_linear,rtol=1e-12,atol=1e-12)
            predictions['frozen_linear28']=linear
        for pth in [origin/'decoder.npz',origin/'readout.npz',origin/'assignments.npz',saved/'predictions.parquet']:
            sources[str(pth)]=sha(pth)
    logf=sum(frozen_terms)+np.log(prior)[None,:]
    target_prior=(np.bincount(y[ad],minlength=2)+1)/(ad.sum()+2)
    loga=sum(adapt_terms)+np.log(target_prior)[None,:]
    predictions['frozen_nb']=expit(logf[:,1]-logf[:,0]);predictions['adapt128_nb']=expit(loga[:,1]-loga[:,0])
    joint=pd.read_parquet(target/'joint/p0_last.parquet').set_index('sample_id').loc[ids]
    np.testing.assert_allclose(predictions.frozen_nb,joint.frozen_source_full,rtol=1e-10,atol=1e-12)
    predictions.to_parquet(dest/'transfer_predictions.parquet',index=False)
    write_json(dest/'adaptation.json',{'sample_ids':adapt,'prior':target_prior.tolist(),'source':'128of541 existing official-test adaptation pool','confirmation_ids':tf.loc[te,'sample_id'].tolist()})
    btable=[]
    for scope,ix,methods in [('original_confirmation536',te,['frozen_nb','frozen_linear28','adapt128_nb']),
                             ('all1319_descriptive',np.ones(len(tf),bool),['frozen_nb','frozen_linear28'])]:
        for method in methods:btable.append({'scope':scope,'method':method,**metrics(y[ix],predictions[method].to_numpy()[ix])})
    # D uses fixed geometric correspondence only. Verify old mapping, then
    # add paired bootstrap intervals for flow minus independent occupancy chance.
    previous=json.loads((source/'alignment.json').read_text());flows=[]
    for li,(a,b) in enumerate(zip([7,14,21],[14,21,28])):
        ca,cb=models[li:li+2]
        # Reproduce the frozen encoder's original float32 cosine arithmetic.
        ca32,cb32=ca.astype(np.float32),cb.astype(np.float32)
        sim=(ca32@cb32.T)/(np.linalg.norm(ca32,axis=1)[:,None]*np.linalg.norm(cb32,axis=1)[None,:])
        ia,ib=linear_sum_assignment(sim,maximize=True);mapping=dict(zip(ia.tolist(),ib.tolist()))
        old=next(r for r in previous if r['from']==a and r['to']==b and r['mapping']=='cosine')
        assert old['mapping_pairs']==list(map(list,zip(ia.tolist(),ib.tolist())))
        x=assignments[li][0][se];z=assignments[li+1][0][se]
        matched=np.array([mapping.get(int(i),-1)==int(j) for i,j in zip(x,z)])
        def chance(ix):
            pa=np.bincount(x[ix],minlength=len(ca))/len(ix);pb=np.bincount(z[ix],minlength=len(cb))/len(ix)
            return float(np.sum(pa[ia]*pb[ib]))
        observed=float(matched.mean());ch=chance(np.arange(len(x)))
        np.testing.assert_allclose(observed,old['heldout_flow']['estimate'],rtol=0,atol=1e-15)
        np.testing.assert_allclose(ch,old['heldout_independent_marginal_chance'],rtol=0,atol=1e-15)
        rng=np.random.default_rng(42);boot=[]
        for _ in range(2000):
            ix=rng.integers(len(x),size=len(x));boot.append(float(matched[ix].mean()-chance(ix)))
        flows.append({'from':a,'to':b,'n':len(x),'flow':observed,'chance':ch,'excess':observed-ch,
                      'excess_ci95':np.quantile(boot,[.025,.975]).tolist(),'mapping_pairs':old['mapping_pairs']})
    write_npz(dest/'source_assignments.npz',sample_ids=sf.sample_id.to_numpy(str),split=sf.split.to_numpy(str),
        states=np.column_stack([x[0] for x in assignments]),labels=sy)
    sources[str(source/'alignment.json')]=sha(source/'alignment.json');sources[str(target/'audit.json')]=sha(target/'audit.json')
    write_json(dest/'sources.json',sources)
    write_json(dest/'summary.json',{'complete':True,'reconstruction':atable,'transfer':btable,'support':support_rows,'flow':flows,
        'scope':'Frozen4blocks [7,14,21,28], prefix0/last, raw pre-finalRMS; source-only NB and last-block linear. Adapt128 is not zero-shot.',
        'full_target_accuracy':float(1-y.mean()),'seconds':time.time()-started,'inputs_sha256':sha(dest/'sources.json')})
    write_json(dest/'_SUCCESS.json',{'complete':True,'summary_sha256':sha(dest/'summary.json'),
        'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}})
    print(json.dumps({'cached_complete':True,'seconds':time.time()-started,'transfer':btable}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):run(a.root)
