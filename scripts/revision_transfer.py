"""Frozen-map transfer with calibration, target refit and sample-matched controls.

No target confirmation labels participate in fitting, K selection, priors or
coverage thresholds. Ordinary generation/label integrity checks do read labels;
the confirmation split is method-held-out, not claimed never inspected by any
operational tool. This experiment concerns prediction/geometry, not causality.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp,expit,logit
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from revision_common import config,freeze,provenance,write_json,write_npz,sha,status,nearest,paired_ratio_ci
from revision_statistics import paired_auc_ci
from fit_revision_geometry import read_view


ROOT=Path('/lambda/nfs/dami/hss/revision-gsm8k-20260923')
SOURCE=Path('/lambda/nfs/dami/hss/revision-foundations-20260923')
COLLECTION=Path('/lambda/nfs/dami/openact/runs/gsm8k_transfer_20260923')


def prepare():
    if not (COLLECTION/'_SUCCESS').exists():
        raise ValueError('Full verified GSM8K collection required')
    ROOT.mkdir(parents=True,exist_ok=True)
    records=[];sources={}
    for number,marker in enumerate(sorted((COLLECTION/'qwen2').glob('shard_*/_COPY_VERIFIED.json'))):
        shard=marker.parent;receipt=json.loads(marker.read_text())
        for relative in ['data.parquet','manifest.json','labels/correctness.parquet']:
            if sha(shard/relative)!=receipt['files'][relative]['sha256']:
                raise ValueError(f'Metadata checksum mismatch: {shard/relative}')
        sources[str(shard)]=sha(marker)
        data=pd.read_parquet(shard/'data.parquet')
        manifest=json.loads((shard/'manifest.json').read_text())
        if manifest['model']['identifier']!='Qwen/Qwen2-7B-Instruct' or manifest['model'].get('revision')!='f2826a00ceef68f0f2b946d945ecc0477ce4450c':
            raise ValueError('Unexpected collection model/revision')
        if not data.prompt_template_hash.eq('507f30ec8346341e').all():
            raise ValueError('Transfer prompt differs from frozen MATH template')
        labels=pd.read_parquet(shard/'labels/correctness.parquet').set_index('sample_idx').loc[data.sample_idx]
        if not np.array_equal(labels.meta_sample_id.astype(str),data.sample_id.astype(str)):
            raise ValueError('Label/source identity mismatch')
        if not labels.is_correct.isin([0,1]).all() or not data.status.eq(1).all():
            raise ValueError('Incomplete generation or labels')
        for i,row in enumerate(data.itertuples()):
            question=json.loads(row.prompt_fields_json)['problem']
            group=hashlib.sha256(' '.join(question.split()).encode()).hexdigest()
            records.append({'sample_id':row.sample_id,'question_group':group,'label':int(labels.iloc[i].is_correct),
                'source_run':number,'sample_idx':row.sample_idx,'category':'gsm8k','level':0,
                'n_tokens':row.n_response_tokens,'n_prompt_tokens':row.n_prompt_tokens,
                'prompt_text':row.prompt_text,'response_text':row.response_text,'ground_truth':row.ground_truth,
                'transfer_split':row.transfer_split,'finish_reason':row.finish_reason})
    rows=pd.DataFrame(records)
    if len(rows)!=1319 or rows.sample_id.duplicated().any() or rows.question_group.duplicated().any():
        raise ValueError('Expected 1,319 unique question identities')
    source_cfg=json.loads((SOURCE/'prefixes/plan.json').read_text())['config']
    source_rows=pd.read_parquet(source_cfg['rows'],columns=['prompt_text'])
    normalized=lambda value:' '.join(str(value).split())
    overlap=rows.prompt_text.map(normalized).isin(set(source_rows.prompt_text.map(normalized)))
    if overlap.any():
        write_json(ROOT/'source_target_overlap.json',rows.loc[overlap,['sample_id','transfer_split']].to_dict('records'))
        raise ValueError('Source-target exact-question overlap requires an explicit exclusion amendment')
    splits=rows[['sample_id','transfer_split']].rename(columns={'transfer_split':'split'})
    splits['split']=splits['split'].map({'adaptation':'train','validation':'validation','confirmation':'test'})
    if splits.split.isna().any() or splits.split.value_counts().to_dict()!={'train':541,'test':536,'validation':242}:
        raise ValueError('Frozen transfer split changed')
    rows.to_parquet(ROOT/'rows.parquet',index=False);splits.to_parquet(ROOT/'splits.parquet',index=False)
    cfg={**source_cfg,'source':str(COLLECTION/'qwen2'),'rows':str(ROOT/'rows.parquet'),
         'splits':str(ROOT/'splits.parquet'),'output':str(ROOT),'expected_rows':1319}
    cfg.pop('intervention_geometry',None);cfg.pop('intervention_positions_per_question',None)
    freeze(ROOT/'target_config.json',cfg)
    write_json(ROOT/'source_audit.json',{'rows':len(rows),'sources':sources,
        'split':splits.split.value_counts().to_dict(),'labels':'is_correct=1; failures=1-is_correct',
        'source_complete_sha256':sha(COLLECTION/'_SUCCESS'),
        'operational_smoke_overlap':['gsm8k_0','gsm8k_1'],
        'confirmation_policy':'No target confirmation metric used to choose representation, model, K or readout.'})


def nb_fit(codes,y,k):
    counts=np.array([np.bincount(codes[y==c],minlength=k) for c in (0,1)],dtype=float)+1
    prior=np.bincount(y,minlength=2).astype(float)+1
    return {'nb_likelihood':counts/counts.sum(1,keepdims=True),'nb_prior':prior/prior.sum()}


def nb_predict(readout,codes):
    logp=np.log(readout['nb_likelihood'][:,codes].T)+np.log(readout['nb_prior'])
    return expit(logp[:,1]-logp[:,0])


def density(x,model):
    out=[];mu=model['centers'].astype(float);var=model['variances'].astype(float)
    for start in range(0,len(x),64):
        block=x[start:start+64].astype(float)
        logp=np.stack([-.5*(np.square(block-m)/v+np.log(2*np.pi*v)).sum(1) for m,v in zip(mu,var)],axis=1)
        out.append(logsumexp(logp+np.log(model['weights'])[None],axis=1))
    return np.concatenate(out)


def fit_map(x,path,grid,seeds):
    path.mkdir(parents=True,exist_ok=True);x=x.astype(np.float64)
    trials=[];best=None
    for k in grid:
        for seed in seeds:
            info=path/f'k{k}_seed{seed}.json';file=info.with_suffix('.npz')
            if info.exists():
                row=json.loads(info.read_text())
                with np.load(file) as a:
                    model={key:a[key].copy() for key in a.files}
            else:
                start=time.monotonic()
                g=GaussianMixture(k,covariance_type='diag',n_init=1,random_state=seed,
                    max_iter=300,tol=1e-4,reg_covar=1e-5)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',ConvergenceWarning);g.fit(x)
                    if not g.converged_:
                        g.set_params(warm_start=True,max_iter=600).fit(x)
                p=g.predict_proba(x).astype(float)
                entropy=float(-(p*np.log(np.maximum(p,1e-300))).sum())
                row={'k':k,'seed':seed,'icl':float(g.bic(x))+2*entropy,'bic':float(g.bic(x)),
                     'entropy':entropy,'converged':bool(g.converged_),'n_iter':int(g.n_iter_),
                     'seconds':time.monotonic()-start,'fit_samples':len(x)}
                model={'centers':g.means_,'variances':g.covariances_,'weights':g.weights_,'train_mean':x.mean(0)}
                write_npz(file,**model);write_json(info,row)
            trials.append(row)
            if row['converged'] and (best is None or row['icl']<best[0]['icl']):
                best=(row,model)
    if best is None:
        raise RuntimeError('No converged GMM')
    write_npz(path/'selected.npz',**best[1]);write_json(path/'selection.json',{'selected':best[0],'trials':trials,
        'K_at_upper_boundary':best[0]['k']==max(grid),'criterion':'training ICL only; no confirmation data'})
    return best[1]


def evaluate_job(job):
    source_cfg,target_cfg,prefix,layer,view=job
    with threadpool_limits(limits=2):
        return _evaluate_job(source_cfg,target_cfg,prefix,layer,view)


def _evaluate_job(source_cfg,target_cfg,prefix,layer,view):
    dest=ROOT/'transfer'/f'p{prefix}_l{layer}_{view}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'_SUCCESS.json').exists():
        return json.loads((dest/'summary.json').read_text())
    source,sx=read_view(source_cfg,prefix,layer,view)
    target,tx=read_view(target_cfg,prefix,layer,view)
    st=source.split.eq('train').to_numpy();sv=source.split.eq('validation').to_numpy()
    tt=target.split.eq('train').to_numpy();tv=target.split.eq('validation').to_numpy();test=target.split.eq('test').to_numpy()
    sy=1-source.label.to_numpy(int);ty=1-target.label.to_numpy(int)
    origin=SOURCE/'geometry'/f'p{prefix}_l{layer}_{view}'
    with np.load(origin/'decoder.npz') as d:
        fixed={k:d[k].copy() for k in ['centers','variances','weights','train_mean']}
    with np.load(origin/'readout.npz') as d:
        fixed_readout={k:d[k].copy() for k in ['nb_likelihood','nb_prior']}
        linear={k:d[k].copy() for k in ['scaler_mean','scaler_scale','linear_coef','linear_intercept']}
    k=len(fixed['centers']);fixed_codes=nearest(tx,fixed['centers'])
    # Match each target view's available adaptation count, rather than assuming
    # that all questions survive the 64-generated-token prefix.
    matched=source.loc[st,['sample_id']].copy()
    matched['hash']=matched.sample_id.map(lambda s:hashlib.sha256(('source-match-v1/'+s).encode()).hexdigest())
    matched_ids=matched.sort_values('hash').head(int(tt.sum())).sample_id
    sm=source.sample_id.isin(matched_ids).to_numpy()
    if sm.sum()!=tt.sum():
        raise ValueError('Cannot sample-match source training count')
    matched_map=fit_map(sx[sm],dest/'source_matched_map',source_cfg['k_grid'],source_cfg['seeds'])
    target_map=fit_map(tx[tt],dest/'target_map',source_cfg['k_grid'],source_cfg['seeds'])
    matched_codes=nearest(tx,matched_map['centers']);target_codes=nearest(tx,target_map['centers'])
    matched_readout=nb_fit(nearest(sx[sm],matched_map['centers']),sy[sm],len(matched_map['centers']))
    regimes=[('frozen_source_full',fixed,fixed_readout,fixed_codes,sx[sv],int(st.sum()),int(st.sum()),'MATH'),
             ('fixed_source_full_target_calibration',fixed,nb_fit(fixed_codes[tt],ty[tt],k),fixed_codes,sx[sv],int(st.sum()),int(tt.sum()),'GSM8K adaptation'),
             ('target_refit',target_map,nb_fit(target_codes[tt],ty[tt],len(target_map['centers'])),target_codes,tx[tv],int(tt.sum()),int(tt.sum()),'GSM8K adaptation'),
             ('frozen_source_matched',matched_map,matched_readout,matched_codes,sx[sv],int(sm.sum()),int(sm.sum()),'MATH matched subset'),
             ('fixed_source_matched_target_calibration',matched_map,nb_fit(matched_codes[tt],ty[tt],len(matched_map['centers'])),matched_codes,sx[sv],int(sm.sum()),int(tt.sum()),'GSM8K adaptation')]
    confidence=pd.read_parquet(ROOT/'confidence'/f'prefix_{prefix}.parquet').set_index('sample_id').loc[target.sample_id]
    predictions=pd.DataFrame({'sample_id':target.sample_id,'split':target.split,'failure':ty,
                              'entropy':confidence.next_token_entropy.to_numpy(),
                              'frozen_source_linear':expit(((tx-linear['scaler_mean'])/linear['scaler_scale'])@linear['linear_coef'].ravel()+linear['linear_intercept'][0])})
    summary=[]
    denominator=np.square(tx[test].astype(float)-tx[tt].mean(0)).sum(1)
    for name,model,readout,codes,calibration,n_map,n_labels,label_source in regimes:
        p=nb_predict(readout,codes);predictions[name]=p
        cal_codes=nearest(calibration,model['centers'])
        cal_distance=np.linalg.norm(calibration-model['centers'][cal_codes],axis=1)
        distance=np.linalg.norm(tx[test]-model['centers'][codes[test]],axis=1)
        threshold=float(np.quantile(cal_distance,.95))
        density_cut=float(np.quantile(density(calibration,model),.05))
        ll=density(tx[test],model)
        write_npz(dest/(name+'_readout.npz'),**readout)
        error=np.square(tx[test].astype(float)-model['centers'][codes[test]]).sum(1)
        summary.append({'regime':name,'K':len(model['centers']),'map_fit_samples':n_map,'label_fit_samples':n_labels,
            'label_fit_source':label_source,'confirmation_questions':int(test.sum()),
            'auroc':float(roc_auc_score(ty[test],p[test])),'log_loss':float(log_loss(ty[test],p[test])),
            'brier':float(brier_score_loss(ty[test],p[test])),
            'versus_entropy':paired_auc_ci(ty[test],p[test],predictions.entropy.to_numpy()[test]),
            'centroid_NMSE_common_target_train_center':paired_ratio_ci(error,denominator),
            'distance_coverage':float((distance<=threshold).mean()),'density_coverage':float((ll>=density_cut).mean()),
            'distance_threshold':threshold,'density_threshold':density_cut,
            'threshold_domain':'GSM8K validation' if name=='target_refit' else 'MATH validation'})
    comparisons=[]
    for name,baseline in [('fixed_source_full_target_calibration','frozen_source_full'),
                          ('target_refit','fixed_source_full_target_calibration'),
                          ('target_refit','fixed_source_matched_target_calibration'),
                          ('frozen_source_full','frozen_source_matched')]:
        comparisons.append({'regime':name,'baseline':baseline,
            **paired_auc_ci(ty[test],predictions[name].to_numpy()[test],predictions[baseline].to_numpy()[test])})
    predictions.to_parquet(dest/'predictions.parquet',index=False)
    write_json(dest/'fit_question_ids.json',{'source_matched':source.loc[sm,'sample_id'].tolist(),
                                           'target_adaptation':target.loc[tt,'sample_id'].tolist(),
                                           'target_validation':target.loc[tv,'sample_id'].tolist()})
    result={'prefix':prefix,'block':layer,'view':view,'target_split_counts':target.split.value_counts().to_dict(),
            'assignment':'nearest centroid','normalization':'none','regimes':summary,'paired_comparisons':comparisons,
            'geometry_denominator':'same target adaptation mean for every regime',
            'caution':'Prediction/geometry transfer, no causal equivalence claim; pointwise intervals'}
    result['continuous_baseline']={'source_linear_auroc':float(roc_auc_score(ty[test],predictions.frozen_source_linear.to_numpy()[test])),
                                   'entropy_auroc':float(roc_auc_score(ty[test],predictions.entropy.to_numpy()[test]))}
    write_json(dest/'summary.json',result);write_json(dest/'_SUCCESS.json',{'summary_sha256':sha(dest/'summary.json')})
    print(json.dumps({'transfer_complete':dest.name}),flush=True);return result


def evaluate():
    target_cfg=config(ROOT/'target_config.json')
    source_cfg=json.loads((SOURCE/'prefixes/plan.json').read_text())['config']
    if not (ROOT/'prefixes/_SUCCESS.json').exists() or not (ROOT/'confidence/_SUCCESS.json').exists():
        raise ValueError('Complete target features and current-prefix confidence required')
    plan={'source_config':source_cfg,'target_config':target_cfg,
          'prefixes':[0,16,64],'layers':[7,14,21,28],'views':['last','mean16','all_mean'],
          'scope':'No outcome-dependent selection across 36 views; report all',
          'primary_hypothesis':'Block28 prefix16 mean16 versus last, frozen-source NB AUROC on same target confirmation IDs; selected using source-only evidence',
          'no_confirmation_fitting':True}
    freeze(ROOT/'transfer_plan.json',provenance(plan,[Path(__file__),Path(__file__).with_name('revision_statistics.py'),
        SOURCE/'geometry_summary.json',SOURCE/'prefixes/plan.json',ROOT/'source_audit.json',ROOT/'prefixes/plan.json']))
    jobs=[(source_cfg,target_cfg,p,l,v) for p in plan['prefixes'] for l in plan['layers'] for v in plan['views']]
    results=[];status(ROOT,'transfer',state='running',completed=0,expected=len(jobs))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(evaluate_job,j) for j in jobs]):
            results.append(f.result());status(ROOT,'transfer',state='running',completed=len(results),expected=len(jobs))
    write_json(ROOT/'transfer_summary.json',results)
    # Secondary four-block naive Bayes: multiply emissions, count the prior ONCE.
    # Local state IDs do not need a cross-layer renaming for this readout.
    joint=[]
    for prefix in plan['prefixes']:
        for view in plan['views']:
            paths=[ROOT/'transfer'/f'p{prefix}_l{layer}_{view}' for layer in plan['layers']]
            frames=[pd.read_parquet(p/'predictions.parquet').set_index('sample_id') for p in paths]
            common=sorted(set.intersection(*(set(f.index) for f in frames)))
            frames=[f.loc[common] for f in frames]
            y=frames[0].failure.to_numpy();test=frames[0].split.eq('test').to_numpy()
            for f in frames[1:]:
                np.testing.assert_array_equal(y,f.failure)
                np.testing.assert_array_equal(frames[0].split,f.split)
            pred=pd.DataFrame({'sample_id':common,'split':frames[0].split.to_numpy(),'failure':y})
            for regime in ['frozen_source_full','fixed_source_full_target_calibration','target_refit',
                           'frozen_source_matched','fixed_source_matched_target_calibration']:
                priors=[np.load(p/(regime+'_readout.npz'))['nb_prior'] for p in paths]
                for prior in priors[1:]:
                    np.testing.assert_allclose(prior,priors[0],rtol=0,atol=0)
                score=expit(sum(logit(np.clip(f[regime].to_numpy(),1e-12,1-1e-12)) for f in frames)
                            -(len(frames)-1)*np.log(priors[0][1]/priors[0][0]))
                pred[regime]=score
                joint.append({'prefix':prefix,'view':view,'regime':regime,'blocks':plan['layers'],
                    'confirmation_questions':int(test.sum()),'auroc':float(roc_auc_score(y[test],score[test])),
                    'brier':float(brier_score_loss(y[test],score[test])),'log_loss':float(log_loss(y[test],score[test])),
                    'caution':'Four-block conditional-independence approximation; no cross-layer identity claim'})
            (ROOT/'joint').mkdir(exist_ok=True)
            pred.to_parquet(ROOT/'joint'/f'p{prefix}_{view}.parquet',index=False)
    write_json(ROOT/'joint_summary.json',joint)
    a=pd.read_parquet(ROOT/'transfer/p16_l28_mean16/predictions.parquet')
    b=pd.read_parquet(ROOT/'transfer/p16_l28_last/predictions.parquet')
    a=a.loc[a.split.eq('test')].set_index('sample_id');b=b.loc[b.split.eq('test')].set_index('sample_id')
    common=sorted(a.index.intersection(b.index));a,b=a.loc[common],b.loc[common]
    np.testing.assert_array_equal(a.failure,b.failure)
    write_json(ROOT/'primary_comparison.json',{'definition':plan['primary_hypothesis'],
        'frozen_state_NB':paired_auc_ci(a.failure,a.frozen_source_full,b.frozen_source_full),
        'secondary_frozen_linear':paired_auc_ci(a.failure,a.frozen_source_linear,b.frozen_source_linear),
        'secondary_recalibrated_state_NB':paired_auc_ci(a.failure,a.fixed_source_full_target_calibration,b.fixed_source_full_target_calibration)})
    status(ROOT,'transfer',state='complete',completed=len(results),expected=len(jobs))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','evaluate'])
    args=p.parse_args()
    try:
        prepare() if args.stage=='prepare' else evaluate()
    except BaseException:
        status(ROOT,'transfer_'+args.stage,state='failed',traceback=traceback.format_exc());raise
