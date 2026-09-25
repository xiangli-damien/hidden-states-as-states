"""Landmark risk evaluation using only observed prefix routes and confidence."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from hss.analysis.change_bayes import bootstrap_auc,far_threshold,MonotonePlatt
from evaluate_change_bayes import fit_adjustment
from revision_common import config,sha,write_json,write_npz,freeze,status
from run_delta_online_pilot import verify
from delta_online_common import route_features,RouteBayes,prefix_controls


def run(cfg):
    r=Path(cfg['output']);plan=verify(r);out=r/'risk';out.mkdir(exist_ok=True)
    if (out/'_SUCCESS.json').exists():return
    started=time.monotonic();rows=pd.read_parquet(r/'rows.parquet');idx={s:i for i,s in enumerate(rows.sample_id)}
    with np.load(r/'prefix_sequences.npz') as z:arrays={k:z[k].copy() for k in z.files}
    confidence=np.full((5000,max(cfg['prefixes'])+1,4),np.nan,np.float32)
    receipt=json.loads((r/'confidence/_SUCCESS.json').read_text())
    for name,h in receipt['files'].items():assert sha(r/'confidence'/name)==h
    for p in sorted((r/'confidence').glob('shard*.npz')):
        with np.load(p) as z:
            for sid,v in zip(z['sample_ids'],z['values']):confidence[idx[str(sid)]]=v
    results=[];contrasts=[];coverage=[]
    for p in cfg['prefixes']:
        dest=out/f'p{p}';dest.mkdir(exist_ok=True)
        if (dest/'_SUCCESS.json').exists():
            results.extend(json.loads((dest/'metrics.json').read_text()));contrasts.extend(json.loads((dest/'contrasts.json').read_text()));coverage.append(json.loads((dest/'coverage.json').read_text()));continue
        ix=np.flatnonzero(arrays['lengths']>p);frame=rows.iloc[ix].reset_index(drop=True)
        cov={'prefix':p,'eligible':len(ix),'excluded_finished':int(5000-len(ix)),'split_counts':frame.split.value_counts().to_dict(),
             'test_failures':int((1-frame.loc[frame.split.eq('test'),'label']).sum())}
        write_json(dest/'coverage.json',cov);coverage.append(cov)
        tr=frame.split.eq('train').to_numpy();va=frame.split.eq('validation').to_numpy();te=frame.split.eq('test').to_numpy();y=1-frame.label.to_numpy(int)
        controls=np.stack([prefix_controls(confidence[i],rows.n_prompt_tokens.iloc[i],arrays['kinds'][i],p) for i in ix]);assert np.isfinite(controls).all()
        features={};shapes={}
        for rep in ['delta','state']:
            for kind,depth,temporal in [('occupancy',False,False),('depth',True,False),('full',True,True)]:
                data=[route_features(arrays[rep][i,:p],plan['cards'][rep],depth,temporal) for i in ix]
                name=rep+'_'+kind;features[name]=np.stack([v[0] for v in data]);shapes[name]=data[0][1];del data
            sh=[]
            for i in ix:
                # Fixed position shuffle uses only the visible prefix, shared across layers.
                perm=np.random.default_rng(cfg['seed']+int(i)*1000+p).permutation(p)
                sh.append(route_features(arrays[rep][i,:p][perm],plan['cards'][rep],True,True)[0])
            features[rep+'_shuffled']=np.stack(sh)
        write_npz(dest/'features.npz',row_index=ix,controls=controls,**features)
        folds=list(StratifiedKFold(5,shuffle=True,random_state=cfg['seed']).split(np.zeros(tr.sum()),y[tr]));fold_id=np.full(len(frame),-1)
        for j,(_,held) in enumerate(folds):fold_id[np.flatnonzero(tr)[held]]=j
        write_npz(dest/'folds.npz',fold=fold_id,row_index=ix)
        scores={};oofs={};selection={};models=dest/'models';models.mkdir(exist_ok=True)
        with threadpool_limits(cfg['threads']):
            folder=models/'controls';folder.mkdir(exist_ok=True)
            s,h,info=fit_adjustment(controls[tr],controls,y[tr],y[va],va,cfg,folder)
            scores['controls']=s;scores['controls_hgb']=h;selection['controls']=info
            for name in shapes:
                x=features[name];folder=models/name;folder.mkdir(exist_ok=True)
                m=RouteBayes(shapes[name],cfg['alpha']).fit(x[tr],y[tr]);scores[name]=m.decision_function(x);joblib.dump(m,folder/'bayes.joblib')
                oof=np.full(tr.sum(),np.nan)
                for j,(fit,hold) in enumerate(folds):
                    fm=RouteBayes(shapes[name],cfg['alpha']).fit(x[tr][fit],y[tr][fit]);oof[hold]=fm.decision_function(x[tr][hold]);joblib.dump(fm,folder/f'fold{j}.joblib')
                assert np.isfinite(oof).all();oofs[name]=oof
                sub=folder/'adjusted';sub.mkdir(exist_ok=True)
                s,h,info=fit_adjustment(np.column_stack([controls[tr],oof]),np.column_stack([controls,scores[name]]),y[tr],y[va],va,cfg,sub)
                scores[name+'_adjusted']=s;selection[name]=info
                if name.endswith('_full'):scores[name+'_frozen_shuffle']=m.decision_function(features[name.removesuffix('_full')+'_shuffled'])
                print(json.dumps({'prefix':p,'fit':name}),flush=True)
        probs={};thresholds={};calibrations={}
        for name,s in scores.items():
            cal=MonotonePlatt().fit(s[va],y[va]);probs[name]=cal.predict_proba(s);calibrations[name]=vars(cal)
            thresholds[name]=far_threshold(s[va & (y==0)],cfg['far'])
        pd.DataFrame({'sample_id':frame.sample_id,'split':frame.split,**scores}).to_parquet(dest/'scores.parquet',index=False)
        pd.DataFrame({'sample_id':frame.sample_id,**probs}).to_parquet(dest/'probabilities.parquet',index=False)
        write_npz(dest/'train_oof.npz',**oofs);write_json(dest/'selection.json',{'models':selection,'thresholds':thresholds,'calibrations':calibrations})
        freeze(dest/'SCORES_FROZEN.json',{'scores_sha256':sha(dest/'scores.parquet'),'selection_sha256':sha(dest/'selection.json'),'features_sha256':sha(dest/'features.npz'),'plan_sha256':sha(r/'plan.json')})
        yt=y[te];positive=np.flatnonzero(yt);negative=np.flatnonzero(1-yt);rng=np.random.default_rng(cfg['seed']+p)
        bootstrap=np.column_stack([rng.choice(positive,(cfg['bootstrap'],len(positive))),rng.choice(negative,(cfg['bootstrap'],len(negative)))])
        write_npz(dest/'bootstrap.npz',indices=bootstrap);metrics=[];draws={}
        for name,s in scores.items():
            boot=bootstrap_auc(yt,s[te],bootstrap);draws[name]=boot;alarm=s[te]>thresholds[name]
            metrics.append({'prefix':p,'name':name,'n_test':int(te.sum()),'auroc':float(roc_auc_score(yt,s[te])),
                'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),
                'auprc':float(average_precision_score(yt,s[te])),'brier':float(brier_score_loss(yt,probs[name][te])),
                'test_far':float(alarm[yt==0].mean()),'failure_recall':float(alarm[yt==1].mean()),
                'validation_far':float((s[va & (y==0)]>thresholds[name]).mean())})
        tab={m['name']:m for m in metrics};checks=[]
        pairs=[('delta_full_adjusted','controls'),('delta_full_adjusted','state_full_adjusted'),
               ('delta_full_adjusted','delta_occupancy_adjusted'),('delta_depth_adjusted','delta_occupancy_adjusted'),
               ('delta_full_adjusted','delta_depth_adjusted'),('delta_full','delta_full_frozen_shuffle')]
        for a,b in pairs:
            d=draws[a]-draws[b];checks.append({'prefix':p,'a':a,'b':b,'delta':tab[a]['auroc']-tab[b]['auroc'],
                'ci_low':float(np.quantile(d,.025)),'ci_high':float(np.quantile(d,.975))})
        write_json(dest/'metrics.json',metrics);write_json(dest/'contrasts.json',checks);write_npz(dest/'bootstrap_auc.npz',**draws)
        assert all(m['validation_far']<=cfg['far']+1e-10 for m in metrics)
        write_json(dest/'_SUCCESS.json',{'records':len(frame),'metrics':len(metrics),'scores_sha256':sha(dest/'scores.parquet')})
        results.extend(metrics);contrasts.extend(checks);status(r,'risk',state='running',completed_prefix=p,seconds=time.monotonic()-started)
        del features,shapes
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    write_json(out/'coverage.json',coverage);write_json(out/'_SUCCESS.json',{'seconds':time.monotonic()-started,'prefixes':cfg['prefixes'],
        'scope':'Supervised eventual-failure risk conditional on still ongoing; no final answer length, future entropy, test fitting, score flipping or step-error labels.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(config(a.config))
