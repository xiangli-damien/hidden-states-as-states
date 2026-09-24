"""Execute frozen-map P1/P2 route analyses; correctness never fits scores.

This is a descriptive study on previously explored, whole-dataset maps, NOT
the independent-fit P3 experiment. All completed-answer scores are retrospective.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, adjusted_rand_score
from threadpoolctl import threadpool_limits

from hss.experiments.artifacts import file_digest, save_json
from hss.route.counts import crossfit_scores, group_folds
from hss.route.nulls import ConditionalRouting, occupancy_null, codes


def quantile_bins(x, bins):
    return np.searchsorted(np.unique(np.quantile(x, np.arange(1,bins)/bins)), x, side='right')


def nuisance_codes(meta):
    category=pd.factorize(meta.category.fillna('unknown'))[0]
    level=pd.factorize(meta.level.fillna(-1))[0]
    length=quantile_bins(meta.n_tokens,4)
    prompt=quantile_bins(meta.n_prompt_tokens,3)
    finish=pd.factorize(meta.finish_reason)[0]
    return dict(raw=np.zeros(len(meta),int), length=length,
                task_length=codes(category,level,length),
                full=codes(category,level,length,prompt,finish))


def fdr(p):
    p=np.asarray(p,float); order=np.argsort(p); out=np.empty(len(p))
    out[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return out


def conditional_auc(y,score,strata):
    wins=0.; pairs=0; supported=0
    for g in np.unique(strata):
        idx=strata==g; yy=y[idx]; n1=int(yy.sum()); n0=len(yy)-n1
        if n1 and n0:
            wins+=rankdata(score[idx])[yy==1].sum()-n1*(n1+1)/2
            pairs+=n1*n0; supported+=len(yy)
    return dict(auc=float(wins/pairs) if pairs else None, pairs=pairs, support_n=supported)


def evaluate_scores(meta,scores,controls,root):
    # This is the first point where correctness is supplied to scoring evaluation.
    y=1-meta.label.to_numpy(int); rows=[]; curves={}; rng=np.random.default_rng(8241)
    boot=[rng.integers(len(y),size=len(y)) for _ in range(400)]
    length=scores['baseline__length'].to_numpy()
    length_boot=np.array([roc_auc_score(y[i],length[i]) for i in boot])
    for name in scores.columns:
        if name in ['sample_id','fold']: continue
        s=scores[name].to_numpy(float)
        if not np.isfinite(s).all(): raise ValueError('Nonfinite score: '+name)
        auc=float(roc_auc_score(y,s)); values=np.array([roc_auc_score(y[i],s[i]) for i in boot])
        cond=conditional_auc(y,s,controls['full'])
        order=np.argsort(s,kind='stable'); cumulative=y[order].cumsum()/np.arange(1,len(y)+1)
        idx=np.maximum(0,np.ceil(np.linspace(.01,1,100)*len(y)).astype(int)-1)
        curves[name]=dict(coverage=((idx+1)/len(y)).tolist(),risk=cumulative[idx].tolist())
        rows.append(dict(method=name.split('__')[0],score=name.split('__')[1],failure_auroc=auc,
            ci_low=float(np.quantile(values,.025)),ci_high=float(np.quantile(values,.975)),
            failure_auprc=float(average_precision_score(y,s)),conditional_auc=cond['auc'],conditional_support_n=cond['support_n'],
            delta_vs_length=auc-float(roc_auc_score(y,length)),delta_length_ci_low=float(np.quantile(values-length_boot,.025)),
            delta_length_ci_high=float(np.quantile(values-length_boot,.975)),risk_at_50=float(cumulative[int(.5*len(y))-1]),
            length_spearman=float(spearmanr(s,length).statistic)))
    table=pd.DataFrame(rows);table.to_csv(root/'score_evaluation.csv',index=False)
    save_json(root/'risk_coverage.json',curves)
    return table


def edge_tables(states,y):
    tables=[]; occupancy=[]
    for method in ['gmm','mfa']:
        z=states[method]
        for l in range(z.shape[1]):
            for a in np.unique(z[:,l]):
                idx=z[:,l]==a; n=int(idx.sum()); nc=int(y[idx].sum())
                occupancy.append(dict(method=method,layer=l+1,cluster=int(a),n=n,n_correct=nc,accuracy=nc/n,
                    p_given_correct=nc/y.sum(),p_given_incorrect=(n-nc)/(len(y)-y.sum())))
        for l in range(z.shape[1]-1):
            for a in np.unique(z[:,l]):
                origin=z[:,l]==a; nc=int(y[origin].sum()); ne=int(origin.sum())-nc
                for b in np.unique(z[:,l+1]):
                    idx=origin&(z[:,l+1]==b); n=int(idx.sum()); cc=int(y[idx].sum()); ce=n-cc
                    tables.append(dict(method=method,layer_from=l+1,layer_to=l+2,origin=int(a),destination=int(b),n=n,
                        correct=cc,incorrect=ce,origin_correct=nc,origin_incorrect=ne,
                        p_correct=cc/nc if nc else np.nan,p_incorrect=ce/ne if ne else np.nan,
                        delta=(ce/ne-cc/nc) if nc and ne else np.nan,
                        displayed_main_edge=bool(n>=30 and nc>=10 and ne>=10)))
    return pd.DataFrame(tables),pd.DataFrame(occupancy)


def run(args):
    start=time.monotonic(); source=Path(args.inputs); root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    for name,value in manifest['files'].items():
        if file_digest(source/name)!=value: raise ValueError('Changed input: '+name)
    meta=pd.read_parquet(source/'rows.parquet')
    if not meta.question_group.is_unique: raise ValueError('Use group bootstrap for repeated question groups before this study')
    states=dict(np.load(source/'states.npz')); probs=dict(np.load(source/'probabilities.npz')); geo=dict(np.load(source/'geometry.npz'))
    controls=nuisance_codes(meta); folds=group_folds(meta.question_group)
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),samples=len(meta),source_sha256=file_digest(source/'manifest.json'),
        driver_sha256=file_digest(Path(__file__)),stage='P1 and existing-model P2 only',permutations=args.permutations,
        occupancy_permutations=args.occupancy_permutations,bootstrap=400,folds='five SHA256 question-group folds, independent of correctness',
        score_direction='Larger is failure risk; no post-hoc reversal',primary_score='context',alpha=1,second_order_strength=10,
        main_conditional_test='Equal-class JS of destination given origin and nuisance. Common pooled group weights; >=2 rows per class per group.',
        primary_controls='category x level x response-length quartile x prompt-length tertile x finish reason',
        multiplicity='BH across 54 main method/layer comparisons separately per control tier; global Bonferroni min-p valid under dependent layers.',
        global_test_amendment='Use Bonferroni union test rather than summing independently permuted layer statistics; no assumption of independent layers.',
        no_new_fit=True,raw_cluster_inputs=True,matching_unchanged=True,
        limitations=['Maps/K/rank fitted on all 5000; only count fitting cross-validated. Prior data exploration, not a fresh confirmatory set.',
          'Controls use coarse bins; residual within-bin confounding remains. Sparse groups are excluded and coverage is explicit.',
          'Full-response mean features cannot establish prospective detection or first-error location.',
          'Seed/rank-specific K sensitivity is not same-K covariance control. Bootstrap intervals condition on fixed maps/scores.',
          'No causal test, no calibrated correctness probability, no claim of 10% FAR.'])
    marker=root/'protocol.json'
    if marker.exists(): raise ValueError('Use a fresh output directory; do not overwrite a frozen analysis')
    save_json(marker,protocol)
    scores=pd.DataFrame(dict(sample_id=meta.sample_id,fold=folds)); layer_scores={}; gains=[]
    for method,z in states.items():
        if method.endswith('_global'): continue
        pp=[probs[f'{method}_{l}'] for l in range(1,29)] if method in ['gmm','mfa'] else None
        s,ls=crossfit_scores(z,folds,pp)
        gain=s.pop('second_order_predictive_gain')
        gains.append(dict(method=method,mean_gain_nats=float(gain.mean()),median_gain_nats=float(np.median(gain)),
                          fraction_positive=float((gain>0).mean()),se=float(gain.std(ddof=1)/np.sqrt(len(gain)))))
        for name,v in s.items(): scores[method+'__'+name]=v
        if method in ['gmm','mfa']:
            for name,v in ls.items(): layer_scores[method+'__'+name]=v
            scores[method+'__global_switch']=(states[method+'_global'][:,:-1]!=states[method+'_global'][:,1:]).mean(1)
            for name in ['rms','mahal','angle','nll']:
                scores[method+'__final_'+name]=geo[f'{method}_28_{name}']
        print(json.dumps(dict(phase='scores',method=method,seconds=round(time.monotonic()-start,1))),flush=True)
    scores['baseline__length']=meta.n_tokens.to_numpy()
    scores['baseline__mean_token_entropy']=meta.entropy.to_numpy()
    scores['baseline__perplexity']=meta.perplexity.to_numpy()
    # Persist all unlabeled scores before any label-based comparison.
    scores.to_parquet(root/'unlabeled_scores.parquet',index=False)
    np.savez_compressed(root/'layer_scores.npz',**layer_scores)
    pd.DataFrame(gains).to_csv(root/'second_order_prediction.csv',index=False)
    metrics=evaluate_scores(meta,scores,controls,root)
    y=meta.label.to_numpy(int); tasks=[]
    for mi,method in enumerate(['gmm','mfa']):
        for ci,(control,c) in enumerate(controls.items()):
            for layer in range(27): tasks.append((method,control,layer,c,6200+mi*1000+ci*100+layer))
    def job(task):
        method,control,l,c,seed=task; z=states[method]
        r=ConditionalRouting(z[:,l],z[:,l+1],y,c).test(np.random.default_rng(seed),args.permutations)
        return dict(method=method,control=control,layer_from=l+1,layer_to=l+2,**r)
    tests=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,t) for t in tasks]):
            tests.append(f.result())
            if len(tests)%27==0: print(json.dumps(dict(phase='conditional_tests',done=len(tests),total=len(tasks),seconds=round(time.monotonic()-start,1))),flush=True)
    table=pd.DataFrame(tests).sort_values(['control','method','layer_from'])
    omnibus=[]
    for control,part in table.groupby('control'):
        table.loc[part.index,'fdr_q']=fdr(part.p)
        omnibus.append(dict(control=control,tests=len(part),global_bonferroni_p=min(1,float(part.p.min()*len(part))),
                             significant_layers=int((fdr(part.p)<.05).sum())))
    table.to_csv(root/'conditional_routing.csv',index=False)
    nulls={}
    for method in ['gmm','mfa']:
        for control in ['raw','task_length','full']:
            nulls[method+'__'+control]=occupancy_null(states[method],y,controls[control],np.random.default_rng(9071),args.occupancy_permutations)
            print(json.dumps(dict(phase='occupancy_null',method=method,control=control,seconds=round(time.monotonic()-start,1))),flush=True)
    save_json(root/'occupancy_nulls.json',nulls)
    edges,occupancy=edge_tables(states,y);edges.to_parquet(root/'edges.parquet',index=False);occupancy.to_parquet(root/'occupancy.parquet',index=False)
    sensitivity=[]
    for method,z in states.items():
        if method.endswith('_global'):continue
        reference=states['gmm' if method.startswith('gmm') else 'mfa']
        for l in range(28):
            row=dict(method=method,layer=l+1,k=len(np.unique(z[:,l])),ari_vs_reference=float(adjusted_rand_score(reference[:,l],z[:,l])))
            if method in ['gmm','mfa']:
                row.update(uncertain_fraction=float((geo[f'{method}_{l+1}_confidence']<.9).mean()),
                           nearest_disagreement=float((z[:,l]!=states[method+'_nearest'][:,l]).mean()))
            sensitivity.append(row)
    pd.DataFrame(sensitivity).to_csv(root/'sensitivity.csv',index=False)
    save_json(root/'summary.json',dict(protocol_sha256=file_digest(marker),samples=len(meta),correct=int(y.sum()),incorrect=int((1-y).sum()),
        seconds=time.monotonic()-start,omnibus=omnibus,completed=['P1 conditional route tests','P1 occupancy-preserving null','P2 nearest assignment','P2 existing seed and rank-specific-K sensitivity','P2 MFA pre-final sensitivity','Cross-fitted unlabeled scores and retrospective evaluation'],
        pending=['Same-K GMM/MFA and fixed-K rank controls','Fitted-map bootstrap','Continuous geometry-controlled route tests','Gaussian simulation null','Independent-fit P3','Task/model transfer','Prompt/prefix detection','Causal intervention'],
        source_manifest_sha256=file_digest(source/'manifest.json'),limitations=protocol['limitations']))
    print(json.dumps(dict(phase='complete',seconds=round(time.monotonic()-start,1),output=str(root))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--permutations',type=int,default=4999);p.add_argument('--occupancy-permutations',type=int,default=499)
    a=p.parse_args()
    with threadpool_limits(2):run(a)
