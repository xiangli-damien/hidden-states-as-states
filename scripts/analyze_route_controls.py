"""Matched-K, geometric, seed and edge-preserving controls for frozen routes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import adjusted_rand_score

from hss.experiments.artifacts import file_digest,save_json
from hss.route.counts import crossfit_scores,group_folds
from hss.route.nulls import ConditionalRouting,codes,occupancy_null,shuffle_suffixes
from run_route_study import nuisance_codes,quantile_bins,evaluate_scores,fdr


def run(args):
    start=time.monotonic();source=Path(args.inputs);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(source/'rows.parquet');states=dict(np.load(source/'states.npz'));geo=dict(np.load(source/'geometry.npz'))
    folds=group_folds(meta.question_group);controls=nuisance_codes(meta)
    marker=root/'protocol.json'
    if marker.exists():raise ValueError('Use a fresh output folder')
    save_json(marker,dict(stage='Exploratory P2 controls',created_at=time.time(),source=file_digest(source/'manifest.json'),
        driver=file_digest(Path(__file__)),permutations=4999,suffix_permutations=args.suffix_permutations,
        suffix_null='Whole-suffix exchange at common middle states within each fold; preserves every adjacent edge table in each fold, without correctness labels.',
        added_controls='Task/length + median RMS + median within-component Mahalanobis distance; full controls + token-entropy median.',
        occupancy_null_amendment='Retain raw JS comparison and additionally subtract its exact conditional-label null expectation in each original/shuffled map; shuffling changes occupied-cell sparsity.',
        main_model_matching='Unchanged. Auxiliary fitted GMM shares each layer K with MFA; no member-overlap rematching.',
        limitations=['Whole-data frozen maps, repeated exploration, coarse nuisance bins, sparse common support.',
          'Matched-K changes both covariance family and independently estimated centers; not a pure intervention on covariance.',
          'Auxiliary rank controls use their own K; fixed-K MFA rank and resampling remain pending.']))
    matched=[];probabilities=[];fitrows=[]
    for l in range(1,29):
        p=Path(args.matched)/f'layer_{l}';c=json.loads((p/'complete.json').read_text())
        if not c['converged'] or c['n_init_completed']!=3:raise ValueError('Incomplete matched fit')
        if file_digest(p/'model.npz')!=c['model_sha256']:raise ValueError('Changed matched fit')
        d=dict(np.load(p/'diagnostics.npz'));matched.append(d['posterior']);probabilities.append(d['probability'])
        fitrows.append(dict(layer=l,k=c['k'],seconds=c['seconds'],converged_restarts=sum(r['converged'] for r in c['restarts']),
                           ari_vs_mfa=float(adjusted_rand_score(d['posterior'],states['mfa'][:,l-1]))))
        for name in ['rms','mahal','confidence']:geo[f'gmm_matched_{l}_{name}']=d[name]
    states['gmm_matched']=np.column_stack(matched)
    np.savez_compressed(root/'matched_states.npz',states=states['gmm_matched'])
    pd.DataFrame(fitrows).to_csv(root/'matched_fits.csv',index=False)
    scores=pd.DataFrame(dict(sample_id=meta.sample_id,fold=folds));s,_=crossfit_scores(states['gmm_matched'],folds,probabilities)
    gain=s.pop('second_order_predictive_gain')
    for name,v in s.items():scores['gmm_matched__'+name]=v
    scores['baseline__length']=meta.n_tokens.to_numpy();scores.to_parquet(root/'unlabeled_scores.parquet',index=False)
    evaluate_scores(meta,scores,controls,root)
    # Correctness is used only below and in evaluation above.
    y=meta.label.to_numpy(int);tasks=[]
    for mi,method in enumerate(['gmm','mfa','gmm_matched','mfa_seed1042','mfa_seed2042']):
        z=states[method]
        for l in range(27):
            tasks.append((method,'full',l,controls['full'],51000+1000*mi+l))
            if method in ['gmm','mfa','gmm_matched']:
                geometry=codes(controls['task_length'],quantile_bins(geo[f'{method}_{l+1}_rms'],2),quantile_bins(geo[f'{method}_{l+1}_mahal'],2))
                tasks.append((method,'geometry',l,geometry,61000+1000*mi+l))
                entropy=codes(controls['full'],quantile_bins(meta.entropy,2))
                tasks.append((method,'entropy',l,entropy,71000+1000*mi+l))
    def job(task):
        method,control,l,c,seed=task;z=states[method]
        t=ConditionalRouting(z[:,l],z[:,l+1],y,c).test(np.random.default_rng(seed),4999)
        return dict(method=method,control=control,layer_from=l+1,layer_to=l+2,**t)
    tests=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,t) for t in tasks]):
            tests.append(f.result())
            if len(tests)%54==0:print(json.dumps(dict(phase='controls',done=len(tests),total=len(tasks),seconds=time.monotonic()-start)),flush=True)
    table=pd.DataFrame(tests)
    for control,p in table.groupby('control'):table.loc[p.index,'fdr_q']=fdr(p.p)
    table.to_csv(root/'conditional_controls.csv',index=False)
    memory={}
    for method in ['gmm','mfa','gmm_matched']:
        z=states[method];observed=crossfit_scores(z,folds)[0]['second_order_predictive_gain'].mean()
        rng=np.random.default_rng(69123);null=[]
        for b in range(args.suffix_permutations):
            swapped=shuffle_suffixes(z,folds,rng)
            null.append(float(crossfit_scores(swapped,folds)[0]['second_order_predictive_gain'].mean()))
        memory[method]=dict(observed_gain_nats=float(observed),null_mean_nats=float(np.mean(null)),null_95=np.quantile(null,[.025,.975]).tolist(),
                            excess_nats=float(observed-np.mean(null)),p=float((1+(np.array(null)>=observed).sum())/(1+len(null))),permutations=len(null))
        print(json.dumps(dict(phase='suffix_null',method=method,result=memory[method],seconds=time.monotonic()-start)),flush=True)
        save_json(root/'memory_null.json',memory)
    corrected_null={}
    for method in ['gmm','mfa','gmm_matched']:
        corrected_null[method]=occupancy_null(states[method],y,controls['full'],np.random.default_rng(9071),499,bias_correct=True)
        save_json(root/'corrected_occupancy_nulls.json',corrected_null)
        print(json.dumps(dict(phase='bias_corrected_occupancy',method=method,seconds=time.monotonic()-start)),flush=True)
    save_json(root/'summary.json',dict(complete=True,seconds=time.monotonic()-start,layers=28,matched_gmm_initializations=84,
        all_selected_converged=True,scope='P2 matched-K, covariance/entropy strata, seed replication and unlabeled suffix null; not independent-fit evaluation',
        memory=memory,corrected_occupancy_nulls=corrected_null))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True);p.add_argument('--matched',required=True);p.add_argument('--output',required=True);p.add_argument('--suffix-permutations',type=int,default=199)
    a=p.parse_args()
    with threadpool_limits(2):run(a)
