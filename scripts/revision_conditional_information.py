"""Does a generated-prefix representation add beyond the initial prompt state?

Fits direct joint readouts, never stacking in-sample supervised predictions.
Every C candidate is trained only on train; validation log loss selects C;
test outcomes are only used for final, paired question-level evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from fit_revision_geometry import read_view
from revision_common import config,freeze,provenance,write_json,write_npz,sha,status,nearest
from revision_statistics import paired_auc_ci


def fit_readout(x_train,y_train,x_validation,y_validation,x_all,path,grid,max_iter,seed):
    """No test-label argument, including during candidate selection."""
    path.mkdir(parents=True,exist_ok=True)
    trials=[];best=None
    for c in grid:
        start=time.monotonic()
        model=LogisticRegression(C=c,solver='lbfgs',max_iter=max_iter,tol=1e-5,random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            model.fit(x_train,y_train)
            initial_iterations=int(model.n_iter_.max())
            if initial_iterations>=max_iter:
                model.set_params(warm_start=True,max_iter=2*max_iter).fit(x_train,y_train)
        converged=int(model.n_iter_.max())<model.max_iter
        row={'C':c,'converged':converged,'initial_iterations':initial_iterations,
             'last_iterations':int(model.n_iter_.max()),'seconds':time.monotonic()-start,
             'validation_log_loss':float(log_loss(y_validation,model.predict_proba(x_validation)[:,1])),
             'n_features':x_train.shape[1]}
        trials.append(row)
        write_npz(path/f'C_{c:g}.npz',coef=model.coef_,intercept=model.intercept_)
        write_json(path/f'C_{c:g}.json',row)
        if converged and (best is None or row['validation_log_loss']<best[0]['validation_log_loss']):
            best=(row,model)
    if best is None:
        raise RuntimeError('No converged supervised readout; refusing partial comparison')
    write_json(path/'selection.json',{'selected':best[0],'trials':trials,
                                     'criterion':'validation log loss, no test-label fitting'})
    return best[1].predict_proba(x_all)[:,1],best[0]


def question_text(text):
    prefix='Question: '
    instruction='\nPlease reason step by step, and put your final answer within \\boxed{}.'
    if not text.startswith(prefix) or not text.endswith(instruction):
        raise ValueError('Unexpected frozen MATH prompt format')
    return text[len(prefix):-len(instruction)]


def run_one(args):
    cfg,prefix,layer,view=args
    with threadpool_limits(limits=cfg['blas_threads']):
        return _run_one(cfg,prefix,layer,view)


def _run_one(cfg,prefix,layer,view):
    root=Path(cfg['source_root']);dest=Path(cfg['output'])/f'p{prefix}_l{layer}_{view}'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'_SUCCESS.json').exists():
        return json.loads((dest/'summary.json').read_text())
    # The source config preserves the actual [7,14,21,28] tensor axis mapping.
    source_cfg=json.loads((root/'prefixes/plan.json').read_text())['config']
    initial,prompt=read_view(source_cfg,0,layer,'last')
    current,late=read_view(source_cfg,prefix,layer,view)
    index=initial.set_index('sample_id').index.get_indexer(current.sample_id)
    if np.any(index<0) or current.sample_id.duplicated().any():
        raise ValueError('Question identity mismatch')
    prompt=prompt[index];initial=initial.iloc[index].reset_index(drop=True)
    np.testing.assert_array_equal(initial.label,current.label)
    np.testing.assert_array_equal(initial.split,current.split)
    np.testing.assert_array_equal(initial.question_group,current.question_group)
    train=current.split.eq('train').to_numpy();val=current.split.eq('validation').to_numpy();test=current.split.eq('test').to_numpy()
    y=1-current.label.to_numpy(int)
    if set(current.split)!=set(['train','validation','test']) or current.question_group.duplicated().any():
        raise ValueError('Invalid fixed question-level splits')
    # Clustering remains raw. These scalers belong only to supervised readouts.
    ps=StandardScaler().fit(prompt[train]);ls=StandardScaler().fit(late[train])
    p=ps.transform(prompt);l=ls.transform(late)
    confidence=pd.read_parquet(root/'confidence'/f'prefix_{prefix}.parquet').set_index('sample_id').loc[current.sample_id]
    numeric=np.column_stack([np.log1p(current.n_prompt_tokens.to_numpy(float)),
        np.linalg.norm(prompt,axis=1),np.linalg.norm(late,axis=1),
        confidence.next_token_entropy.to_numpy(),confidence.next_token_logit_margin.to_numpy()])
    ns=StandardScaler().fit(numeric[train])
    categorical=current[['category','level']].fillna('unknown').astype(str)
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(categorical[train])
    controls=np.column_stack([ns.transform(numeric),encoder.transform(categorical)]).astype(np.float32)
    with np.load(root/'geometry'/f'p{prefix}_l{layer}_{view}'/'decoder.npz') as decoder:
        centers=decoder['centers']
    codes=nearest(late,centers);state=np.eye(len(centers),dtype=np.float32)[codes]
    texts=current.prompt_text.map(question_text)
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=20000,sublinear_tf=True,
                              token_pattern=r'(?u)\b\w+\b',dtype=np.float64)
    vectorizer.fit(texts[train]);lexical=vectorizer.transform(texts)
    write_json(dest/'feature_metadata.json',{
        'prompt':'raw same-block prompt-last, standardized by train statistics for LR',
        'current':f'raw prefix{prefix} {view}, standardized by train statistics for LR',
        'nuisance':['log prompt length','prompt norm','current representation norm','current next-token entropy',
                    'current next-token margin','category','difficulty'],
        'categorical_levels':[c.tolist() for c in encoder.categories_],
        'lexical':'question-only word/bigram TF-IDF; lexical baseline, not a pretrained semantic encoder',
        'vocabulary':{k:int(v) for k,v in vectorizer.vocabulary_.items()},
        'no_future_answer_length':True,'no_in_sample_stacking':True,
        'state_assignment':'nearest GMM center of current representation; train-only map',
        'state_K':len(centers)})
    write_npz(dest/'feature_transforms.npz',prompt_mean=ps.mean_,prompt_scale=ps.scale_,
        current_mean=ls.mean_,current_scale=ls.scale_,control_mean=ns.mean_,control_scale=ns.scale_,
        tfidf_idf=vectorizer.idf_,centers=centers)
    # Duplicate-prompt comparison exposes improvements arising just from a
    # different effective L2 penalty after doubling the feature count.
    families={
        'prompt_only':p,
        'current_only':l,
        'prompt_duplicate':np.column_stack([p,p]),
        'prompt_plus_current':np.column_stack([p,l]),
        'prompt_plus_controls':np.column_stack([p,controls]),
        'prompt_plus_current_plus_controls':np.column_stack([p,l,controls]),
        'prompt_plus_state_plus_controls':np.column_stack([p,state,controls]),
        'question_tfidf_plus_controls':sparse.hstack([lexical,sparse.csr_matrix(controls)],format='csr')}
    predictions=pd.DataFrame({'sample_id':current.sample_id,'question_group':current.question_group,
        'split':current.split,'failure':y})
    metrics=[]
    for name,x in families.items():
        prob,selection=fit_readout(x[train],y[train],x[val],y[val],x,dest/name,
                                  cfg['C_grid'],cfg['max_iter'],cfg['seed'])
        predictions[name]=prob
        metrics.append({'method':name,'test_auroc':float(roc_auc_score(y[test],prob[test])),
                        'test_log_loss':float(log_loss(y[test],prob[test])),
                        'test_brier':float(brier_score_loss(y[test],prob[test])),
                        'selected_C':selection['C'],'n_features':x.shape[1]})
    comparisons=[]
    pairs=[('current_only','prompt_only'),('prompt_plus_current','prompt_only'),
           ('prompt_duplicate','prompt_only'),('prompt_plus_current','prompt_duplicate'),
           ('prompt_plus_current_plus_controls','prompt_plus_controls'),
           ('prompt_plus_state_plus_controls','prompt_plus_controls'),
           ('prompt_plus_current_plus_controls','question_tfidf_plus_controls')]
    for name,base in pairs:
        comparisons.append({'method':name,'baseline':base,
            **paired_auc_ci(y[test],predictions[name].to_numpy()[test],predictions[base].to_numpy()[test])})
    predictions.to_parquet(dest/'predictions.parquet',index=False)
    result={'prefix':prefix,'block':layer,'view':view,'split_counts':current.split.value_counts().to_dict(),
            'metrics':metrics,'paired_comparisons':comparisons,
            'scope':'Exploratory reused MATH test; conditional prediction, not causality or conditional mutual information',
            'normalization':'none for GMM; train-only feature scaling for linear readout',
            'selection':'validation log loss; no test-label fitting/selection'}
    write_json(dest/'summary.json',result)
    write_json(dest/'_SUCCESS.json',{'summary_sha256':sha(dest/'summary.json'),'predictions_sha256':sha(dest/'predictions.parquet')})
    print(json.dumps({'conditional_complete':dest.name}),flush=True);return result


def run(cfg):
    source=Path(cfg['source_root']);root=Path(cfg['output'])
    files=[Path(__file__),Path(__file__).with_name('revision_statistics.py'),source/'prefixes/plan.json',source/'geometry_summary.json']
    freeze(root/'plan.json',provenance(cfg,files))
    jobs=[(cfg,*j) for j in cfg['jobs']];results=[]
    status(root,'fit',state='running',completed=0,expected=len(jobs))
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for future in as_completed([pool.submit(run_one,j) for j in jobs]):
            results.append(future.result());status(root,'fit',state='running',completed=len(results),expected=len(jobs))
    write_json(root/'summary.json',results);status(root,'fit',state='complete',completed=len(results),expected=len(jobs))
    write_json(root/'_SUCCESS.json',{'views':len(results),'scope':cfg['scope']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    cfg=config(parser.parse_args().config)
    try:
        run(cfg)
    except BaseException:
        status(cfg['output'],'fit',state='failed',traceback=traceback.format_exc());raise
