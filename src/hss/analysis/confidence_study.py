"""Discovery-only direction fits and paired held-out confidence comparisons."""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss, r2_score
from sklearn.model_selection import StratifiedKFold

from hss.analysis.channel_data import ChannelData, load_config
from hss.analysis.channel_study import auc_ci, paired_auc_delta_ci
from hss.analysis.confidence_data import layer_array
from hss.experiments.artifacts import save_json, file_digest, runtime_versions


def partitions(cfg, name, rows):
    saved = pd.read_parquet(Path(cfg['prior_root'])/name/'samples.parquet')
    if not np.array_equal(saved.sample_id,rows.sample_id) or not np.array_equal(saved.y,rows.y):
        raise ValueError('Prior study identity/order mismatch')
    if not np.array_equal(saved.prompt_sha256,rows.prompt_sha256):
        raise ValueError('Prompt identity mismatch')
    train = np.flatnonzero(saved.partition.eq('discovery'))
    test = np.flatnonzero(saved.partition.eq('confirmation'))
    if len(train)+len(test)!=len(rows) or set(saved.prompt_sha256.iloc[train]) & set(saved.prompt_sha256.iloc[test]):
        raise ValueError('Invalid/overlapping partitions')
    return train,test,saved


def control_features(rows,x,train,include_opening=False):
    values = []
    names = []
    fields = ['category','level']+(['first_token'] if include_opening else [])
    for field in fields:
        v = rows[field].astype(str)
        counts = v.iloc[train].value_counts()
        known = counts[counts >= (20 if field=='first_token' else 1)].index
        v = v.where(v.isin(known),'__other__')
        for label in sorted(v.iloc[train].unique())[1:]:
            values.append(v.eq(label).to_numpy(float)); names.append(field+':'+label)
    rms = np.sqrt(np.mean(x.astype(float)**2,axis=1))
    values += [np.log1p(rows.n_prompt_tokens.to_numpy(float)),np.log(np.maximum(rms,1e-12))]
    names += ['log_prompt_length','log_rms']
    return np.column_stack(values),names


def fit_fixed(x,y,train,c,global_scale=False):
    mu = x[train].mean(0)
    sd = x[train].std(0)
    if global_scale:
        sd[:] = np.sqrt(np.mean(sd**2))
    sd = np.maximum(sd,1e-8)
    z = (x-mu)/sd
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LogisticRegression(C=c,max_iter=3000,tol=1e-6,solver='lbfgs').fit(z[train],y[train])
    coef = model.coef_[0]/sd
    intercept = float(model.intercept_[0]-mu@coef)
    return {'coef':coef,'intercept':intercept,'c':float(c),'mean':mu,'std':sd,
            'score':x@coef+intercept,'iterations':int(model.n_iter_[0])}


def tuned_fit(x,y,train,cfg,global_scale=False):
    x = np.asarray(x,dtype=np.float64)
    folds = list(StratifiedKFold(cfg['inner_folds'],shuffle=True,random_state=cfg['seed']).split(train,y[train]))
    cv = []
    for c in cfg['c_grid']:
        aucs = []
        for tr,va in folds:
            fit = fit_fixed(x,y,train[tr],c,global_scale)
            aucs.append(roc_auc_score(y[train[va]],fit['score'][train[va]]))
        cv.append({'c':c,'mean_auc':float(np.mean(aucs)),'fold_auc':aucs})
    best = max(range(len(cv)),key=lambda i:cv[i]['mean_auc'])
    fit = fit_fixed(x,y,train,cv[best]['c'],global_scale)
    fit['cv'] = cv
    return fit


def summary_score(y,score,cfg):
    return {'auc':float(roc_auc_score(y,score)),
            'ci':auc_ci(y,score,cfg['seed'],cfg['bootstrap']),'n':len(y)}


def cosine(a,b):
    return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))


def ridge_direction(x,target,train):
    mu = x[train].mean(0); sd = np.maximum(x[train].std(0),1e-8)
    model = Ridge(alpha=100.).fit((x[train]-mu)/sd,target[train])
    return np.atleast_2d(model.coef_)/sd,model.predict((x-mu)/sd)


def direction_geometry(x,rows,train,test,w,v,entropy,out):
    e,ep = ridge_direction(x,entropy,train)
    category = pd.get_dummies(rows.category).to_numpy(float)
    category_coef,_ = ridge_direction(x,category,train)
    _,s,vt = np.linalg.svd(category_coef,full_matrices=False)
    basis = vt[s>s[0]*1e-8]
    result = {'cosine_v_min':cosine(w,v),'cosine_entropy_regression':cosine(w,e[0]),
        'entropy_regression_test_r2':float(r2_score(entropy[test],ep[test])),
        'category_subspace_rank':len(basis),
        'category_subspace_w_fraction':float(np.linalg.norm(basis@w)/np.linalg.norm(w)),
        'category_note':'Multiclass coefficient subspace; no single category direction.'}
    arrays = {'entropy_direction':e[0],'category_basis':basis}
    if rows.level.nunique()>1:
        difficulty = pd.to_numeric(rows.level).to_numpy(float)
        d,dp = ridge_direction(x,difficulty,train)
        result.update({'cosine_difficulty':cosine(w,d[0]),
                       'difficulty_test_r2':float(r2_score(difficulty[test],dp[test]))})
        arrays['difficulty_direction'] = d[0]
    np.savez(out,**arrays)
    return result


def analyse(cfg):
    cc = load_config(cfg['channel_config']); root = Path(cfg['output_root'])
    for ds in cc['datasets']:
        name = ds['name']; data = ChannelData(cc,name); rows = data.rows
        out = root/name; train,test,saved = partitions(cfg,name,rows)
        saved.to_parquet(out/'samples.parquet',index=False)
        scalars = pd.read_parquet(out/'scalars.parquet')
        assert np.array_equal(scalars.sample_id,rows.sample_id)
        y = rows.y.to_numpy(int)
        prior = json.loads((Path(cfg['prior_root'])/name/'probes.json').read_text())
        geometry = np.load(out/'readout_geometry.npz')
        results = {'dataset':name,'n':len(rows),'train_n':len(train),'test_n':len(test),
                   'scalars':[],'views':{},'adaptation':'Previously inspected holdout; exploratory follow-up.'}
        predictions = rows[['sample_id','y']].copy()
        for position in ['prompt_last','t1']:
            for metric in ['entropy','logit_margin','probability_margin','top1_probability','rms',
                           'v_min_projection','v_min_absolute','v_min_unit_projection','low_readout_fraction']:
                col = position+'_'+metric; value = scalars[col].to_numpy()
                sign = 1 if roc_auc_score(y[train],value[train])>=.5 else -1
                record = {'position':position,'metric':metric,'discovery_sign':sign,
                    'raw_auc':float(roc_auc_score(y[test],value[test])),
                    **summary_score(y[test],sign*value[test],cfg)}
                results['scalars'].append(record)
        for position in ['prompt_last','t1']:
            view = 'pre_'+position
            x = data.array('pre_prompt_last') if position=='prompt_last' else ChannelData(cc,name,'tokens').array('pre',0)
            x = x.astype(np.float64)
            indices = next(r['indices'] for r in prior[view] if r.get('kind')=='middle_channels' and r.get('k')==16)
            controls,names = control_features(rows,x,train,False)
            diagnostic,dnames = control_features(rows,x,train,True)
            entropy = scalars[['prompt_last_entropy']].to_numpy()
            current_entropy = scalars[[position+'_entropy']].to_numpy()
            confidence = scalars[[position+'_'+m for m in ['entropy','logit_margin','top1_probability','v_min_projection','low_readout_fraction']]].to_numpy()
            blocks = {
                'channels16':x[:,indices],
                'nuisance':controls,
                'nuisance_entropy':np.column_stack([controls,entropy]),
                'nuisance_entropy_channels':np.column_stack([controls,entropy,x[:,indices]]),
                'diagnostic_nuisance':diagnostic,
                'diagnostic_nuisance_entropy':np.column_stack([diagnostic,entropy]),
                'diagnostic_nuisance_entropy_channels':np.column_stack([diagnostic,entropy,x[:,indices]]),
                'nuisance_confidence':np.column_stack([controls,confidence]),
                'nuisance_confidence_channels':np.column_stack([controls,confidence,x[:,indices]]),
                'nuisance_confidence_full':np.column_stack([controls,confidence,x]),
                'full_direction':x,
                'full_direction_global_scale':x,
            }
            if position=='t1':
                blocks['nuisance_current_entropy'] = np.column_stack([controls,current_entropy])
                blocks['nuisance_current_entropy_channels'] = np.column_stack([controls,current_entropy,x[:,indices]])
            scores = {}; records = {}; fits = {}
            for kind,features in blocks.items():
                fit = tuned_fit(features,y,train,cfg,kind=='full_direction_global_scale')
                score = fit['score']; scores[kind] = score; fits[kind] = fit
                record = {**summary_score(y[test],score[test],cfg),'c':fit['c'],'cv':fit['cv'],
                          'log_loss':float(log_loss(y[test],expit(score[test])))}
                records[kind] = record
                predictions[view+'__'+kind] = score
                np.savez(out/(view+'__'+kind+'.npz'),**{k:fit[k] for k in ['coef','intercept','mean','std','c']})
                print(json.dumps({'dataset':name,'view':view,'fit':kind,'auc':record['auc'],'c':fit['c']}),flush=True)
            comparisons = [('nuisance','nuisance_entropy'),('nuisance_entropy','nuisance_entropy_channels'),
                ('diagnostic_nuisance_entropy','diagnostic_nuisance_entropy_channels'),
                ('nuisance_confidence','nuisance_confidence_channels'),('nuisance_confidence','nuisance_confidence_full')]
            if position=='t1':
                comparisons.append(('nuisance_current_entropy','nuisance_current_entropy_channels'))
            increments = []
            for base,aug in comparisons:
                ci = paired_auc_delta_ci(y[test],scores[base][test],scores[aug][test],cfg['seed'],cfg['bootstrap'])
                eq = cfg['equivalence_auc']
                decision = ('within_practical_equivalence_bound' if ci[0]>-eq and ci[1]<eq else
                            'positive_increment' if ci[0]>0 else 'negative_increment' if ci[1]<0 else 'inconclusive')
                increments.append({'base':base,'augmented':aug,'delta':records[aug]['auc']-records[base]['auc'],
                                   'ci':ci,'decision':decision,'equivalence_bound':eq})
            w = fits['full_direction']['coef']
            dg = direction_geometry(x,rows,train,test,w,geometry['v_min'],current_entropy[:,0],out/(view+'_reference_directions.npz'))
            dg['cosine_standardized_vs_global_scale_probe'] = cosine(w,fits['full_direction_global_scale']['coef'])
            results['views'][view] = {'models':records,'incremental':increments,'geometry':dg,
                'historical_channels':indices,'nuisance_columns':names,'diagnostic_columns':dnames}
        predictions.to_parquet(out/'predictions.parquet',index=False)
        save_json(out/'analysis.json',results)
    transfer(cfg)
    save_json(root/'analysis.json',{'config':cfg,'code_sha256':file_digest(__file__),'versions':runtime_versions(),
        'datasets':[ds['name'] for ds in cc['datasets']],
        'causal_experiment':{'status':'not_run','reason':'Conditional upstream intervention requires established target and free GPU capacity; collectors remain active.'}})


def transfer(cfg):
    root = Path(cfg['output_root']); cc = load_config(cfg['channel_config'])
    source,target = 'llama32_math','llama32_mmlu'
    sr = json.loads((root/source/'analysis.json').read_text())
    td = ChannelData(cc,target); _,test,_ = partitions(cfg,target,td.rows)
    ts = pd.read_parquet(root/target/'scalars.parquet'); y = td.rows.y.to_numpy(int)
    result = []
    for r in sr['scalars']:
        col = r['position']+'_'+r['metric']; score = ts[col].to_numpy()*r['discovery_sign']
        result.append({'position':r['position'],'metric':r['metric'],'source_sign':r['discovery_sign'],
                       'source_auc':r['auc'],**summary_score(y[test],score[test],cfg)})
    for pos in ['prompt_last','t1']:
        x = td.array('pre_prompt_last') if pos=='prompt_last' else ChannelData(cc,target,'tokens').array('pre',0)
        for kind in ['full_direction','full_direction_global_scale','channels16']:
            fit = np.load(root/source/('pre_'+pos+'__'+kind+'.npz'))
            if kind=='channels16':
                features = x[:,sr['views']['pre_'+pos]['historical_channels']]
            else:
                features = x
            score = features@fit['coef']+fit['intercept']
            result.append({'position':pos,'metric':kind,'source_sign':1,
                           'source_auc':sr['views']['pre_'+pos]['models'][kind]['auc'],
                           **summary_score(y[test],score[test],cfg)})
    save_json(root/'transfer.json',{'source':source,'target':target,'target_labels_used_for_fitting':False,'records':result})


def layer_analysis(cfg):
    cc = load_config(cfg['channel_config']); root = Path(cfg['output_root'])
    for ds in cc['datasets']:
        name = ds['name']
        if name not in cfg['layer_datasets']:
            continue
        data = ChannelData(cc,name); rows = data.rows; y = rows.y.to_numpy(int)
        tr,te,_ = partitions(cfg,name,rows)
        enough = rows.n_tokens.to_numpy()>=cfg['layer_position']
        train,test = tr[enough[tr]],te[enough[te]]
        out = root/name
        frozen = {pos:np.load(out/('pre_'+pos+'__full_direction.npz')) for pos in ['prompt_last','t1']}
        records = []
        n_layers = data.info['model']['n_layers']
        for pos in ['prompt_last','t16']:
            for layer in range(1,n_layers):
                if pos=='prompt_last':
                    x = data.array('pre_prompt_last') if layer==n_layers-1 else data.array('prompt_last',layer)
                else:
                    x = layer_array(cfg,data,name,layer)
                for source,fit in frozen.items():
                    score = x[test]@fit['coef']+fit['intercept']
                    records.append({'position':pos,'layer':layer,'method':'frozen_'+source,
                                    **summary_score(y[test],score,cfg)})
                # No new hyperparameter search at each layer: reuse discovery-selected prompt C.
                fit = fit_fixed(x.astype(float),y,train,float(frozen['prompt_last']['c']))
                records.append({'position':pos,'layer':layer,'method':'layer_specific_probe',
                    **summary_score(y[test],fit['score'][test],cfg),'train_n':len(train),'c':fit['c']})
                print(json.dumps({'dataset':name,'position':pos,'layer':layer,'auc':records[-1]['auc']}),flush=True)
        save_json(out/'layers.json',{'same_cohort_n':len(test),'train_n':len(train),'position':cfg['layer_position'],
             'final_layer':'pre_RMS','records':records,
             'limit':'Frozen readout failure does not prove erasure; layer-specific fits are auxiliary. Prompt states remain causally unchanged after generation and do not prove attention retrieval.'})
