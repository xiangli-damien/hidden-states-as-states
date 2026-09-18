"""Independent identity, score, precision and reference-definition audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from hss.analysis.channel_data import load_config,ChannelData
from hss.analysis.confidence_study import partitions
from hss.analysis.confidence_bootstrap import auc_samples,auc_ci
from hss.experiments.artifacts import save_json,file_digest


def paired(y,a,b,cfg):
    delta=auc_samples(y,b,cfg['seed'],cfg['bootstrap'])-auc_samples(y,a,cfg['seed'],cfg['bootstrap'])
    return {'delta':float(roc_auc_score(y,b)-roc_auc_score(y,a)),
            'ci':np.quantile(delta,[.025,.975]).tolist()}


def run(cfg):
    cc=load_config(cfg['channel_config']);root=Path(cfg['output_root']);reports={}
    for ds in cc['datasets']:
        name=ds['name'];data=ChannelData(cc,name);rows=data.rows;out=root/name
        train,test,saved=partitions(cfg,name,rows);y=rows.y.to_numpy(int)
        predictions=pd.read_parquet(out/'predictions.parquet');scalars=pd.read_parquet(out/'scalars.parquet')
        precision=pd.read_parquet(out/'precision_scalars.parquet')
        assert np.array_equal(predictions.sample_id,rows.sample_id)
        assert np.array_equal(precision.sample_id,rows.sample_id)
        a=json.loads((out/'analysis.json').read_text());g=np.load(out/'readout_geometry.npz')
        results={};checks=[]
        for pos in ['prompt_last','t1']:
            view='pre_'+pos
            x=data.array('pre_prompt_last') if pos=='prompt_last' else ChannelData(cc,name,'tokens').array('pre',0)
            fitted=np.load(out/(view+'__full_direction.npz'));raw_score=x@fitted['coef']+fitted['intercept']
            error=float(np.max(np.abs(raw_score-predictions[view+'__full_direction'])))
            if error>1e-8:
                raise ValueError('Saved raw-coordinate probe fails reconstruction')
            check={'view':view,'score_reconstruction_max_error':error}
            if abs(roc_auc_score(y[test],raw_score[test])-a['views'][view]['models']['full_direction']['auc'])>1e-12:
                raise ValueError('Reported AUROC mismatch')
            checks.append(check)
            directions=np.load(out/(view+'_reference_directions.npz'))
            projections={'correctness_w':raw_score[test],'weakest_v':x[test]@g['v_min'],
                'entropy_direction':x[test]@directions['entropy_direction'],
                'measured_entropy':scalars[pos+'_entropy'].to_numpy()[test]}
            if 'difficulty_direction' in directions:
                projections['difficulty_direction']=x[test]@directions['difficulty_direction']
            corr=pd.DataFrame(projections).corr()
            corr.to_csv(out/(view+'_projection_correlations.csv'))
            results[view]={'projection_correlations':corr.to_dict(),
                'channel_vs_entropy_only':paired(y[test],-scalars.prompt_last_entropy.to_numpy()[test],predictions[view+'__channels16'].to_numpy()[test],cfg)}
        agreement=precision.policy_argmax.to_numpy()==rows.first_token.to_numpy()
        cohort=test[agreement[test]]
        entropy=-scalars.prompt_last_entropy.to_numpy()[cohort]
        results['argmax_agree_subset']={'n':len(cohort),'entropy_auc':float(roc_auc_score(y[cohort],entropy)),
            'ci':auc_ci(y[cohort],entropy,cfg['seed'],cfg['bootstrap']),
            'full_direction_auc':float(roc_auc_score(y[cohort],predictions.pre_prompt_last__full_direction.to_numpy()[cohort]))}
        # Orthogonality and numerical full-rank: no exact nullspace is asserted.
        err=float(np.max(np.abs(g['low_basis'].T@g['low_basis']-np.eye(g['low_basis'].shape[1]))))
        if err>1e-8:
            raise ValueError('Low-readout basis not orthonormal')
        results['validation']={'checks':checks,'unique_prompts':int(rows.prompt_sha256.nunique()),
            'train_test_overlap':0,'basis_orthogonality_max_error':err,
            'numerical_null_claim':False,'minimum_singular_value':float(g['singular_values'][0]),
            'source_cache_identity':data.info['source_keys'],
            'artifacts_sha256':{p.name:file_digest(p) for p in [out/'samples.parquet',out/'scalars.parquet',out/'readout_geometry.npz',out/'predictions.parquet',out/'analysis.json']}}
        save_json(out/'audit.json',results);reports[name]=results['validation']
    save_json(root/'validation.json',{'datasets':reports,'code_sha256':file_digest(__file__),
        'reference':{'repository':'https://github.com/xiangli-damien/terminal-readout',
            'branch':'codex/pilot-infrastructure','commit':'0ac9b89ee6318cfbd3a1107f76c5a3f68a5d71ca',
            'document':'docs/readout_geometry_zh.md','document_sha256':'db2b53d09f9142053ef351d9bf7512cfa02b88e1775a8a488a48e0c572a92957',
            'code':'src/terminal_readout/readout_geometry.py','code_sha256':'f6e95d638d13998a35dae1ba85fea3155b3024bff59cb9871c4b7641e35efe57',
            'definition':'Smallest right singular vector of vocab-centered U diag(gamma). Same definition, independently recomputed for each model.',
            'initial_main_only_inspection_corrected':True}})
