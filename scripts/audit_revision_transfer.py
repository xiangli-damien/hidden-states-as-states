"""Reconstruct transfer predictions from frozen features/maps/readouts and audit all views.

No fitting or model selection occurs here. The prior-only score is a descriptive
calibration control fitted on adaptation labels only; it changes no saved result.
"""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from threadpoolctl import threadpool_limits
from fit_revision_geometry import read_view
from revision_common import sha, write_json, nearest, paired_ratio_ci
from revision_statistics import paired_auc_ci
from revision_transfer import ROOT, SOURCE, density, nb_fit, nb_predict


def check(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)


def check_ci(a,b):
    for k in ['estimate','ci95'] if 'estimate' in a else ['n','auroc','baseline_auroc','delta','ci95']:
        check(a[k],b[k])


def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def selected_map(path,ids,hashes):
    selection=json.loads((path/'selection.json').read_text());trials=selection['trials']
    assert len(trials)==12 and {(x['k'],x['seed']) for x in trials}=={(k,s) for k in [1,4,8,16,32,64] for s in [42,137]}
    for row in trials:
        assert row==json.loads((path/f'k{row["k"]}_seed{row["seed"]}.json').read_text())
        assert row['fit_samples']==len(ids) and np.isfinite(row['icl'])
    chosen=min((x for x in trials if x['converged']),key=lambda x:x['icl'])
    assert chosen==selection['selected'] and selection['K_at_upper_boundary']==(chosen['k']==64)
    candidate=load(path/f'k{chosen["k"]}_seed{chosen["seed"]}.npz');model=load(path/'selected.npz')
    for key in model:np.testing.assert_array_equal(model[key],candidate[key])
    assert (model['variances']>0).all() and (model['weights']>0).all();check(model['weights'].sum(),1.)
    for p in path.glob('*.json'):hashes[str(p)]=sha(p)
    for p in path.glob('*.npz'):hashes[str(p)]=sha(p)
    return model, len(trials), sum(r['converged'] for r in trials)


def run():
    plan=json.loads((ROOT/'transfer_plan.json').read_text());cfg=plan['config']
    assert json.loads((ROOT/'transfer_status.json').read_text())['state']=='complete'
    for filename,digest in plan['files'].items():assert sha(filename)==digest,(filename,'provenance changed')
    source_cfg,target_cfg=cfg['source_config'],cfg['target_config'];hashes={};flat=[];counts=[0,0];priors=[]
    expected={f'p{p}_l{l}_{v}' for p in cfg['prefixes'] for l in cfg['layers'] for v in cfg['views']}
    assert {p.parent.name for p in (ROOT/'transfer').glob('*/_SUCCESS.json')}==expected
    for name in sorted(expected):
        dest=ROOT/'transfer'/name;s=json.loads((dest/'summary.json').read_text())
        assert sha(dest/'summary.json')==json.loads((dest/'_SUCCESS.json').read_text())['summary_sha256']
        p,l,v=s['prefix'],s['block'],s['view'];source,sx=read_view(source_cfg,p,l,v);target,tx=read_view(target_cfg,p,l,v)
        fit_ids=json.loads((dest/'fit_question_ids.json').read_text())
        st=source.split.eq('train').to_numpy();sv=source.split.eq('validation').to_numpy()
        tt=target.split.eq('train').to_numpy();tv=target.split.eq('validation').to_numpy();test=target.split.eq('test').to_numpy()
        assert fit_ids['target_adaptation']==target.loc[tt,'sample_id'].tolist()
        assert fit_ids['target_validation']==target.loc[tv,'sample_id'].tolist()
        assert not set(fit_ids['target_adaptation'])&set(target.loc[test,'sample_id'])
        import hashlib
        ordered=sorted(source.loc[st,'sample_id'],key=lambda sid:hashlib.sha256(('source-match-v1/'+sid).encode()).hexdigest())
        sm=source.sample_id.isin(ordered[:int(tt.sum())]).to_numpy()
        assert fit_ids['source_matched']==source.loc[sm,'sample_id'].tolist() and sm.sum()==tt.sum()
        matched,n,c=selected_map(dest/'source_matched_map',fit_ids['source_matched'],hashes);counts[0]+=n;counts[1]+=c
        adapted,n,c=selected_map(dest/'target_map',fit_ids['target_adaptation'],hashes);counts[0]+=n;counts[1]+=c
        origin=SOURCE/'geometry'/name;fixed=load(origin/'decoder.npz');readout=load(origin/'readout.npz')
        sy=1-source.label.to_numpy(int);ty=1-target.label.to_numpy(int)
        pred=pd.read_parquet(dest/'predictions.parquet')
        for key,values in [('sample_id',target.sample_id),('split',target.split),('failure',ty)]:np.testing.assert_array_equal(pred[key],values)
        assert not pred.sample_id.duplicated().any()
        confidence=pd.read_parquet(ROOT/'confidence'/f'prefix_{p}.parquet').set_index('sample_id').loc[target.sample_id]
        check(pred.entropy,confidence.next_token_entropy)
        check(pred.frozen_source_linear,expit(((tx-readout['scaler_mean'])/readout['scaler_scale'])@readout['linear_coef'].ravel()+readout['linear_intercept'][0]))
        maps={'frozen_source_full':fixed,'fixed_source_full_target_calibration':fixed,'target_refit':adapted,
              'frozen_source_matched':matched,'fixed_source_matched_target_calibration':matched}
        denominator=np.square(tx[test].astype(float)-tx[tt].mean(0)).sum(1)
        for r in s['regimes']:
            regime=r['regime'];model=maps[regime];codes=nearest(tx,model['centers']);k=len(model['centers'])
            nr=load(dest/(regime+'_readout.npz'))
            if 'target_calibration' in regime or regime=='target_refit':expected_nb=nb_fit(codes[tt],ty[tt],k)
            elif regime=='frozen_source_full':expected_nb={key:readout[key] for key in ['nb_likelihood','nb_prior']}
            else:expected_nb=nb_fit(nearest(sx[sm],model['centers']),sy[sm],k)
            for key in expected_nb:check(nr[key],expected_nb[key])
            check(pred[regime],nb_predict(nr,codes));score=pred[regime].to_numpy()[test]
            assert r['confirmation_questions']==test.sum() and r['K']==k
            for key,value in [('auroc',roc_auc_score(ty[test],score)),('brier',brier_score_loss(ty[test],score)),('log_loss',log_loss(ty[test],score))]:check(r[key],value)
            check_ci(r['versus_entropy'],paired_auc_ci(ty[test],score,pred.entropy.to_numpy()[test]))
            error=np.square(tx[test].astype(float)-model['centers'][codes[test]]).sum(1)
            check_ci(r['centroid_NMSE_common_target_train_center'],paired_ratio_ci(error,denominator))
            calibration=tx[tv] if regime=='target_refit' else sx[sv]
            cal_codes=nearest(calibration,model['centers']);threshold=np.quantile(np.linalg.norm(calibration-model['centers'][cal_codes],axis=1),.95)
            cut=np.quantile(density(calibration,model),.05)
            check(r['distance_threshold'],threshold);check(r['density_threshold'],cut)
            check(r['distance_coverage'],(np.linalg.norm(tx[test]-model['centers'][codes[test]],axis=1)<=threshold).mean())
            check(r['density_coverage'],(density(tx[test],model)>=cut).mean())
            flat.append({'view':name,'regime':regime,'K':k,'n':int(test.sum()),'failures':int(ty[test].sum()),
                'AUROC':r['auroc'],'Brier':r['brier'],'log_loss':r['log_loss'],
                'NMSE':r['centroid_NMSE_common_target_train_center']['estimate'],
                'distance_coverage':r['distance_coverage'],'density_coverage':r['density_coverage']})
        for row in s['paired_comparisons']:check_ci(row,paired_auc_ci(ty[test],pred[row['regime']].to_numpy()[test],pred[row['baseline']].to_numpy()[test]))
        check(s['continuous_baseline']['source_linear_auroc'],roc_auc_score(ty[test],pred.frozen_source_linear.to_numpy()[test]))
        check(s['continuous_baseline']['entropy_auroc'],roc_auc_score(ty[test],pred.entropy.to_numpy()[test]))
        rate=(ty[tt].sum()+1)/(tt.sum()+2);score=np.full(test.sum(),rate)
        priors.append({'view':name,'adaptation_failure_prior':float(rate),'confirmation_failure_rate':float(ty[test].mean()),
            'AUROC':.5,'log_loss':float(log_loss(ty[test],score)),'Brier':float(brier_score_loss(ty[test],score)),
            'scope':'post-hoc descriptive prior-only calibration control; no target-confirmation fitting'})
        for file in ['summary.json','predictions.parquet','fit_question_ids.json']:hashes[str(dest/file)]=sha(dest/file)
        print(json.dumps({'audited_transfer_view':name}),flush=True)
    primary=json.loads((ROOT/'primary_comparison.json').read_text())
    a=pd.read_parquet(ROOT/'transfer/p16_l28_mean16/predictions.parquet').query("split == 'test'").set_index('sample_id').sort_index()
    b=pd.read_parquet(ROOT/'transfer/p16_l28_last/predictions.parquet').query("split == 'test'").set_index('sample_id').sort_index()
    assert a.index.equals(b.index);np.testing.assert_array_equal(a.failure,b.failure)
    for key,column in [('frozen_state_NB','frozen_source_full'),('secondary_frozen_linear','frozen_source_linear'),('secondary_recalibrated_state_NB','fixed_source_full_target_calibration')]:
        check_ci(primary[key],paired_auc_ci(a.failure,a[column],b[column]))
    joint=json.loads((ROOT/'joint_summary.json').read_text());assert len(joint)==45
    for row in joint:
        prefix,view,regime=row['prefix'],row['view'],row['regime'];frames=[];prior=[]
        for layer in cfg['layers']:
            path=ROOT/'transfer'/f'p{prefix}_l{layer}_{view}'
            frames.append(pd.read_parquet(path/'predictions.parquet').set_index('sample_id').sort_index())
            prior.append(load(path/(regime+'_readout.npz'))['nb_prior'])
        for f in frames[1:]:assert f.index.equals(frames[0].index);np.testing.assert_array_equal(f.failure,frames[0].failure)
        for pr in prior[1:]:check(pr,prior[0])
        scores=expit(sum(logit(f[regime].clip(1e-12,1-1e-12).to_numpy()) for f in frames)-3*np.log(prior[0][1]/prior[0][0]))
        data=pd.read_parquet(ROOT/'joint'/f'p{prefix}_{view}.parquet').set_index('sample_id').loc[frames[0].index]
        check(data[regime],scores);test=data.split.eq('test').to_numpy();y=data.failure.to_numpy()[test]
        for key,value in [('auroc',roc_auc_score(y,scores[test])),('brier',brier_score_loss(y,scores[test])),('log_loss',log_loss(y,scores[test]))]:check(row[key],value)
    dest=ROOT/'report';dest.mkdir(exist_ok=True)
    table=pd.DataFrame(flat);table.to_parquet(dest/'all_regimes.parquet',index=False)
    write_json(dest/'prior_only_controls.json',priors);write_json(dest/'input_hashes.json',hashes)
    result={'views':len(expected),'regime_rows':len(flat),'joint_rows':len(joint),'candidates':counts[0],
        'converged_candidates':counts[1],'selection_training_ids_predictions_metrics_CIs_geometry_coverage_and_joint_prior_verified':True,
        'primary':primary,'transfer_plan_sha256':sha(ROOT/'transfer_plan.json'),'code_sha256':sha(Path(__file__)),
        'input_hashes_sha256':sha(dest/'input_hashes.json')}
    write_json(ROOT/'audit.json',result)
    (dest/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>GSM8K frozen transfer</title>'
        '<style>body{font:16px system-ui;margin:32px;line-height:1.6}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}.scroll{overflow:auto}</style>'
        '<h1>MATH → GSM8K：冻结状态迁移</h1><p>36视图、五设置；固定541适配/242验证/536确认。无确认集选层、选K或选表示。</p>'
        '<h2>预定主比较</h2><pre>'+html.escape(json.dumps(primary,indent=2))+'</pre>'
        '<p>主检验是block28/prefix16冻结源NB：mean16对last。其它均为次要；所有区间pointwise，不挑最佳视图代替主终点。</p>'
        '<p>目标校准使用适配集标签；几何覆盖是相对指定域验证阈值，不证明同一个功能状态。常数先验基线用于区分排序与校准。</p>'
        '<h2>全部设置</h2><div class="scroll">'+table.to_html(index=False,float_format=lambda v:f'{v:.5f}')+'</div>'
        '<details><summary>仅适配集先验的校准对照</summary>'+pd.DataFrame(priors).to_html(index=False)+'</details>'
        '<p><a href="../audit.json">完整审计</a></p></html>')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
