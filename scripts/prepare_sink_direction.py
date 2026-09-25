"""Audit the frozen mean map and freeze a finite additive-direction pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_common import sha, freeze, write_json, write_npz, provenance
from sink_direction_common import composition, ordered


def run(cfg):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        raise RuntimeError('Already frozen; do not regenerate a plan')
    cache = Path(cfg['mean_cache']); fit = Path(cfg['fit'])
    X = np.load(cache/f'layer_{cfg["layer"]}.npy')
    meta = pd.read_parquet(cache/'rows.parquet')
    assert X.shape == (5000,3584) and np.isfinite(X).all()
    assert (np.linalg.norm(X,axis=1)>0).all() and not meta.sample_id.duplicated().any()
    with np.load(fit/'assignments.npz') as z: assigned = z['posterior'].copy()
    with np.load(fit/'model.npz') as z: centers = z['means'].copy()
    assert len(centers) == cfg['expected_k'] and len(assigned) == len(meta)
    fit_info = json.loads((fit/'fit.json').read_text())
    assert fit_info['context']['layer'] == cfg['layer'] and fit_info['model']['converged']
    assert fit_info['context']['snapshot'] == json.loads((cache/'_SUCCESS.json').read_text())['key']
    foundation = Path(cfg['foundation'])/'prefixes'
    frames, tokens = [], {}
    files = [cache/'rows.parquet', cache/f'layer_{cfg["layer"]}.npy', cache/'_SUCCESS.json',
             fit/'fit.json', fit/'model.npz', fit/'assignments.npz']
    source_cfg = json.loads((foundation/'plan.json').read_text())['config']
    assert source_cfg['model'] == cfg['model'] and source_cfg['revision'] == cfg['revision']
    files.append(foundation/'plan.json')
    for marker in sorted(foundation.glob('shard_*/_SUCCESS.json')):
        receipt = json.loads(marker.read_text())
        for name in ['rows.parquet','tokens.json']:
            assert sha(marker.parent/name) == receipt['sha256'][name]
            files.append(marker.parent/name)
        files.append(marker)
        frames.append(pd.read_parquet(marker.parent/'rows.parquet'))
        for row in json.loads((marker.parent/'tokens.json').read_text()): tokens[row['sample_id']] = row
    frame = pd.concat(frames).set_index('sample_id').loc[meta.sample_id].reset_index()
    assert len(frame)==5000 and not frame.question_group.duplicated().any()
    np.testing.assert_array_equal(meta.label, frame.label)
    frame['cluster'] = assigned
    flags = pd.DataFrame([composition(str(r.response_text), str(r.finish_reason))
                          for r in frame.itertuples()])
    frame = pd.concat([frame,flags], axis=1)
    train = frame.split.eq('train').to_numpy()
    assert train.sum()==3011
    train_table = frame.loc[train].groupby('cluster').agg(
        n=('label','size'), boxed_rate=('complete_boxed','mean'), empty_rate=('empty','mean'))
    eligible = train_table[(train_table.n >= cfg['minimum_train_cluster']) &
                           (train_table.boxed_rate <= cfg['maximum_train_boxed_rate'])]
    table = frame.groupby(['split','cluster']).agg(n=('label','size'),
        original_accuracy=('label','mean'), boxed_rate=('complete_boxed','mean'),
        empty_rate=('empty','mean'), truncated_rate=('truncated','mean'),
        median_tokens=('n_tokens','median'), repeated_12gram_mean=('repeated_12gram_fraction','mean'),
        deferral_phrase_rate=('deferral_phrase','mean'))
    table.to_csv(root/'composition.csv')
    frame[['sample_id','split','cluster','label','n_tokens','finish_reason',
           *flags.columns,'prompt_text','response_text','ground_truth']].to_parquet(root/'composition_rows.parquet',index=False)
    if eligible.empty:
        write_json(root/'STOP.json',{'reason':'No sufficiently populated train cluster with <=30% boxed compliance'})
        return
    sink = int(eligible.sort_values(['boxed_rate','n'], ascending=[True,False]).index[0])
    if eligible.loc[sink,'empty_rate'] >= .5:
        write_json(root/'STOP.json',{'reason':'Candidate mainly empty outputs; no steering launched','cluster':sink})
        return
    mask = assigned == sink
    sink_mean = X[train & mask].astype(float).mean(0)
    nonsink_mean = X[train & ~mask].astype(float).mean(0)
    direction = nonsink_mean-sink_mean
    rng = np.random.default_rng(cfg['seed'])
    random = rng.normal(size=len(direction)); random *= np.linalg.norm(direction)/np.linalg.norm(random)
    assert np.linalg.norm(direction)>0
    write_npz(root/'directions.npz', hss=direction.astype('float32'), random=random.astype('float32'),
              sink_train_center=sink_mean, nonsink_train_center=nonsink_mean,
              frozen_gmm_center=centers[sink])
    cases = []
    by_id = frame.set_index('sample_id')
    pools = [('dev', 'validation', mask, cfg['dev_questions']),
             ('sink_test', 'test', mask, cfg['test_sink_cap']),
             ('normal_test', 'test', ~mask & frame.complete_boxed.to_numpy() & ~frame.truncated.to_numpy(),
              cfg['normal_questions'])]
    for split, source_split, selection, cap in pools:
        ids = frame.loc[frame.split.eq(source_split) & selection,'sample_id'].tolist()
        ids = ordered(ids,f'sink-direction-{cfg["seed"]}-{split}')[:cap]
        if split!='sink_test': assert len(ids)==cap
        else: assert len(ids)>=32
        for sid in ids:
            r=by_id.loc[sid]
            cases.append({'sample_id':sid,'split':split,'source_split':source_split,
                'question_group':str(r.question_group),'prompt_ids':tokens[sid]['prompt_ids'],
                'prompt_text':str(r.prompt_text),'ground_truth':str(r.ground_truth),
                'original_response':str(r.response_text),'original_label':int(r.label),
                'original_complete_boxed':bool(r.complete_boxed),'original_cluster':int(r.cluster)})
    assert len({r['question_group'] for r in cases})==len(cases)
    freeze(root/'cases.json', cases)
    notes = {
        'candidate_cluster':sink, 'paper_sink_id_verified':False,
        'map_fit_scope':'Frozen unsupervised map fit on all5000 responses (transductive); historical test is not fresh data',
        'selection':'Train-only: n>=50, complete-boxed rate<=.30; select minimum rate, then largest count',
        'direction':'Train nonsink empirical mean minus train candidate empirical mean; no correctness labels used',
        'gmm_center_not_used_for_direction':'GMM center used all5000; recompute anchor on train to exclude dev/test from direction',
        'train_sink_questions':int((train&mask).sum()), 'train_nonsink_questions':int((train&~mask).sum()),
        'direction_norm':float(np.linalg.norm(direction)),
        'random_cosine':float(np.dot(direction,random)/np.linalg.norm(direction)**2),
        'candidate_train_median_mean_norm':float(np.median(np.linalg.norm(X[train&mask],axis=1))),
        'zero_means':0,'empty_generations':int(frame['empty'].sum()),
        'candidate_composition_by_split':table.xs(sink,level='cluster').reset_index().to_dict('records'),
        'primary_endpoint':'Presence of a complete nonempty boxed answer, not all semantic final-answer parsability',
        'secondary_endpoints':['Frozen OpenAct automatic correctness','permissive parser success','length','truncation'],
        'success_rule':'Sink test HSS minus random boxed rate: paired95% CI lower>0 and exact one-sided McNemar p<.05; also HSS must improve over zero. All other outcomes descriptive.',
        'normal_test':'100 originally complete-boxed, nontruncated nonsink questions, independent of correctness; all3 arms',
        'operator':'At stored hidden_states index14 = model.model.layers[13] output, shift every forwarded generated token; skip prompt prefill. First generated token is unaffected.',
        'mean_translation_limit':'Fixed-vector algebra is exact before rounding. Autoregressive states, tokens and lengths change, so final response mean need not shift by exactly alpha*d.',
        'scope':'Exploratory current-map degeneration-region intervention; not a reproduction of unlocated original-paper sink, not proof of general controllable reasoning states',
        'test_questions':sum(r['split']=='sink_test' for r in cases)}
    write_json(root/'audit.json',notes)
    files.extend([root/'cases.json',root/'directions.npz',root/'audit.json',root/'composition.csv'])
    files.extend(Path(__file__).with_name(n) for n in ['prepare_sink_direction.py','sink_direction_common.py',
                                                      'run_sink_direction.py','revision_common.py','extract_revision_prefixes.py'])
    plan=provenance(cfg,files);plan['audit']=notes
    plan['expected_generations']=cfg['dev_questions']*(1+len(cfg['alphas']))+3*(notes['test_questions']+cfg['normal_questions'])
    freeze(root/'plan.json',plan)
    write_json(root/'prepare_SUCCESS.json',{'plan_sha256':sha(root/'plan.json'),'expected_generations':plan['expected_generations']})
    print(json.dumps(notes,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()))
