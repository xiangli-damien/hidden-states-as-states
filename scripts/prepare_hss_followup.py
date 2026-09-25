"""Freeze Step 1/2 inputs after the predecessor has completed. CPU only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from hss.align import align_layers
from hss.types import AlignSpec
from revision_common import sha, write_json, write_npz, provenance, freeze
from sink_direction_common import boxed_answer
from hss_followup_tools import (PARAMS, stratified_half_split, sink_direction,
    type_matched_targets, iso_controls, manifold_controls, chart_vectors)


def run(root, previous):
    root, previous = Path(root), Path(previous)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        raise RuntimeError('Plan already frozen')
    complete = json.loads((previous/'COMPLETE.json').read_text())
    assert complete['summary_sha256'] == sha(previous/'summary.json')
    old = json.loads((previous/'plan.json').read_text()); cfg = old['config'].copy()
    for path, digest in old['files'].items():
        assert sha(path) == digest, path
    files = [previous/n for n in ['plan.json','COMPLETE.json','summary.json','directions.npz','cases.json']]
    cache = Path(cfg['mean_cache'])
    meta = pd.read_parquet(cache/'rows.parquet')
    foundation = Path(cfg['foundation'])/'prefixes'
    frames, tokens = [], {}
    for mark in sorted(foundation.glob('shard_*/_SUCCESS.json')):
        rec = json.loads(mark.read_text())
        files.append(mark)
        for name in ['rows.parquet','tokens.json']:
            path = mark.parent/name
            assert sha(path) == rec['sha256'][name]
            files.append(path)
        frames.append(pd.read_parquet(mark.parent/'rows.parquet'))
        tokens.update({r['sample_id']: r for r in json.loads((mark.parent/'tokens.json').read_text())})
    frame = pd.concat(frames).set_index('sample_id').loc[meta.sample_id].reset_index()
    assert len(frame) == 5000 and not frame.sample_id.duplicated().any()
    np.testing.assert_array_equal(frame.label, meta.label)
    selection_path = Path('/lambda/nfs/dami/hss/qwen-math-ablation-20260918/selection_ablation.csv')
    selection = pd.read_csv(selection_path)
    selected = selection[(selection['view']=='post') & (selection.method=='gmm') &
        (selection.criterion=='icl') & np.isclose(selection.tolerance, .02)].sort_values('layer')
    assert selected.layer.tolist() == list(range(29))
    files.append(selection_path)
    centers, maps, labels = [], {}, {}
    for row in selected.itertuples():
        fit = Path(row.fit_path)
        for name in ['model.npz','assignments.npz','fit.json']:
            files.append(fit/name)
        model = dict(np.load(fit/'model.npz'))
        centers.append(model['means'])
        labels[row.layer] = np.load(fit/'assignments.npz')['posterior']
        for key in ['means','covariances','weights']:
            maps[f'l{row.layer}_{key}'] = model[key]
    alignment = align_layers(centers, layers=list(range(29)),
        spec=AlignSpec(similarity='cosine', method='hungarian', threshold=.6))
    layer = PARAMS['hook_layer']; sink = old['audit']['candidate_cluster']
    gid = int(alignment.local_to_global[layer][sink])
    late = max(i for i,g in enumerate(alignment.local_to_global) if gid in g)
    train = frame.split.eq('train').to_numpy()
    frame['cluster'] = labels[layer]; frame['is_sink'] = frame.cluster.eq(sink)
    assert train.sum()==3011 and (train & frame.is_sink).sum()==197
    frame['type'] = frame.category.astype(str); frame['correct'] = frame.label.astype(bool)
    frame['boxed'] = [boxed_answer(str(t)) is not None for t in frame.response_text]
    S = frame[train & frame.is_sink].sort_values('sample_id')
    N = frame[train & ~frame.is_sink].sort_values('sample_id')
    SA, SB = stratified_half_split(S.sample_id, S.type)
    NA, NB = stratified_half_split(N.sample_id, list(zip(N.type, N.correct)))
    normal_pool = sorted(set(frame.loc[frame.correct & frame.boxed,'sample_id']) & NB)
    MB = np.random.default_rng(PARAMS['seed']+3).choice(normal_pool,
        min(PARAMS['normal_screen_questions'],len(normal_pool)),replace=False).tolist()
    tau_ids = np.random.default_rng(PARAMS['seed']+4).choice(sorted(NA),
        min(PARAMS['tau_sample_responses'],len(NA)),replace=False).tolist()
    X = np.load(cache/f'layer_{layer}.npy'); files.append(cache/f'layer_{layer}.npy')
    idindex = {str(s): i for i,s in enumerate(frame.sample_id)}
    def take(ids, x=X): return x[[idindex[s] for s in ids]].astype(np.float64)
    d, s_hat = sink_direction(take(sorted(NA)),take(sorted(SA)))
    avec = sorted(SA | NA); ameta = frame.iloc[[idindex[s] for s in avec]]
    targets = type_matched_targets(ameta, take(avec), frame.type.unique(),np.linalg.norm(d))
    ctrl_iso = iso_controls(X.shape[1]); ctrl_man, pairs = manifold_controls(np.delete(centers[layer],sink,0))
    ctrl = np.concatenate([ctrl_iso,ctrl_man])
    decoder_path = Path('/lambda/nfs/dami/hss/revision-locality-20260923/decoders/p16_l14_tokens/decoder.npz')
    files.append(decoder_path); decoder = dict(np.load(decoder_path))
    bases = decoder['local_basis'][:,:PARAMS['chart_rank']].transpose(0,2,1)
    np.testing.assert_allclose(np.einsum('sdr,sdq->srq',bases,bases),
        np.broadcast_to(np.eye(PARAMS['chart_rank']),(len(bases),PARAMS['chart_rank'],PARAMS['chart_rank'])),atol=1e-5)
    chart, ratio = chart_vectors(bases,d)
    chart_controls = [chart_vectors(bases,v)[0] for v in ctrl]
    with np.load(previous/'directions.npz') as z: drun = z['hss'].copy()
    late_X = np.load(cache/f'layer_{late}.npy'); files.append(cache/f'layer_{late}.npy')
    late_direction = late_X[train & ~frame.is_sink].mean(0,dtype=np.float64)-late_X[train & frame.is_sink].mean(0,dtype=np.float64)
    for L,g in enumerate(alignment.local_to_global):
        maps[f'l{L}_global'] = g
        dominant = []
        for k in range(len(g)):
            values = frame.loc[train & (labels[L]==k),'type'].value_counts()
            dominant.append(str(values.index[0]) if len(values) else '')
        maps[f'l{L}_dominant_type'] = np.asarray(dominant)
    write_npz(root/'maps.npz', **maps)
    arrays = dict(d=d,s_hat=s_hat,controls=ctrl,chart=chart,chart_ratio=ratio,
        chart_controls=np.stack(chart_controls),token_centers=decoder['centers'],d_run=drun,d_late=late_direction,
        type_names=np.array(sorted(targets)),type_vectors=np.stack([targets[t]['v'] for t in sorted(targets)]))
    write_npz(root/'vectors_initial.npz', **arrays)
    splits = dict(S_A=sorted(SA),S_B=sorted(SB),N_A=sorted(NA),N_B=sorted(NB),M_B=MB,tau_ids=tau_ids)
    write_json(root/'split.json',splits)
    test = [r for r in json.loads((previous/'cases.json').read_text()) if r['split']!='dev']
    needed = set(SA) | set(SB) | set(tau_ids) | set(MB) | {r['sample_id'] for r in test}
    cohort = []
    for sid in sorted(needed):
        row = frame.iloc[idindex[sid]]
        cohort.append(dict(sample_id=sid, type=str(row.type), level=int(row.level),correct=bool(row.correct),
            ground_truth=str(row.ground_truth),response_text=str(row.response_text),
            prompt_text=str(row.prompt_text), **{k:v for k,v in tokens[sid].items() if k!='sample_id'}))
    write_json(root/'cohort.json',cohort)
    write_json(root/'test_cases.json',test)
    # Preserve predecessor bytes and receipts as immutable inputs to Step 1.
    for row in test:
        for cond in ['zero','hss','random']:
            p=previous/'outputs'/row['sample_id']/(cond+'.json')
            rr=json.loads(p.with_suffix('.receipt.json').read_text())
            assert sha(p)==rr['record_sha256']; files.append(p)
    target_metadata={k:{kk:vv for kk,vv in v.items() if kk!='v'} for k,v in targets.items()}
    write_json(root/'preparation.json',dict(sink_local=sink,sink_global=gid,L_late=late,
        k_by_layer=[len(c) for c in centers],n_split={k:len(v) for k,v in splits.items()},
        type_targets=target_metadata,control_centroid_pairs=pairs,
        control_centroid_local_ids=[i for i in range(len(centers[layer])) if i!=sink],
        d_norm=float(np.linalg.norm(d)),cos_d_drun=float(d@drun/np.linalg.norm(d)/np.linalg.norm(drun)),
        training_boxed_syntax=dict(complete_boxed=int(frame.loc[train,'boxed'].sum()),
            literal_backslash_boxed=sum('\\boxed{' in t for t in frame.loc[train,'response_text']),
            dollar_before_boxed=sum('$\\boxed{' in t for t in frame.loc[train,'response_text'])),
        force_string=PARAMS['force_string'],token_assignment='nearest_euclidean',response_assignment='diagonal_GMM_MAP'))
    scripts=Path(__file__).parent; repo=scripts.parent
    files += [scripts/n for n in ['hss_followup_tools.py','hss_followup_common.py','prepare_hss_followup.py',
        'run_hss_followup.py','report_hss_followup.py','revision_common.py','extract_revision_prefixes.py','sink_direction_common.py']]
    files += list((repo/'docs/sink-followup-20260925').glob('*'))
    files += [root/n for n in ['maps.npz','vectors_initial.npz','split.json','cohort.json','test_cases.json','preparation.json']]
    cfg['output']=str(root); cfg['previous']=str(previous)
    plan=provenance(cfg,files);plan.update(params=PARAMS,frozen_utc=datetime.now(timezone.utc).isoformat(),
        sink_global=gid,L_late=late,expected_step1=len(test)*4,
        expected_step2=(len(SB)+len(MB))*35, expected_step2_forwards=(2*len(SB)+len(MB))*35,
        step3B_status='blocked_pending_original_Qwen_prefix_map_NB_threshold',
        scope='Exploratory; designed after interim inspection; historical test and reused unsupervised maps')
    freeze(root/'plan.json',plan)
    write_json(root/'FREEZE.json',dict(plan_sha256=sha(root/'plan.json'),frozen_utc=plan['frozen_utc']))
    print(json.dumps({k:plan[k] for k in ['frozen_utc','sink_global','L_late','expected_step1','expected_step2','expected_step2_forwards']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--previous',required=True)
    a=p.parse_args();run(a.root,a.previous)
