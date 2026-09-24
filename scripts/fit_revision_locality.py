"""Train-only shared residual bases with immutable source centers and partitions."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from fit_revision_geometry import read_view,basis
from revision_common import config,freeze,provenance,sha,write_json,write_npz,nearest,status
from revision_locality_common import derangement,stable_seed


def prepare(cfg):
    root=Path(cfg['output']);foundation=Path(cfg['foundation']);prefix=foundation/'prefixes'
    old=json.loads((foundation/'functional/plan.json').read_text())['sample_ids']
    # Metadata and all original extraction receipts identify the fixed corpus.
    files=[Path(__file__),Path(__file__).with_name('fit_revision_geometry.py'),
        Path(__file__).with_name('revision_common.py'),Path(__file__).with_name('revision_locality_common.py'),
        foundation/'functional/plan.json',prefix/'plan.json',prefix/'_SUCCESS.json']
    import pandas as pd
    frames=[];verified={}
    for marker in sorted(prefix.glob('shard_*/_SUCCESS.json')):
        files.extend([marker,marker.parent/'rows.parquet',marker.parent/'tokens.json'])
        receipt=json.loads(marker.read_text())
        for name in ['rows.parquet','tokens.json']+[f'prefix_{p}.npz' for p in sorted({v[0] for v in cfg['views']})]:
            actual=sha(marker.parent/name)
            assert actual==receipt['sha256'][name],str(marker.parent/name)
            verified[str(marker.parent/name)]=actual
        frames.append(pd.read_parquet(marker.parent/'rows.parquet'))
    frame=pd.concat(frames,ignore_index=True)
    assert not frame.sample_id.duplicated().any()
    selected=[]
    for split in ['validation','test']:
        sub=frame.loc[frame.split.eq(split)&~frame.sample_id.isin(old)].copy()
        sub['order']=sub.sample_id.map(lambda sid:stable_seed('locality-new-pilot-v1',sid))
        selected+=sub.sort_values(['order','sample_id']).head(cfg['questions_per_split']).sample_id.tolist()
    assert len(selected)==2*cfg['questions_per_split'] and not set(old)&set(selected)
    identity=provenance(cfg,files);identity['selected_sample_ids']=selected
    identity['excluded_prior_intervention_ids']=old
    identity['verified_prefix_inputs']=verified
    freeze(root/'plan.json',identity)
    return selected


def fit_one(job):
    cfg,view,selected=job
    with threadpool_limits(limits=cfg['blas_threads']):
        return fit_limited(cfg,view,selected)


def fit_limited(cfg,view,selected):
    started=time.monotonic();prefix,layer,role=view;name=f'p{prefix}_l{layer}_{role}'
    root=Path(cfg['output'])/'decoders'/name;root.mkdir(parents=True,exist_ok=True)
    source=Path(cfg['geometry'])/name;old=json.loads((source/'_SUCCESS.json').read_text())
    assert sha(source/'decoder.npz')==old['decoder_sha256']
    assert sha(source/'summary.json')==old['summary_sha256']
    summary=json.loads((source/'summary.json').read_text());assert summary['tokens_per_question_fit']==16
    if (root/'_SUCCESS.json').exists():
        receipt=json.loads((root/'_SUCCESS.json').read_text())
        assert sha(root/'decoder.npz')==receipt['decoder_sha256']
        assert sha(root/'audit.json')==receipt['audit_sha256']
        return json.loads((root/'audit.json').read_text())
    rcfg={**cfg,'prefix_root':str(Path(cfg['foundation'])/'prefixes')}
    frame,x=read_view(rcfg,prefix,layer,role)
    train=frame.split.eq('train').to_numpy();xf=x[train].reshape(-1,x.shape[-1]).astype(np.float64)
    with np.load(source/'decoder.npz') as f:decoder={k:f[k].copy() for k in f.files}
    centers=decoder['centers'].astype(np.float64);assign=nearest(xf,centers)
    rank=max(cfg['ranks']);shared_rank=max(cfg['shared_ranks']);residual=xf-centers[assign]
    assert shared_rank<=min(xf.shape) and len(centers)==64
    decoder['shared_basis']=basis(residual,shared_rank,np.zeros(xf.shape[1]),42)
    empirical=decoder['local_empirical_centers'].astype(np.float64)
    decoder['shared_empirical_basis']=basis(xf-empirical[assign],rank,np.zeros(xf.shape[1]),42)
    counts=np.bincount(assign,minlength=len(centers))
    decoder['train_counts']=counts
    for seed in cfg['seeds']:decoder[f'permutation_{seed}']=derangement(len(centers),seed)
    checks={}
    for field in ['shared_basis','shared_empirical_basis','global_basis']:
        b=decoder[field].astype(np.float64)
        checks[field]=float(np.max(np.abs(b@b.T-np.eye(len(b)))))
        assert checks[field]<2e-5
    # Fixed fitted GMM mean and global PCA training mean MUST be restored.
    from revision_common import reconstruct
    np.testing.assert_allclose(reconstruct(decoder['train_mean'][None,:],decoder,'global_pca_8'),
        decoder['train_mean'][None,:],rtol=1e-6,atol=1e-5)
    audit={'view':name,'train_questions':int(train.sum()),'train_tokens':len(xf),'hidden_size':xf.shape[1],
        'k':len(centers),'ranks':cfg['ranks'],'shared_ranks':cfg['shared_ranks'],'normalization':'none','extra_residual_mean_added':False,
        'shared_fit':'uncentered residual second-moment SVD; same GMM anchors and nearest assignment as local residual SVD',
        'empirical_bridge':'separate shared basis around original empirical anchors; GMM assignment retained',
        'training_question_ids':frame.loc[train,'sample_id'].tolist(),'counts':counts.tolist(),
        'orthogonality_max_error':checks,'global_pca_mean_restore_check':True,
        'source_decoder_sha256':sha(source/'decoder.npz'),'source_summary_sha256':sha(source/'summary.json'),
        'source_global_basis_origin':'existing training-mean-anchored basis; no test recentering',
        'decoder_parameter_counts':{'encoder_centers':len(centers)*xf.shape[1],
            'shared_basis_per_rank':{str(r):r*xf.shape[1] for r in cfg['shared_ranks']},
            'local_bases_per_rank':{str(r):len(centers)*r*xf.shape[1] for r in cfg['ranks']},
            'extra_empirical_anchors':len(centers)*xf.shape[1]},
        'seconds':time.monotonic()-started}
    valid=frame.sample_id.isin(selected).to_numpy()
    write_npz(root/'pilot_activations.npz',sample_ids=frame.loc[valid,'sample_id'].to_numpy(str),x=x[valid])
    write_npz(root/'decoder.npz',**decoder);write_json(root/'audit.json',audit)
    write_json(root/'_SUCCESS.json',{'decoder_sha256':sha(root/'decoder.npz'),'audit_sha256':sha(root/'audit.json'),
        'pilot_activations_sha256':sha(root/'pilot_activations.npz')})
    return audit


def run(cfg):
    selected=prepare(cfg);root=Path(cfg['output']);results=[]
    status(root,'fit',state='running',completed=0,expected=len(cfg['views']))
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs=[pool.submit(fit_one,(cfg,v,selected)) for v in cfg['views']]
        for f in as_completed(jobs):
            r=f.result();results.append(r)
            status(root,'fit',state='running',completed=len(results),expected=len(jobs),last=r['view'])
    write_json(root/'fit_summary.json',results);write_json(root/'fit_SUCCESS.json',{'views':len(results)})
    status(root,'fit',state='complete',completed=len(results),expected=len(results))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);cfg=config(p.parse_args().config)
    try:run(cfg)
    except BaseException:
        status(cfg['output'],'fit',state='failed',traceback=traceback.format_exc());raise
