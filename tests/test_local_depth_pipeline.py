"""Tiny end-to-end fixture tests isolation, terminal pre-norm and frozen scoring."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_local_depth_changes import prepare, fit, assign
from evaluate_local_depth_changes import evaluate


def test_complete_local_depth_pipeline(tmp_path):
    n,t,d=48,12,6; rng=np.random.default_rng(145)
    root=tmp_path/'result';cache=tmp_path/'cache';source=tmp_path/'source';shard=source/'shard_test'
    summary=tmp_path/'summary';reference=tmp_path/'reference';mean=tmp_path/'mean'
    for p in [root,cache,shard,summary,reference,mean]:p.mkdir(parents=True)
    y=np.arange(n)%2
    # Generate exactly bf16-representable source floats, then preserve them losslessly.
    raw=(rng.normal(size=(n,t,3,d))+.5*y[:,None,None,None]).astype(np.float32)
    h=(raw.view(np.uint32)&np.uint32(0xffff0000)).view(np.float32)
    text='abc. def. xy';offsets=np.tile(np.column_stack([np.arange(t),np.arange(1,t+1)]),(n,1))
    rows=pd.DataFrame(dict(sample_id=[f'q{i}' for i in range(n)],question_group=[f'g{i}' for i in range(n)],
        n_tokens=np.full(n,t),n_response_tokens=np.full(n,t),status=np.ones(n),label=1-y,
        prompt_text=['Question']*n,response_text=[text]*n,ground_truth=['answer']*n,
        entropy=rng.uniform(1,3,n),source_run=[str(shard)]*n))
    rows.to_parquet(reference/'rows.parquet',index=False);rows.to_parquet(shard/'data.parquet',index=False)
    pd.DataFrame(dict(sample_id=rows.sample_id,split=['train']*24+['validation']*12+['test']*12)).to_parquet(reference/'splits.parquet',index=False)
    (summary/'_SUCCESS.json').write_text(json.dumps(dict(shards=['shard_test'])))
    (shard/'manifest.json').write_text(json.dumps(dict(model=dict(identifier='tiny'),custom=dict(special_token_ids=[]))))
    (shard/'_SUCCESS').write_text('ok');(shard/'_COPY_VERIFIED.json').write_text('{}')
    z=zarr.open_group(str(shard/'tensors.zarr'),mode='w');stored=h.copy();stored[:,:,-1]*=16
    z.create_dataset('hidden_states/per_token',data=stored.reshape(n*t,3,d),chunks=(24,2,d))
    z.create_dataset('final_norm/pre/per_token',data=h[:,:,-1].reshape(n*t,d))
    z.create_dataset('tokens/sample_ptr',data=np.arange(n+1)*t)
    z.create_dataset('tokens/ids',data=np.tile(np.arange(t),n));z.create_dataset('tokens/offsets',data=offsets)
    zarr.consolidate_metadata(str(shard/'tensors.zarr'))
    for l in [1,2]:
        np.save(mean/f'mean_state_L{l:02d}.npy',h[:,:,l].mean(1))
        np.save(mean/f'mean_delta_L{l:02d}.npy',h[:,:,l].mean(1)-h[:,:,l-1].mean(1))
    cfg=dict(output=str(root),cache=str(cache),source=str(source),summary_cache=str(summary),reference=str(reference),mean_cache=str(mean),
        model='tiny',expected_samples=n,hidden_dim=d,last_layer=2,layers=[1,2],fit_units_per_question=4,
        k_grid=[1,2],matched_k=2,seeds=[1],max_iter=100,retry_max_iter=200,tol=.001,reg_covar=.01,
        fit_workers=1,cpu_threads=1,read_workers=1,read_tokens=48,min_free_gib=0,seed=123,alpha=1.,crossfit_folds=3,
        bootstrap=20,far=.1,logistic_c=[.1])
    prepare(cfg);fit(cfg);assign(cfg);evaluate(cfg)
    assert (root/'evaluation'/'_SUCCESS.json').exists()
    report=json.loads((root/'evaluation'/'summary.json').read_text())
    assert report['n_tokens']==n*t and report['n_questions']==n and report['n_test']==12
    metrics=pd.read_csv(root/'evaluation'/'metrics.csv')
    assert np.isfinite(metrics.auroc).all() and (metrics.validation_far<=.1).all()
    for p in (root/'fits').glob('*/selection.json'):
        s=json.loads(p.read_text());assert s['training_questions']==24
        assert s['training_vectors']==(24 if s['name'].startswith('mean_') else 96)
    a=np.load(root/'assignments'/'shard_test.npz')
    assert a['selected_token_delta'].shape==(n*t,2)
    assert len(list((root/'evaluation'/'report'/'cases').glob('*.json')))==12
