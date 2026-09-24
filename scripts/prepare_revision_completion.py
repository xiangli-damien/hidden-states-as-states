"""Pin cached inputs; select questions without reading their outcomes."""
import argparse
from datetime import datetime,timezone
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from revision_common import sha,write_json,freeze,provenance,nearest
from revision_completion_common import ordered,candidates


def run(cfg):
    root=Path(cfg['output']); root.mkdir(parents=True,exist_ok=True)
    if (root/'prepare_SUCCESS.json').exists(): raise RuntimeError('Already prepared; do not overwrite')
    foundation=Path(cfg['foundation']); locality=Path(cfg['locality']); decoder=Path(cfg['decoder'])
    receipt=json.loads((decoder.parent/'_SUCCESS.json').read_text())
    assert sha(decoder)==receipt['decoder_sha256']
    audit=json.loads((decoder.parent/'audit.json').read_text())
    assert sha(decoder.parent/'audit.json')==receipt['audit_sha256']
    train_ids=set(audit['training_question_ids'])
    source_cfg=json.loads((foundation/'prefixes/plan.json').read_text())['config']
    assert source_cfg['revision']==cfg['revision'] and source_cfg['model']==cfg['model']
    with np.load(decoder) as f: centers=f['centers'].astype(np.float64)
    rows=[];tokens={};files=[decoder,decoder.parent/'audit.json',decoder.parent/'_SUCCESS.json',
        foundation/'prefixes/plan.json',Path(__file__),Path(__file__).with_name('revision_completion_common.py'),
        Path(__file__).with_name('revision_locality_common.py'),Path(__file__).with_name('revision_common.py')]
    count=np.zeros(len(centers),int);correct=np.zeros(len(centers),int);n_train=0;sum_correct=0
    for marker in sorted((foundation/'prefixes').glob('shard_*/_SUCCESS.json')):
        rc=json.loads(marker.read_text());shard=marker.parent
        for name in ['rows.parquet','tokens.json','prefix_16.npz']:
            assert sha(shard/name)==rc['sha256'][name],str(shard/name)
        files.extend([marker,shard/'rows.parquet',shard/'tokens.json'])
        frame=pd.read_parquet(shard/'rows.parquet');rows.append(frame)
        for item in json.loads((shard/'tokens.json').read_text()):tokens[item['sample_id']]=item
        train=frame.sample_id.isin(train_ids).to_numpy()
        with np.load(shard/'prefix_16.npz') as f:
            valid=f['valid'].astype(bool); ix=train&valid
            x=f['window'][ix,source_cfg['layers'].index(cfg['layer'])]
        if len(x):
            code=nearest(x.reshape(-1,x.shape[-1]),centers).reshape(x.shape[:2])
            y=frame.loc[ix,'label'].to_numpy(int)
            for c,label in zip(code,y):
                unique=np.unique(c);count[unique]+=1;correct[unique]+=label
            n_train+=len(y);sum_correct+=int(y.sum())
    frame=pd.concat(rows,ignore_index=True)
    assert not frame.sample_id.duplicated().any() and set(frame.loc[frame.split.eq('train'),'sample_id'])==train_ids
    assert n_train==len(train_ids)
    p=sum_correct/n_train;estimate=(correct+cfg['support_kappa']*p)/(count+cfg['support_kappa'])
    support={'question_count':count.tolist(),'correct_question_count':correct.tolist(),
             'shrunk_correctness':estimate.tolist(),'training_prior':p,'train_questions':n_train,
             'training_ids':sorted(train_ids),'definition':'Unique training question per region over its16 actual tokens; no token label replication'}
    freeze(root/'support.json',support)
    old=set(json.loads((locality/'plan.json').read_text())['selected_sample_ids'])
    old.update(json.loads((foundation/'functional/plan.json').read_text())['sample_ids'])
    cases=[]
    for split,n in [('validation',cfg['validation_questions']),('test',cfg['test_questions'])]:
        pool=frame.loc[frame.split.eq(split)&~frame.sample_id.isin(old),'sample_id'].tolist()
        selected=ordered(pool,'completion-20260924-'+split)[:n]
        assert len(selected)==n and not set(selected)&train_ids
        for sid in selected:
            r=frame.set_index('sample_id').loc[sid];t=tokens[sid]
            cases.append({'sample_id':sid,'split':split,'question_group':str(r.question_group),
                          'prompt_ids':t['prompt_ids'],'prompt_text':str(r.prompt_text),
                          'ground_truth':str(r.ground_truth),'source_sample_idx':int(r.sample_idx)})
    assert len({r['question_group'] for r in cases})==len(cases)
    freeze(root/'inputs.json',cases)
    now=time.time()
    plan=provenance(cfg,files+[root/'support.json',root/'inputs.json'])
    plan.update(started_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),started_unix=now,
                deadline_unix=now+cfg['max_hours']*3600,validation_decision_unix=now+cfg['validation_decision_hours']*3600,
                candidates=candidates(cfg,bool(np.any(count>=cfg['support_min_questions']))),
                source_train_question_count=len(train_ids),excluded_prior_functional_count=len(old),
                raw_prefix_receipts_verified=50,scope='MATH historical questions held out from current policy selection')
    freeze(root/'plan.json',plan)
    write_json(root/'prepare_SUCCESS.json',{'complete':True,'plan_sha256':sha(root/'plan.json'),
        'inputs_sha256':sha(root/'inputs.json'),'support_sha256':sha(root/'support.json'),
        'questions':len(cases),'candidates':len(plan['candidates']),
        'supported_regions':int((count>=cfg['support_min_questions']).sum())})
    print(json.dumps({'prepared':len(cases),'train_questions':n_train,'supported_regions':int((count>=20).sum()),
                      'candidates':len(plan['candidates']),'deadline_utc':datetime.fromtimestamp(plan['deadline_unix'],timezone.utc).isoformat()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    with threadpool_limits(limits=4):run(json.loads(Path(a.config).read_text()))
