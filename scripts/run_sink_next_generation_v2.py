"""Experiment4: frozen online clamp and current-activation norm-matched control."""
import argparse
import copy
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import shutil
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from run_sink_next import (BASE,SOURCE,V3,REPO,SEED,inputs,forward,now,state)
from sink_online_rounding_v2 import MatchedTransform,selftest
from run_hss_followup import generate
from extract_revision_prefixes import load_model
from hss_followup_common import receipt,verify_record,gmm_assign
from sink_direction_common import boxed_answer
from revision_common import sha,write_json,write_npz,freeze
from openact_eval.evaluators.registry import auto_select_evaluator


def prepare():
    root=BASE/'exp4-v2';root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        plan=json.loads((root/'plan.json').read_text())
        for p,h in plan['files'].items():assert sha(p)==h,p
        return root,plan
    old,cfg,split,prep,frame,cohort,files=inputs()
    previous=Path(old['config']['previous']);test=json.loads((SOURCE/'test_cases.json').read_text())
    sinks=sorted(r['sample_id'] for r in test if r['split']=='sink_test')
    normal_pool=sorted(r['sample_id'] for r in test if r['split']=='normal_test')
    assert len(sinks)==55 and len(normal_pool)==100,(len(sinks),len(normal_pool))
    normals=sorted(np.random.default_rng(SEED).choice(normal_pool,50,replace=False).tolist())
    all_sink=sinks+sorted(split['S_B']);assert len(all_sink)==len(set(all_sink))==152
    assert not (set(all_sink)&set(normals))
    cases=[dict(question_id=q,set='sink',origin='historical_test' if q in sinks else 'S_B') for q in all_sink]
    cases += [dict(question_id=q,set='normal',origin='historical_test') for q in normals]
    # Fixed pseudo-random question order, all three arms per question.
    order=np.random.default_rng(SEED).permutation(len(cases));cases=[cases[i] for i in order]
    reuse=sinks+normals
    checks=sorted(np.random.default_rng(SEED).choice(sorted(reuse),10,replace=False).tolist())
    for q in reuse:
        p=previous/'outputs'/q/'zero.json'
        assert sha(p)==json.loads(p.with_suffix('.receipt.json').read_text())['record_sha256']
        files.append(p)
    files += [previous/'generation_config.json']
    write_json(root/'cases.json',cases);write_json(root/'cohort.json',[cohort[c['question_id']] for c in cases])
    write_json(root/'replay_ids.json',checks)
    code=['run_sink_next_generation_v2.py','sink_online_rounding_v2.py','run_sink_next_generation.py','report_sink_next_generation.py','run_sink_next.py','run_sink_energy_matched_v3.py',
          'run_hss_followup.py','hss_followup_common.py','hss_followup_tools.py','extract_revision_prefixes.py','revision_common.py','sink_direction_common.py','report_sink_energy_matched.py']
    files += [REPO/'scripts'/n for n in code]+[REPO/'docs/sink-next-protocol-20260925.zh-CN.md',REPO/'docs/sink-next-generation-numerical-amendment-v2-20260925.md']
    original=BASE/'exp4'
    imported=list(sorted((original/'records').glob('*/*.json')))+list(sorted((original/'replay').glob('*.json')))
    imported=[p for p in imported if not p.name.endswith('.receipt.json')]
    for p in imported:
        verify_record(p,original)
        files.extend([p,p.with_suffix('.receipt.json')])
        if p.with_suffix('.npz').exists():
            assert json.loads(p.read_text())['arrays_sha256']==sha(p.with_suffix('.npz'))
            files.append(p.with_suffix('.npz'))
    files += [original/'plan.json',original/'failure.json',original/'rounding_diagnostic/summary.json',original/'rounding_diagnostic/failed_activation.npz']
    files += [root/n for n in ['cases.json','cohort.json','replay_ids.json']]
    plan=dict(config=cfg,files={str(p):sha(p) for p in files},frozen_utc=now(),git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        numerical_amendment='v2: retry failed relative-energy match without internal absolute stopping floor; outer gates unchanged',imported_records=[str(p.relative_to(original)) for p in imported],previous=str(previous),sink_local=prep['sink_local'],expected_records=606,seed=SEED,bootstrap_replicates=100000,
        max_new_tokens=2048,conditions=['zero','C1','ORTH_MAN_1'],cutoff_utc='2026-09-26T01:59:00+00:00',
        primary='complete nonempty boxed and n_tokens<2048; sink152; two-sided exact paired McNemar vs zero and control; Holm2',
        scope='Exploratory reused questions; per-current-activation rule matched, not equal realized energy across diverged generations; unhooked TF re-encoding.')
    freeze(root/'plan.json',plan);write_json(root/'FREEZE.json',dict(plan_sha256=sha(root/'plan.json'),frozen_utc=plan['frozen_utc']))
    for p in imported:
        dest=root/p.relative_to(original);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
        extra=dict(imported_from=str(p),original_plan_sha256=sha(original/'plan.json'),original_record_sha256=sha(p))
        if p.with_suffix('.npz').exists():
            shutil.copy2(p.with_suffix('.npz'),dest.with_suffix('.npz'));extra['arrays_sha256']=sha(dest.with_suffix('.npz'))
        receipt(dest,root,extra)
    return root,plan


class OnlineTransform:
    def __init__(self,axis,tau,direction,cfg,device):
        self.args=(axis,tau,direction,cfg,device);self.steps=[];self.norms=[]
    def __call__(self,h):
        op=MatchedTransform(*self.args);after=op(h)
        self.steps.append(op.metrics)
        self.norms.append(np.stack([op.arrays[k] for k in ['target_norm','actual_norm','actual_cosine_sink','actual_cosine_intended']],axis=1))
        return after


def run():
    root,plan=prepare();cfg=plan['config']
    lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU occupied'
    torch.set_num_threads(4);start=time.time();state(root,phase='loading',expected=606)
    model,tok=load_model(cfg);selftest(cfg,model.device)
    evaluator=auto_select_evaluator('math')
    generation=copy.deepcopy(model.generation_config);generation.do_sample=False;generation.num_beams=1;generation.repetition_penalty=1.
    if generation.pad_token_id is None:generation.pad_token_id=tok.pad_token_id or tok.eos_token_id
    old_generation=json.loads((Path(plan['previous'])/'generation_config.json').read_text())
    assert generation.to_dict()==old_generation,'Generation config differs from pilot'
    freeze(root/'generation_config.json',generation.to_dict())
    cohort={r['sample_id']:r for r in json.loads((root/'cohort.json').read_text())}
    vectors=dict(np.load(V3/'directions.npz'));axis=vectors['axis'];tau=float(vectors['tau']);direction=vectors['orthogonal_controls'][5]
    maps=dict(np.load(SOURCE/'maps.npz'));cases=json.loads((root/'cases.json').read_text())
    checks=json.loads((root/'replay_ids.json').read_text());replays=[]
    for i,q in enumerate(checks):
        p=root/'replay'/(q+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():verify_record(p,root);r=json.loads(p.read_text())
        else:
            tick=time.time();ids,geometry=generate(model,tok,cohort[q]['prompt_ids'],[],generation,2048)
            old=json.loads((Path(plan['previous'])/'outputs'/q/'zero.json').read_text())
            r=dict(question_id=q,ids=ids,exact=ids==old['generated_ids'],seconds=time.time()-tick)
            write_json(p,r);receipt(p,root)
        replays.append(r);state(root,phase='zero_replay',completed=i+1,expected=10)
    replay_ok=all(r['exact'] for r in replays)
    write_json(root/'replay_audit.json',dict(all10_exact=replay_ok,checks=[dict(question_id=r['question_id'],exact=r['exact'],seconds=r['seconds']) for r in replays],policy='reuse105' if replay_ok else 'regenerate_all202'))
    records=[]
    for case in cases:
        sid=case['question_id'];row=cohort[sid]
        for cond in plan['conditions']:
            p=root/'records'/sid/(cond+'.json');p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                verify_record(p,root);r=json.loads(p.read_text());assert sha(p.with_suffix('.npz'))==r['arrays_sha256']
            else:
                tick=time.time();op=None;reused=False
                if cond=='zero' and case['origin']=='historical_test' and replay_ok:
                    old=json.loads((Path(plan['previous'])/'outputs'/sid/'zero.json').read_text())
                    ids=old['generated_ids'];geometry=None;reused=True
                elif cond=='zero' and sid in checks:
                    ids=next(r['ids'] for r in replays if r['question_id']==sid);geometry=None
                else:
                    if cond!='zero':op=OnlineTransform(axis,tau,None if cond=='C1' else direction,cfg,model.device)
                    ids,geometry=generate(model,tok,row['prompt_ids'],[],generation,2048,op)
                assert 0<len(ids)<=2048
                text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                answer=boxed_answer(text)
                scored=evaluator.evaluate_sample(SimpleNamespace(sample_idx=int(sid.split('_')[-1]),response_text=text,
                    ground_truth=row['ground_truth'],meta={'sample_id':sid}))
                assert scored.is_correct is not None and not scored.error
                # Always use the original, unhooked model to map the new text.
                x=forward(model,dict(row,response_ids=ids),14,capture=True)
                mean=x.mean(0)
                assigned=int(gmm_assign(mean,maps['l14_means'],maps['l14_covariances'],maps['l14_weights'])[0])
                normrows=np.concatenate(op.norms) if op and op.norms else np.empty((0,4))
                if op:assert len(normrows)==len(ids)-1
                write_npz(p.with_suffix('.npz'),response_mean=mean,update_norms_and_cosines=normrows)
                r=dict(**case,condition=cond,generated_ids=ids,response_text=text,n_tokens=len(ids),boxed=answer is not None,
                    primary_success=bool(answer is not None and len(ids)<2048),correct=bool(scored.is_correct),
                    sink=assigned==plan['sink_local'],cluster=assigned,geometry=geometry,reused_zero=reused,
                    online_audit=dict(steps=len(normrows),all_passed=True,
                        max_tolerance_fraction=max((s['maximum_tolerance_fraction'] for s in op.steps),default=0.) if op else 0.,
                        max_step_energy_error=max((s['relative_total_energy_error'] for s in op.steps),default=0.) if op else 0.,
                        total_energy=float((normrows[:,1]**2).sum()) if len(normrows) else 0.,mean_norm=float(normrows[:,1].mean()) if len(normrows) else 0.),
                    arrays_sha256=sha(p.with_suffix('.npz')),seconds=time.time()-tick)
                write_json(p,r);receipt(p,root,dict(arrays_sha256=r['arrays_sha256']))
            records.append(r)
            state(root,phase='generation',completed=len(records),expected=606,last=sid+'/'+cond,elapsed_seconds=time.time()-start,
                recent_seconds=r['seconds'])
    flat=[{k:v for k,v in r.items() if k not in ['generated_ids','response_text','geometry','online_audit']} for r in records]
    pd.DataFrame(flat).to_parquet(root/'scores.parquet',index=False)
    write_json(root/'GPU_COMPLETE.json',dict(utc=now(),seconds=time.time()-start,records=len(records),plan_sha256=sha(root/'plan.json'),table_sha256=sha(root/'scores.parquet')))
    state(root,phase='gpu_complete',completed=606,expected=606)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    try:
        if a.prepare_only:
            root,plan=prepare();print(json.dumps(dict(root=str(root),records=plan['expected_records'],frozen=plan['frozen_utc'])))
        else:run()
    except BaseException:
        root=BASE/'exp4-v2';root.mkdir(parents=True,exist_ok=True)
        write_json(root/'failure.json',dict(utc=now(),traceback=traceback.format_exc()));raise
