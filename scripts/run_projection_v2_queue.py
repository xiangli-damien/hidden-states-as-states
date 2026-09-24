"""Timing-only v2 follower. Protect the primary queue and keep one GPU process."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from revision_common import sha,freeze,write_json


def choose_tier(plan,seconds_per_condition,now):
    available=plan['no_new_batch_unix']-now
    valid=[c['name'] for c in plan['conditions'] if c['asset_available']]
    standard_n=256+32*len(valid)+32*sum(x in valid for x in ['D07','D08'])
    if available>=standard_n*seconds_per_condition:
        return dict(name='standard',transfer_n=64,transfer_conditions=['baseline','local8','shared8','wrong_local8'],
                    validation_ids=valid,expected_generations=standard_n)
    reduced=[x for x in ['D01','D02','D03','D05'] if x in valid]
    n=96+32*len(reduced)
    if available>=n*seconds_per_condition:
        return dict(name='reduced',transfer_n=32,transfer_conditions=['baseline','local8','shared8'],
                    validation_ids=reduced,expected_generations=n)
    return dict(name='no_new_batch_budget',transfer_n=0,transfer_conditions=[],validation_ids=[],expected_generations=0)


def run(root):
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'queue_status.json').exists():raise RuntimeError('Existing follower: do not start a duplicate')
    repo=Path(__file__).resolve().parents[1];plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    primary=Path(cfg['protected_primary']);state=dict(state='waiting_primary',pid=os.getpid(),stages={},started_unix=time.time())
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    cpu=str(repo/'.venv/bin/python');gpu='/lambda/nfs/dami/openact/.venv/bin/python'
    def save():state['updated_unix']=time.time();write_json(root/'queue_status.json',state)
    def step(name,args):
        for p,h in plan['files'].items():assert sha(p)==h,p
        with (root/(name+'.log')).open('x') as log:
            child=subprocess.Popen(args,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(state='running',phase=name);state['stages'][name]=dict(state='running',pid=child.pid,started_unix=time.time());save()
            rc=child.wait()
        state['stages'][name].update(state='complete' if rc==0 else 'failed',returncode=rc,ended_unix=time.time());save()
        if rc:raise RuntimeError(f'{name} failed; preserve artifacts, no automatic change of protocol')
    save()
    try:
        while True:
            upstream=json.loads((primary/'queue_status.json').read_text())
            if upstream['state']=='failed':raise RuntimeError('Protected primary failed; do not bypass its audit')
            if upstream['state']=='complete':break
            os.kill(upstream['pid'],0)
            if time.time()>plan['no_new_batch_unix']:raise TimeoutError('Primary still running at v2 no-new-batch deadline; primary untouched')
            state['upstream_phase']=upstream.get('phase');save();time.sleep(30)
        assert json.loads((primary/'steering_statistics_audit.json').read_text())['complete']
        assert json.loads((root/'preflight_SUCCESS.json').read_text())['complete']
        timing=json.loads((primary/'selection.json').read_text())
        cost=max(25.,timing['p90_smoke_condition_seconds'])
        tier=choose_tier(plan,cost,time.time());tier.update(frozen_unix=time.time(),criterion='Timing/assets only; no primary-test effect read',
             conservative_seconds_per_condition=cost,plan_sha256=sha(root/'plan.json'))
        freeze(root/'tier.json',tier);state['tier']=tier['name'];save()
        order=['transfer','D01','D02','D03','D05','D06','D04','D07','D08','D09','D10']
        for batch in order:
            admitted=(tier['transfer_n']>0 if batch=='transfer' else batch in tier['validation_ids'])
            if not admitted:
                state['stages'][batch]=dict(state='not_run',reason='budget_tier_or_missing_asset');save();continue
            n=tier['transfer_n']*len(tier['transfer_conditions']) if batch=='transfer' else (64 if batch in ['D07','D08'] else 32)
            if time.time()+n*cost>plan['no_new_batch_unix']:
                state['stages'][batch]=dict(state='not_run',reason='Timing-only full-batch admission reserve');save();continue
            step(batch,[gpu,'scripts/evaluate_projection_v2.py','--root',str(root),'--batch',batch])
            step(batch+'_audit',[gpu,'scripts/audit_projection_v2.py','--root',str(root),'--batch',batch])
        step('summarize',[cpu,'scripts/summarize_projection_v2.py','--root',str(root)])
        step('statistics_audit',[cpu,'scripts/audit_projection_v2_statistics.py','--root',str(root)])
        step('report',[cpu,'scripts/report_projection_v2.py','--root',str(root)])
        state.update(state='complete',report_state='audited_visual_and_semantic_case_review_pending');save()
    except BaseException:
        state.update(state='failed',traceback=traceback.format_exc());save();raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);run(ap.parse_args().root)
