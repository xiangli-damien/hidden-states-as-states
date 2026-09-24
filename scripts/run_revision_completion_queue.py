"""Bounded real-task steering supervisor. Never rescues an unfavorable outcome."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from revision_common import write_json,sha


def run(root):
    repo=Path(__file__).resolve().parents[1]
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'queue_status.json').exists():raise RuntimeError('Existing queue state; inspect before any recovery')
    cpu=str(repo/'.venv/bin/python');gpu='/lambda/nfs/dami/openact/.venv/bin/python'
    names=['run_revision_completion_queue.py','evaluate_revision_completion.py','audit_revision_completion.py',
           'summarize_revision_completion.py','audit_revision_completion_statistics.py','revision_completion_common.py',
           'revision_common.py','revision_locality_common.py','check_revision_completion_hook.py','report_revision_completion.py']
    pinned={str(repo/'scripts'/n):sha(repo/'scripts'/n) for n in names}
    write_json(root/'queue_plan.json',{'files':pinned,'plan_sha256':sha(root/'plan.json')})
    state={'state':'running','pid':os.getpid(),'started_unix':time.time(),'stages':{}}
    env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
    def save():state['updated_unix']=time.time();write_json(root/'queue_status.json',state)
    def step(name,command):
        for path,digest in pinned.items():assert sha(path)==digest,path
        with (root/(name+'.log')).open('x') as log:
            p=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state['stages'][name]={'state':'running','pid':p.pid,'started_unix':time.time()};state['phase']=name;save()
            rc=p.wait()
        state['stages'][name].update(state='complete' if rc==0 else 'failed',returncode=rc,ended_unix=time.time());save()
        if rc:raise RuntimeError(f'{name} failed with {rc}; preserve artifacts and inspect')
    save()
    try:
        step('tiny_hook',[gpu,'scripts/check_revision_completion_hook.py'])
        for stage in ['smoke','validation']:
            step(stage,[gpu,'scripts/evaluate_revision_completion.py','--root',str(root),'--stage',stage])
            step(stage+'_audit',[gpu,'scripts/audit_revision_completion.py','--root',str(root),'--stage',stage])
        step('select',[cpu,'scripts/summarize_revision_completion.py','--root',str(root),'select'])
        selected=json.loads((root/'selection.json').read_text())
        if selected['run_test']:
            step('test',[gpu,'scripts/evaluate_revision_completion.py','--root',str(root),'--stage','test'])
            step('test_audit',[gpu,'scripts/audit_revision_completion.py','--root',str(root),'--stage','test'])
        step('summarize',[cpu,'scripts/summarize_revision_completion.py','--root',str(root),'summarize'])
        step('statistics_audit',[cpu,'scripts/audit_revision_completion_statistics.py','--root',str(root)])
        step('report',[cpu,'scripts/report_revision_completion.py','--root',str(root),'--mode','final'])
        state.update(state='complete',scientific_stop=None if selected['run_test'] else selected['reason'],
            report_state='statistics_audited_visual_review_pending')
        save()
    except BaseException:
        state.update(state='failed',traceback=traceback.format_exc());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
