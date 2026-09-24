"""Independent CPU companion; no generation, fitting or policy selection."""
import argparse,json,os,subprocess,time,traceback,fcntl
from pathlib import Path
from revision_common import write_json,sha


def run(root):
    repo=Path(__file__).resolve().parents[1];python=str(repo/'.venv/bin/python')
    lock=(root/'cached_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'cached_queue_status.json').exists():raise RuntimeError('Existing CPU queue; inspect before recovery')
    names=['analyze_revision_completion_cached.py','audit_revision_completion_cached.py','report_revision_completion.py']
    files={str(repo/'scripts'/n):sha(repo/'scripts'/n) for n in names}
    write_json(root/'cached_queue_plan.json',{'files':files,'source_plan_sha256':sha(root/'plan.json')})
    state={'state':'running','pid':os.getpid(),'started_unix':time.time(),'stages':{}}
    def save():state['updated_unix']=time.time();write_json(root/'cached_queue_status.json',state)
    save()
    try:
        for name,script,args in [('analyze',names[0],[]),('audit',names[1],[]),('report',names[2],['--mode','cached'])]:
            for p,h in files.items():assert sha(p)==h
            with (root/('cached_'+name+'.log')).open('x') as f:
                p=subprocess.Popen([python,'scripts/'+script,'--root',str(root),*args],cwd=repo,
                    env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},
                    stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                state['stages'][name]={'state':'running','pid':p.pid};state['phase']=name;save();rc=p.wait()
            state['stages'][name].update(state='complete' if rc==0 else 'failed',returncode=rc);save()
            if rc:raise RuntimeError(f'CPU {name} failed; preserve artifacts')
        state.update(state='complete',report_state='audit_complete_visual_review_pending');save()
    except BaseException:state.update(state='failed',traceback=traceback.format_exc());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
