"""Run transfer prerequisites after the existing single-GPU queue releases it."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from revision_common import write_json


def run():
    repo=Path(__file__).resolve().parents[1]
    root=Path('/lambda/nfs/dami/hss/revision-gsm8k-20260923');root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=Path('/lambda/nfs/dami/hss/revision-foundations-20260923')
    cpu=str(repo/'.venv/bin/python');gpu='/lambda/nfs/dami/openact/.venv/bin/python'
    state={'state':'waiting_for_primary_gpu_queue','pid':os.getpid(),'stages':{}}
    def save():
        state['updated_unix']=time.time();write_json(root/'queue_status.json',state)
    def step(name,cmd):
        with (root/(name+'.log')).open('a') as f:
            proc=subprocess.Popen(cmd,cwd=repo,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
            state['stages'][name]={'state':'running','pid':proc.pid,'started_unix':time.time(),'command':cmd};save()
            rc=proc.wait()
        state['stages'][name].update(state='complete' if rc==0 else 'failed',returncode=rc,ended_unix=time.time());save()
        return rc==0
    save()
    while True:
        path=parent/'queue_status.json'
        if path.exists():
            s=json.loads(path.read_text())
            if s['status'] in ('implemented_queue_complete','finished_with_failures'):
                # Popen's supervisor records termination after each child wait.
                break
            try:
                os.kill(s['pid'],0)
            except ProcessLookupError:
                state.update(state='blocked_primary_supervisor_missing');save();return 1
        time.sleep(30)
    if not Path('/lambda/nfs/dami/openact/runs/gsm8k_transfer_20260923/_SUCCESS').exists():
        state['state']='blocked_gsm8k_incomplete';save();return 1
    state['state']='running';save()
    stages=[('prepare',[cpu,'-u','scripts/revision_transfer.py','prepare']),
            ('extract',[gpu,'-u','scripts/extract_revision_prefixes.py','--config',str(root/'target_config.json')]),
            ('confidence',[gpu,'-u','scripts/compute_revision_confidence.py','--config',str(root/'target_config.json')]),
            ('evaluate',[cpu,'-u','scripts/revision_transfer.py','evaluate'])]
    for name,cmd in stages:
        if not step(name,cmd):
            state['state']='failed';save();return 1
    state['state']='complete';save();return 0


if __name__=='__main__':
    raise SystemExit(run())
