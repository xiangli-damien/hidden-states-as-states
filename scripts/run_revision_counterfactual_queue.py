"""Run the small GPU capability gate after transfer extraction releases the GPU."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from revision_common import write_json


def run():
    root=Path('/lambda/nfs/dami/hss/revision-counterfactual-gate-20260923-greedy-v2');root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    repo=Path(__file__).resolve().parents[1]
    state={'state':'waiting_for_transfer_gpu_extraction','pid':os.getpid()}
    def save():
        state['updated_unix']=time.time();write_json(root/'queue_status.json',state)
    save()
    path=Path('/lambda/nfs/dami/hss/revision-gsm8k-20260923/queue_status.json')
    while True:
        if path.exists():
            transfer=json.loads(path.read_text())
            if transfer.get('stages',{}).get('confidence',{}).get('state')=='complete':
                break
            if transfer['state'] in ('failed','blocked_primary_supervisor_missing','blocked_gsm8k_incomplete'):
                state['state']='blocked_upstream_gpu_schedule';save();return 1
        time.sleep(30)
    cmd=['/lambda/nfs/dami/openact/.venv/bin/python','-u','scripts/revision_counterfactual_gate.py']
    with (root/'gate.log').open('a') as log:
        proc=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
        state.update(state='running',child_pid=proc.pid,command=cmd);save();rc=proc.wait()
    state.update(state='complete' if rc==0 else 'failed',returncode=rc);save();return rc


if __name__=='__main__':
    raise SystemExit(run())
