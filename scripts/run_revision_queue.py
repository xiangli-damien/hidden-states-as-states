"""Persisted first-night queue, one GPU decoder process and bounded CPU fitting.

This schedules only implemented prerequisites/reconstruction pilots. It does
not mark E00-E26 or steering/transfer complete when their data become available.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from revision_common import config, write_json


def run(cfg,path):
    repo=Path(__file__).resolve().parents[1]
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    guard=(root/'queue.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu='/lambda/nfs/dami/openact/.venv/bin/python'
    cpu=str(repo/'.venv/bin/python')
    openact=Path('/lambda/nfs/dami/openact')
    state={'status':'waiting_for_extraction','pid':os.getpid(),'stages':{},
           'not_in_this_queue':['MFA comparison','selective steering','controlled donor patching',
                                'frozen-map GSM8K transfer evaluation','online FAR evaluation'],
           'note':'Prerequisite/data/pilot queue. Remaining checklist requires separate implementation and evidence gates.'}
    def save():
        state['updated_unix']=time.time();write_json(root/'queue_status.json',state)
    def launch(name,cmd,cwd):
        log=(root/f'{name}.log').open('a')
        env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
        p=subprocess.Popen(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        log.close();state['stages'][name]={'status':'running','pid':p.pid,'command':cmd,'started_unix':time.time()};save()
        return p
    def finish(name,p):
        rc=p.wait();state['stages'][name].update(status='complete' if rc==0 else 'failed',returncode=rc,ended_unix=time.time());save()
        return rc==0
    save()
    while not (root/'prefixes/_SUCCESS.json').exists():
        f=root/'extract_status.json'
        if f.exists():
            s=json.loads(f.read_text())
            if s['state']=='failed':
                state['status']='blocked_extraction_failed';save();return 1
            try:
                os.kill(s['pid'],0)
            except ProcessLookupError:
                state['status']='blocked_extractor_missing';save();return 1
        time.sleep(15)
    state['status']='running';save()
    if not finish('confidence',launch('confidence',[gpu,'-u','scripts/compute_revision_confidence.py','--config',str(path)],repo)):
        state['status']='failed_confidence';save();return 1
    geometry=launch('geometry',[cpu,'-u','scripts/fit_revision_geometry.py','--config',str(path)],repo)
    smoke=launch('gsm8k_smoke',[gpu,'-u','scripts/run_gsm8k_transfer.py','--smoke',
        '--output','/lambda/nfs/dami/openact/runs/gsm8k_transfer_smoke_20260923',
        '--local-root','/home/ubuntu/openact-gsm8k-smoke-20260923'],openact)
    if finish('gsm8k_smoke',smoke):
        finish('gsm8k_full',launch('gsm8k_full',[gpu,'-u','scripts/run_gsm8k_transfer.py'],openact))
    if finish('geometry',geometry):
        # Functional and free-generation tests are independent of GSM8K collection.
        finish('functional',launch('functional',[gpu,'-u','scripts/evaluate_revision_patches.py','--config',str(path),'--phase','functional'],repo))
        finish('behavior',launch('behavior',[gpu,'-u','scripts/evaluate_revision_patches.py','--config',str(path),'--phase','behavior'],repo))
    finish('report',launch('report',[cpu,'-u','scripts/report_revision_foundations.py','--config',str(path)],repo))
    state['status']='finished_with_failures' if any(s['status']=='failed' for s in state['stages'].values()) else 'implemented_queue_complete'
    save();return int(state['status']=='finished_with_failures')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    a=p.parse_args();raise SystemExit(run(config(a.config),a.config.resolve()))
