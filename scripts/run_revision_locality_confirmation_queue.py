"""Audited smoke -> frozen target interventions -> independent audits/report."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from revision_common import config,status,write_json


def run(cfg):
    base=Path(cfg['output']);root=base/'interventions'
    lock=(base/'functional_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (base/'collection_audit.json').exists() and (root/'plan.json').exists()
    if (base/'functional_queue_SUCCESS.json').exists():
        print('Complete queue exists; do not restart.');return
    gpu='/lambda/nfs/dami/openact/.venv/bin/python';cpu=str(Path.cwd()/'.venv/bin/python')
    path=str(root/'evaluation_config.json')
    stages=[('smoke',gpu,['scripts/evaluate_revision_locality.py','--config',path,'--smoke'],True),
        ('smoke_audit',gpu,['scripts/audit_revision_locality.py','--root',str(root),'--smoke'],False),
        ('functional',gpu,['scripts/evaluate_revision_locality.py','--config',path],True),
        ('functional_audit',gpu,['scripts/audit_revision_locality.py','--root',str(root)],False),
        ('report',cpu,['scripts/report_revision_locality_confirmation.py','--root',str(root)],False),
        ('statistics_audit',cpu,['scripts/audit_revision_locality_report.py','--root',str(root)],False)]
    deadline=time.monotonic()+4*3600
    for name,python,args,uses_gpu in stages:
        while uses_gpu and subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            if time.monotonic()>deadline:raise TimeoutError('GPU occupied')
            status(base,'functional_queue',state='waiting',stage_name=name);time.sleep(20)
        if shutil.disk_usage('/home/ubuntu').free<50*1024**3:raise RuntimeError('SSD reserve below50GiB')
        env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
        if not uses_gpu:env['CUDA_VISIBLE_DEVICES']=''
        with (base/(name+'.log')).open('a') as log:
            child=subprocess.Popen([python,*args],stdout=log,stderr=subprocess.STDOUT,env=env)
            status(base,'functional_queue',state='running',stage_name=name,child_pid=child.pid)
            code=child.wait()
        if code:raise RuntimeError(f'{name} exited{code}; inspect without changing frozen outcomes')
        if name=='smoke_audit':
            audit=json.loads((root/'smoke/audit.json').read_text());assert audit['complete'] and audit['conditions']==audit['expected']
    write_json(base/'functional_queue_SUCCESS.json',{'completed_unix':time.time(),'stages':[s[0] for s in stages]})
    status(base,'functional_queue',state='complete',stage_name='audited_confirmation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);cfg=config(p.parse_args().config)
    try:run(cfg)
    except BaseException:
        status(cfg['output'],'functional_queue',state='failed',traceback=traceback.format_exc());raise
