"""Follow the existing CPU fit with audited smoke, real pilot, audit and report."""
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


def run(cfg,path):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+8*3600
    while not (root/'fit_SUCCESS.json').exists():
        state=root/'fit_status.json'
        if state.exists() and json.loads(state.read_text())['state']=='failed':raise RuntimeError('CPU fit failed')
        if time.monotonic()>deadline:raise TimeoutError('CPU fit not ready')
        status(root,'queue',state='waiting',stage_name='fit');time.sleep(20)
    gpu='/lambda/nfs/dami/openact/.venv/bin/python'
    cpu=str(Path.cwd()/'.venv/bin/python')
    jobs=[('smoke',gpu,['scripts/evaluate_revision_locality.py','--config',path,'--smoke'],True),
        ('smoke_audit',gpu,['scripts/audit_revision_locality.py','--root',str(root),'--smoke'],False),
        ('functional',gpu,['scripts/evaluate_revision_locality.py','--config',path],True),
        ('functional_audit',gpu,['scripts/audit_revision_locality.py','--root',str(root)],False),
        ('report',cpu,['scripts/report_revision_locality.py','--root',str(root)],False)]
    for name,python,args,uses_gpu in jobs:
        if not Path(args[0]).exists():raise FileNotFoundError(args[0])
        while uses_gpu and subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            if time.monotonic()>deadline:raise TimeoutError('GPU occupied')
            status(root,'queue',state='waiting',stage_name=name,reason='GPU in use');time.sleep(20)
        if shutil.disk_usage('/home/ubuntu').free<50*1024**3:raise RuntimeError('SSD reserve below50GiB')
        env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
        if not uses_gpu:env['CUDA_VISIBLE_DEVICES']=''
        with (root/f'{name}.log').open('a') as log:
            child=subprocess.Popen([python,*args],stdout=log,stderr=subprocess.STDOUT,env=env)
            status(root,'queue',state='running',stage_name=name,child_pid=child.pid)
            code=child.wait()
        if code:raise RuntimeError(f'{name} exited{code}')
        if name=='smoke_audit':
            result=json.loads((root/'smoke/audit.json').read_text())
            assert result['complete'] and result['conditions']==result['expected']
    write_json(root/'queue_SUCCESS.json',{'completed_unix':time.time(),'stages':[j[0] for j in jobs]})
    status(root,'queue',state='complete',stage_name='all_implemented_locality_stages')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=config(a.config)
    try:run(cfg,a.config)
    except BaseException:
        status(cfg['output'],'queue',state='failed',traceback=traceback.format_exc());raise
