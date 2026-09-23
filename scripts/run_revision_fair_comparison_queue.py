"""Wait for the authorized collector, then smoke/audit/full/audit/report on one GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from revision_common import write_json,sha


def main(root,model_python):
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config']
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(__file__).resolve().parents[1]
    state={'pid':os.getpid(),'state':'waiting_collection','started_unix':time.time(),
           'input_plan_sha256':sha(root/'plan.json'),'completed_stages':[]}
    def save():state['updated_unix']=time.time();write_json(root/'queue_status.json',state)
    def stage(name,script,python,smoke=False,gpu=False):
        state.update(state='running',stage=name,child_pid=None);save()
        env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
             'TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'0' if gpu else ''}
        command=[str(python),str(source/'scripts'/script),'--root',str(root),*(['--smoke'] if smoke else [])]
        with (root/(name+'.log')).open('a') as log:
            log.write(json.dumps({'started_unix':time.time(),'command':command})+'\n');log.flush()
            proc=subprocess.Popen(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,env=env)
            state['child_pid']=proc.pid;save()
            while proc.poll() is None:time.sleep(10);save()
            if proc.returncode:raise RuntimeError(f'{name} failed ({proc.returncode}); inspect {name}.log')
        state['completed_stages'].append(name);state['child_pid']=None;save()
    try:
        save();collection=Path(cfg['wait_for_collection'])
        while True:
            parent=json.loads((collection/'queue_status.json').read_text())
            if parent['state']=='failed':raise RuntimeError('GSM8K collector failed; preserve its GPU priority and inspect it')
            if parent['state']=='complete':
                receipt=json.loads((collection/'collection_manifest.json').read_text())
                assert receipt['complete'] and all(r['samples']==1319 for r in receipt['models'].values())
                state['collection_manifest_sha256']=sha(collection/'collection_manifest.json');break
            state['collection_stage']=parent.get('stage');save();time.sleep(30)
        state['state']='waiting_cpu_geometry';save()
        while not (root/'geometry/_SUCCESS.json').exists():
            geometry=json.loads((root/'geometry_status.json').read_text())
            try:os.kill(geometry['pid'],0)
            except ProcessLookupError:raise RuntimeError('CPU geometry stopped without success receipt')
            save();time.sleep(30)
        stage('geometry_audit','audit_revision_fair_geometry.py',sys.executable)
        state['state']='waiting_gpu_idle';save()
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            time.sleep(30);save()
        for smoke in [True,False]:
            label='smoke' if smoke else 'functional'
            if not (root/label/'_SUCCESS.json').exists():
                stage(label,'evaluate_revision_fair_comparison.py',model_python,smoke,gpu=True)
            stage(label+'_audit','audit_revision_fair_comparison.py',model_python,smoke)
        stage('report','report_revision_fair_comparison.py',sys.executable)
        stage('report_audit','audit_revision_fair_report.py',sys.executable)
        state.update(state='complete',stage='complete');save()
    except BaseException:
        state.update(state='failed',traceback=traceback.format_exc());save();raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--model-python',type=Path,default=Path('/lambda/nfs/dami/openact/.venv/bin/python'))
    a=p.parse_args();main(a.root,a.model_python)
