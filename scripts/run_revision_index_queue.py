"""Single follower: fair FA/MFA completion, capability gate, then bounded test."""
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


def run(root):
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'plan.json').read_text())['config']
    model_python='/lambda/nfs/dami/openact/.venv/bin/python'
    state={'pid':os.getpid(),'state':'waiting_fair_comparison','started_unix':time.time(),
           'plan_sha256':sha(root/'plan.json'),'completed_stages':[]}
    def save():state['updated_unix']=time.time();write_json(root/'queue_status.json',state)
    def execute(name,script,python,gpu=False,stage=None):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0' if gpu else '', 'OMP_NUM_THREADS':'4',
             'OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
        command=[str(python),str(source/'scripts'/script),'--root',str(root)]
        if stage:command+=['--stage',stage]
        state.update(state='running',stage=name);save()
        with (root/(name+'.log')).open('a') as log:
            child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            while child.poll() is None:time.sleep(10);save()
            if child.returncode:raise RuntimeError(f'{name} failed: {child.returncode}; inspect its log')
        state['completed_stages'].append(name);state['child_pid']=None;save()
    try:
        save();fair=Path(cfg['wait_for_fair_queue'])
        while True:
            previous=json.loads((fair/'queue_status.json').read_text())
            if previous['state']=='failed':raise RuntimeError('Fair comparison failed; do not bypass predecessor')
            if previous['state']=='complete':
                assert json.loads((fair/'report/statistics_audit.json').read_text())['complete'];break
            state['predecessor_state']=previous['state'];save();time.sleep(30)
        state['state']='waiting_gpu_idle';save()
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            time.sleep(30);save()
        for stage in ['validation','test']:
            if stage=='test' and not json.loads((root/'validation_audit.json').read_text())['capability_gate']['passed']:
                state['scientific_stop']='validation_capability_gate_failed';save();break
            if not (root/stage/'_SUCCESS.json').exists():
                execute(stage,'evaluate_revision_index_interchange.py',model_python,gpu=True,stage=stage)
            execute(stage+'_audit','audit_revision_index_interchange.py',model_python,stage=stage)
        execute('report','report_revision_index_interchange.py',sys.executable)
        execute('report_audit','audit_revision_index_report.py',sys.executable)
        state.update(state='complete',stage='complete');save()
    except BaseException:
        state.update(state='failed',traceback=traceback.format_exc());save();raise
    finally:lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);run(ap.parse_args().root)
