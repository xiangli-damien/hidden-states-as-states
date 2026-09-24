"""One GPU collection at a time, with mandatory smoke and independent audits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from revision_common import write_json, sha


def run(root):
    lock = (root/'queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Path(__file__).resolve().parents[1]
    python = '/lambda/nfs/dami/openact/.venv/bin/python'
    state = {'pid': os.getpid(), 'state': 'starting', 'started_unix': time.time(),
             'preparation_sha256': sha(root/'plan.json'), 'completed_stages': []}
    def save():
        state['updated_unix'] = time.time(); write_json(root/'queue_status.json', state)
    def execute(stage, script, args, gpu=False):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0' if gpu else '', 'OMP_NUM_THREADS': '4',
               'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
        state.update(state='running', stage=stage); save()
        with (root/(stage+'.log')).open('a') as log:
            child = subprocess.Popen([python, str(source/'scripts'/script), *args], cwd=source, env=env,
                                      stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid; save()
            while child.poll() is None:
                time.sleep(10); save()
            if child.returncode:
                raise RuntimeError(f'{stage} failed ({child.returncode}); inspect saved log')
        state['completed_stages'].append(stage); state['child_pid'] = None; save()
    try:
        save()
        execute('small_model_audit', 'check_revision_sameprompt_capture.py',
                ['--output', str(root/'small_model_audit.json')])
        for stage in ['smoke', 'full']:
            if not (root/stage/'_SUCCESS.json').exists():
                state.update(state='waiting_gpu_idle', stage=stage); save()
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    time.sleep(30); save()
                execute(stage, 'collect_revision_sameprompt.py', ['--root', str(root), '--stage', stage], gpu=True)
            execute(stage+'_audit', 'audit_revision_sameprompt.py', ['--root', str(root), '--stage', stage])
        state.update(state='complete', stage='collection_and_audit_complete'); save()
    except BaseException:
        state.update(state='failed', traceback=traceback.format_exc()); save(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True, type=Path); run(p.parse_args().root)
