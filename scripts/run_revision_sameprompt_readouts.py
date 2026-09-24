"""CPU follower for the audited full collection; no decoder or new grid search."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from revision_common import freeze, write_json, sha


def run(root):
    lock = (root/'readout_queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Path(__file__).resolve().parents[1]
    files = [source/'scripts'/name for name in ['fit_revision_sameprompt.py', 'audit_revision_sameprompt_readouts.py',
        'revision_sameprompt_io.py', 'revision_common.py']]
    pinned = {str(p): sha(p) for p in files}
    freeze(root/'readout_queue_plan.json', {'preparation_sha256': sha(root/'plan.json'), 'files': pinned})
    state = {'state': 'waiting_collection_audit', 'pid': os.getpid(), 'started_unix': time.time(), 'completed_stages': []}
    def save():
        state['updated_unix'] = time.time(); write_json(root/'readout_queue_status.json', state)
    try:
        save()
        while True:
            collection = json.loads((root/'queue_status.json').read_text())
            if collection['state'] == 'failed':
                raise RuntimeError('Collection failed; do not bypass its audit')
            if collection['state'] == 'complete':
                audit = json.loads((root/'full/audit.json').read_text())
                assert audit['complete'] and audit['stage_receipt_sha256'] == sha(root/'full/_SUCCESS.json')
                break
            time.sleep(30); save()
        for stage, filename in [('fit', 'fit_revision_sameprompt.py'), ('audit', 'audit_revision_sameprompt_readouts.py')]:
            for path, digest in pinned.items():
                assert sha(path) == digest, path
            if stage == 'fit' and (root/'readouts/_SUCCESS.json').exists():
                continue
            state.update(state='running', stage=stage); save()
            with (root/f'readout_{stage}.log').open('a') as log:
                child = subprocess.Popen([sys.executable, str(source/'scripts'/filename), '--root', str(root)],
                    cwd=source, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                         'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
                state['child_pid'] = child.pid; save()
                while child.poll() is None:
                    time.sleep(10); save()
                if child.returncode:
                    raise RuntimeError(f'Readout {stage} failed; inspect its log')
            state['completed_stages'].append(stage); state['child_pid'] = None; save()
        state.update(state='complete', stage='readouts_audited_report_pending'); save()
    except BaseException:
        state.update(state='failed', traceback=traceback.format_exc()); save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root)
