"""One CPU report follower; waits for both frozen upstream audits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from revision_common import sha, freeze, write_json


def run(root):
    lock = (root/'report_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Path(__file__).resolve().parents[1]
    names = ['report_revision_sameprompt.py', 'audit_revision_sameprompt_report.py',
             'revision_sameprompt_report_metrics.py', 'revision_sameprompt_statistics.py',
             'revision_common.py', Path(__file__).name]
    pinned = {str(source/'scripts'/name): sha(source/'scripts'/name) for name in names}
    freeze(root/'report_queue_plan.json', {'preparation_sha256': sha(root/'plan.json'), 'files': pinned})
    state = dict(state='waiting_readout_audit', pid=os.getpid(), started_unix=time.time(), completed_stages=[])
    def save():
        state['updated_unix'] = time.time(); write_json(root/'report_queue_status.json', state)
    try:
        save()
        while True:
            upstream = json.loads((root/'readout_queue_status.json').read_text())
            if upstream['state'] == 'failed':
                raise RuntimeError('Readout queue failed; preserve outputs and stop')
            if upstream['state'] == 'complete':
                audit = json.loads((root/'readouts/audit.json').read_text())
                assert audit['complete'] and audit['readout_receipt_sha256'] == sha(root/'readouts/_SUCCESS.json')
                break
            time.sleep(30); save()
        for stage, filename in [('build', 'report_revision_sameprompt.py'), ('audit', 'audit_revision_sameprompt_report.py')]:
            for name, digest in pinned.items():
                assert sha(name) == digest, name
            if stage == 'build' and (root/'report/_SUCCESS.json').exists():
                continue
            state.update(state='running', stage=stage); save()
            with (root/f'report_{stage}.log').open('a') as log:
                process = subprocess.Popen([sys.executable, str(source/'scripts'/filename), '--root', str(root)],
                    cwd=source, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                         'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
                state['child_pid'] = process.pid; save()
                while process.poll() is None:
                    time.sleep(10); save()
                if process.returncode:
                    raise RuntimeError(f'Report {stage} failed; inspect log and preserve partial directory')
            state['completed_stages'].append(stage); state['child_pid'] = None; save()
        audit = json.loads((root/'report/statistics_audit.json').read_text())
        assert audit['complete'] and audit['report_receipt_sha256'] == sha(root/'report/_SUCCESS.json')
        state.update(state='complete', stage='statistics_audited_visual_review_pending'); save()
    except BaseException:
        state.update(state='failed', traceback=traceback.format_exc()); save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root)
