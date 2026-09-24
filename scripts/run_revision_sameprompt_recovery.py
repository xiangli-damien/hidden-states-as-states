"""Explicit audit-only recovery, preserving all original failed queue records."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from revision_common import sha, write_json, freeze


def run(root):
    work = root/'recovery_rms_cuda_20260924'; work.mkdir(exist_ok=True)
    lock = (work/'queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (work/'queue_status.json').exists():
        raise RuntimeError('Recovery already attempted; inspect without overwriting')
    source = Path(__file__).resolve().parents[1]
    assert not (root/'full/audit.json').exists() and not (root/'readouts').exists() and not (root/'report').exists()
    old = json.loads((root/'queue_status.json').read_text())
    assert old['state'] == 'failed' and old['stage'] == 'full_audit' and 'full' in old['completed_stages']
    for queue in ['queue_status.json', 'readout_queue_status.json', 'report_queue_status.json']:
        record = json.loads((root/queue).read_text()); assert record['state'] == 'failed'
        try:
            os.kill(record['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Original queue still alive: '+queue)
    assert json.loads((root/'full/_SUCCESS.json').read_text())['trajectories'] == 896
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    pins = {}
    for name in ['execution_plan.json', 'readout_queue_plan.json', 'report_queue_plan.json']:
        pins[str(root/name)] = sha(root/name)
        pins.update(json.loads((root/name).read_text())['files'])
    for name in ['queue_status.json', 'readout_queue_status.json', 'report_queue_status.json', 'full_audit.log', 'full/_SUCCESS.json']:
        pins[str(root/name)] = sha(root/name)
    for name in ['revision_sameprompt_rms_replay.py', 'audit_revision_sameprompt_cuda.py', Path(__file__).name]:
        pins[str(source/'scripts'/name)] = sha(source/'scripts'/name)
    for name, digest in pins.items():
        assert sha(name) == digest, name
    freeze(work/'plan.json', {'files': pins, 'scope': 'New exact-CUDA RMS audit, then never-started CPU readouts/report; no regeneration or protocol change.'})
    state = dict(state='running', pid=os.getpid(), started_unix=time.time(), completed_stages=[])
    def save():
        state['updated_unix'] = time.time(); write_json(work/'queue_status.json', state)
    openact = '/lambda/nfs/dami/openact/.venv/bin/python'; hss = str(source/'.venv/bin/python')
    stages = [('cuda_check', openact, 'revision_sameprompt_rms_replay.py', ['--output', str(work/'cuda_check.json')], True),
              ('full_audit', openact, 'audit_revision_sameprompt_cuda.py', ['--root', str(root), '--stage', 'full'], True),
              ('fit', hss, 'fit_revision_sameprompt.py', ['--root', str(root)], False),
              ('readout_audit', hss, 'audit_revision_sameprompt_readouts.py', ['--root', str(root)], False),
              ('report', hss, 'report_revision_sameprompt.py', ['--root', str(root)], False),
              ('statistics_audit', hss, 'audit_revision_sameprompt_report.py', ['--root', str(root)], False)]
    try:
        save()
        for stage, python, filename, args, gpu in stages:
            for name, digest in pins.items():
                assert sha(name) == digest, name
            state.update(stage=stage); save()
            with (work/(stage+'.log')).open('x') as log:
                child = subprocess.Popen([python, str(source/'scripts'/filename), *args], cwd=source,
                    stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0' if gpu else '',
                         'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                         'TOKENIZERS_PARALLELISM': 'false'})
                state['child_pid'] = child.pid; save()
                while child.poll() is None:
                    time.sleep(5); save()
                if child.returncode:
                    raise RuntimeError(stage+' failed; preserve the recovery namespace')
            state['completed_stages'].append(stage); state['child_pid'] = None; save()
        state.update(state='complete', stage='statistics_audited_visual_review_pending'); save()
    except BaseException:
        state.update(state='failed', traceback=traceback.format_exc()); save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root)
