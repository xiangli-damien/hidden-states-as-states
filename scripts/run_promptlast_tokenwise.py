"""Frozen prompt-last map applied at every step of six full reference answers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from revision_common import config, freeze, provenance, sha, write_json, write_npz, status
from promptlast_tokenwise_common import names, selected_cases, trajectory, metrics


def prepare(cfg):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists(): return verify(root)
    src = Path(cfg['source']); old = json.loads((src/'plan.json').read_text())
    for name in ['decoder.npz', 'cases.json']:
        assert sha(src/name) == old['files'][str(src/name)]
    cases = selected_cases(json.loads((src/'cases.json').read_text()), cfg['per_dataset'])
    assert len(cases) == 6 and len({c['sample_id'] for c in cases}) == 6
    write_json(root/'cases.json', cases); shutil.copyfile(src/'decoder.npz', root/'decoder.npz')
    files = [src/f for f in ['plan.json', 'decoder.npz', 'cases.json']]
    files += [root/'cases.json', root/'decoder.npz']
    files += [Path(__file__).with_name(f) for f in ['run_promptlast_tokenwise.py', 'promptlast_tokenwise_common.py',
              'audit_report_promptlast_tokenwise.py', 'promptlast_replacement_common.py', 'revision_common.py', 'extract_revision_prefixes.py']]
    plan = provenance(cfg, files)
    plan.update(conditions=names(cfg['methods']), questions=len(cases), predictions=sum(len(c['response_ids']) for c in cases),
                source_plan_sha256=sha(src/'plan.json'), selection='First three per dataset in frozen source order; all stored response positions.')
    freeze(root/'plan.json', plan)
    return plan


def verify(root):
    plan = json.loads((root/'plan.json').read_text())
    for p, h in plan['files'].items():
        if sha(p) != h: raise ValueError('Frozen input changed: ' + p)
    return plan


def run(cfg, smoke=False):
    import torch
    from extract_revision_prefixes import load_model
    root = Path(cfg['output']); plan = verify(root)
    lock = (root/'gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dest = root/('smoke' if smoke else 'functional'); dest.mkdir(exist_ok=True)
    if (dest/'_SUCCESS.json').exists(): return
    if not smoke: assert json.loads((root/'smoke/audit.json').read_text())['complete']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    assert shutil.disk_usage('/home/ubuntu').free > 30 * 1024**3
    cases = json.loads((root/'cases.json').read_text())
    if smoke: cases = selected_cases(cases, 1)
    with np.load(root/'decoder.npz') as z: d = {k:z[k].copy() for k in z.files}
    torch.set_num_threads(cfg['threads']); model, tokenizer = load_model(cfg)
    started = time.monotonic(); total = 0
    expected = sum(min(len(c['response_ids']), cfg['smoke_positions']) if smoke else len(c['response_ids']) for c in cases)
    for c in cases:
        sid = c['sample_id']; out = dest/sid; out.mkdir(exist_ok=True)
        count = min(len(c['response_ids']), cfg['smoke_positions']) if smoke else len(c['response_ids'])
        if (out/'_SUCCESS.json').exists():
            saved = json.loads((out/'_SUCCESS.json').read_text()); assert saved['plan_sha256'] == sha(root/'plan.json')
            for f, h in saved['files'].items(): assert sha(out/f) == h
            total += count; continue
        # Partial cases replay from clean caches; complete cases are immutable.
        if list(out.glob('chunk_*.json')): raise RuntimeError('Partial case requires a separate recovery namespace: ' + sid)
        buffer = []; records = []; files = {}; case_start = time.monotonic()
        for r in trajectory(model, c['prompt_ids'], c['response_ids'][:count], d, cfg['layer'], cfg['methods']):
            row = {k:r[k] for k in ['position', 'input_id', 'target_id']}
            row.update(metrics(r['logp'], r['x'], r['actual'], r['target_id'], d))
            row.update(input_text=tokenizer.decode([r['input_id']]), target_text=tokenizer.decode([r['target_id']]))
            buffer.append(r); records.append(row); total += 1
            if len(buffer) == cfg['chunk'] or r['position'] + 1 == count:
                base = f"chunk_{buffer[0]['position']:05d}"
                write_npz(out/(base+'.npz'), **{k:np.stack([a[k] for a in buffer]) for k in ['position','input_id','target_id','logp','x','actual']})
                write_json(out/(base+'.json'), {'rows':records, 'plan_sha256':sha(root/'plan.json'), 'arrays_sha256':sha(out/(base+'.npz'))})
                for suffix in ['.npz','.json']: files[base+suffix] = sha(out/(base+suffix))
                buffer = []; records = []
                status(root, 'smoke' if smoke else 'functional', state='running', completed=total, expected=expected,
                       seconds=time.monotonic()-started, sample_id=sid, position=r['position'], gpu_peak_bytes=torch.cuda.max_memory_allocated())
                print(json.dumps({'predictions':total, 'expected':expected, 'seconds':time.monotonic()-started, 'sample_id':sid}), flush=True)
        write_json(out/'_SUCCESS.json', {'sample_id':sid, 'dataset':c['dataset'], 'predictions':count, 'files':files,
                   'plan_sha256':sha(root/'plan.json'), 'seconds':time.monotonic()-case_start})
    write_json(dest/'_SUCCESS.json', {'predictions':total, 'questions':len(cases), 'seconds':time.monotonic()-started,
               'plan_sha256':sha(root/'plan.json'), 'gpu_peak_bytes':torch.cuda.max_memory_allocated()})
    status(root, 'smoke' if smoke else 'functional', state='complete', completed=total, expected=expected)


def queue(cfg, path):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    lock = (root/'queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpu = Path(__file__).resolve().parents[1]/'.venv/bin/python'
    gpu = Path('/lambda/nfs/dami/openact/.venv/bin/python')
    jobs = [('prepare',cpu,'run_promptlast_tokenwise.py',['--stage','prepare']),
            ('smoke',gpu,'run_promptlast_tokenwise.py',['--stage','smoke']),
            ('smoke_audit',gpu,'audit_report_promptlast_tokenwise.py',['--smoke']),
            ('functional',gpu,'run_promptlast_tokenwise.py',['--stage','evaluate']),
            ('audit',gpu,'audit_report_promptlast_tokenwise.py',[]),
            ('report',cpu,'audit_report_promptlast_tokenwise.py',['--report'])]
    env = os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[k] = str(cfg['threads'])
    state = {'state':'running','pid':os.getpid(),'started_unix':time.time(),'stages':{}}
    for name, python, script, extra in jobs:
        state['phase'] = name
        with (root/(name+'.log')).open('a') as log:
            child = subprocess.Popen([str(python),str(Path(__file__).with_name(script)),'--config',str(Path(path).resolve()),*extra], env=env, stdout=log, stderr=log)
            state['stages'][name] = {'state':'running','pid':child.pid,'started_unix':time.time()}; write_json(root/'queue_status.json',state)
            rc = child.wait()
        state['stages'][name].update(state='complete' if rc == 0 else 'failed', returncode=rc, ended_unix=time.time())
        if rc:
            state.update(state='failed'); write_json(root/'queue_status.json',state); raise RuntimeError(name+' failed')
    state.update(state='complete',finished_unix=time.time(),delivery='visual_review_pending'); write_json(root/'queue_status.json',state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['prepare','smoke','evaluate','queue'],required=True); a=p.parse_args(); cfg=config(a.config)
    if a.stage == 'queue': queue(cfg,a.config)
    elif a.stage == 'prepare': prepare(cfg)
    else: run(cfg,a.stage == 'smoke')
