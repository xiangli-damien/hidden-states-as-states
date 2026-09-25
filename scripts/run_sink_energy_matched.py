"""Frozen per-token BF16 energy-matched orthogonal controls; no generation."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
import torch

from revision_common import sha, write_json, write_npz, freeze
from extract_revision_prefixes import load_model
from hss_followup_common import verify_plan, verify_record, receipt
from hss_followup_tools import make_torch_transform
from run_hss_followup import logprobs, layer_tokens


class MatchedTransform:
    def __init__(self, axis, tau, direction, cfg, device):
        self.axis = torch.as_tensor(axis, dtype=torch.float32, device=device)
        self.direction = None if direction is None else torch.as_tensor(direction, dtype=torch.float32, device=device)
        self.tau, self.cfg, self.arrays, self.metrics = float(tau), cfg, None, None

    def __call__(self, h):
        assert self.arrays is None, 'Expected one full-sequence block call'
        x = h.float()
        excess = torch.clamp(x @ self.axis-self.tau, min=0.)
        reference = (x-excess[:, None]*self.axis).to(h.dtype)
        target = (reference.float()-x).norm(dim=-1)
        active = target > 0
        if self.direction is None:
            after, scale = reference, excess
        else:
            def candidate(scale):
                return (x-scale[:, None]*self.direction).to(h.dtype)
            def norm(scale):
                return (candidate(scale).float()-x).norm(dim=-1)
            lo = torch.zeros_like(target)
            hi = torch.where(active, target*2+1e-4, torch.zeros_like(target))
            for _ in range(self.cfg['bracket_expansions']):
                hi = torch.where(norm(hi) < target, hi*2, hi)
            assert bool((norm(hi) >= target).all()), 'Could not bracket target norm'
            for _ in range(self.cfg['bisection_steps']):
                mid = (lo+hi)/2
                below = norm(mid) < target
                lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
            scale = torch.where((norm(lo)-target).abs() <= (norm(hi)-target).abs(), lo, hi)
            after = candidate(scale)
        actual_delta = after.float()-x
        actual = actual_delta.norm(dim=-1)
        allowed = self.cfg['norm_match_relative_tolerance']*target+self.cfg['norm_match_absolute_tolerance']
        error = (actual-target).abs()
        match = error <= allowed
        same_mask = torch.equal(actual > 0, active)
        target_energy = float(target.square().sum())
        energy_error = abs(float(actual.square().sum())-target_energy)/max(target_energy, 1e-30)
        cosine = (actual_delta @ self.axis) / actual.clamp_min(1e-30) / self.axis.norm()
        self.arrays = dict(target_norm=target.cpu().numpy(), actual_norm=actual.cpu().numpy(),
            active=active.cpu().numpy(), actual_cosine_sink=cosine.cpu().numpy(), scale=scale.cpu().numpy())
        self.metrics = dict(n_tokens=len(h), active=int(active.sum()), same_active_mask=same_mask,
            all_tokens_within_tolerance=bool(match.all()),
            maximum_absolute_norm_error=float(error.max()),
            maximum_tolerance_fraction=float((error/allowed).max()),
            relative_total_energy_error=energy_error,
            mean_target_norm=float(target.mean()), mean_actual_norm=float(actual.mean()),
            actual_sink_cosine_abs_max=float(cosine[active].abs().max()) if active.any() else 0.,
            actual_sink_cosine_weighted_rms=float(((cosine.square()*actual.square()).sum()/actual.square().sum().clamp_min(1e-30)).sqrt()))
        assert same_mask and bool(match.all()) and energy_error <= self.cfg['relative_total_energy_tolerance'], self.metrics
        return after


def selftest(cfg, device='cpu'):
    torch.set_num_threads(4)
    rng = np.random.default_rng(401)
    D = 3584
    s = rng.normal(size=D); s /= np.linalg.norm(s)
    v = rng.normal(size=D); v -= v.dot(s)*s; v /= np.linalg.norm(v)
    x = rng.normal(size=(40, D)) + np.linspace(-2, 5, 40)[:, None]*s
    h = torch.tensor(x, dtype=torch.bfloat16, device=device)
    ref = make_torch_transform('clamp', s_hat=s, tau=1., device=device)(h)
    c1 = MatchedTransform(s, 1., None, cfg, device)
    assert torch.equal(c1(h), ref)
    matched = MatchedTransform(s, 1., v, cfg, device)
    after = matched(h)
    assert torch.equal(after[~torch.tensor(matched.arrays['active'], device=device)], h[~torch.tensor(matched.arrays['active'], device=device)])
    assert np.max(np.abs(np.asarray(v).dot(s))) < 1e-12
    return dict(passed=True, device=str(device), c1_matches_original_exactly=True, matched=matched.metrics)


def prepare(config_path):
    cfg = json.loads(Path(config_path).read_text())
    root, source = Path(cfg['output']), Path(cfg['source'])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        plan = json.loads((root/'plan.json').read_text())
        assert plan['config'] == cfg
        for p, h in plan['files'].items():
            assert sha(p) == h, p
        return cfg, plan
    verify_plan(source)
    verify_record(source/'step2_vectors.npz', source)
    table = pd.read_parquet(source/'step2_screen_long.parquet')
    assert sha(source/'step2_screen_long.parquet') == cfg['source_table_sha256']
    table = table[table.condition == 'NONE']
    rows = table[(table['set'] == 'normal') | table.m_loop.notna()].sort_values(['set','question_id'])
    assert rows['set'].value_counts().to_dict() == {'normal': cfg['expected_normal'], 'sink': cfg['expected_sink']}
    cohort = {x['sample_id']: x for x in json.loads((source/'cohort.json').read_text())}
    split = json.loads((source/'split.json').read_text())
    assert set(rows[rows['set']=='normal'].question_id) == set(split['M_B'])
    assert all(cohort[q]['correct'] for q in split['M_B'])
    v = dict(np.load(source/'step2_vectors.npz'))
    axis = v['s_hat'].astype(np.float64); unit = axis / np.linalg.norm(axis)
    original = v['controls'].astype(np.float64)
    orth = original-(original@unit)[:, None]*unit
    norms = np.linalg.norm(orth, axis=1)
    assert len(orth) == 10 and (norms > 1e-6).all()
    orth /= norms[:, None]
    assert np.max(abs(orth@unit)) < cfg['axis_orthogonality_tolerance']
    write_npz(root/'directions.npz', axis=v['s_hat'], tau=v['tau'], original_controls=original, orthogonal_controls=orth)
    cases = [dict(question_id=r.question_id, set=r.set, cut=int(r.cut), cut_rule=r.cut_rule,
                  n_tokens=len(cohort[r.question_id]['response_ids'])) for r in rows.itertuples()]
    write_json(root/'cases.json', cases)
    source_files = [source/n for n in ['plan.json','cohort.json','split.json','step2_vectors.npz','step2_screen_long.parquet']]
    code_files = [Path(config_path).resolve(), Path(__file__).resolve()]
    code_files += [Path(__file__).with_name(n) for n in ['run_hss_followup.py','extract_revision_prefixes.py','revision_common.py','hss_followup_tools.py','hss_followup_common.py']]
    files = source_files+code_files+[root/'directions.npz',root/'cases.json']
    plan = dict(config=cfg, files={str(p):sha(p) for p in files},
        frozen_utc=datetime.now(timezone.utc).isoformat(), git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        expected_records=len(cases)*len(cfg['conditions']), orthogonality_max=float(abs(orth@unit).max()),
        excluded_sink_ids=table[(table['set']=='sink') & table.m_loop.isna()].question_id.tolist(),
        normal_selection='All100 selected by frozen automatic correctness and boxed; not independently human-verified.')
    freeze(root/'plan.json', plan)
    return cfg, plan


def run(config_path, preflight_only=False):
    cfg, plan = prepare(config_path)
    root, source = Path(cfg['output']), Path(cfg['source'])
    lock=(root/'gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(), 'GPU occupied'
    write_json(root/'status.json', dict(state='loading_model', started_unix=time.time(), expected=plan['expected_records']))
    cpu_check = selftest(cfg)
    model, tok = load_model(cfg)
    gpu_check = selftest(cfg, model.device)
    cohort = {x['sample_id']: x for x in json.loads((source/'cohort.json').read_text())}
    directions = dict(np.load(root/'directions.npz'))
    axis, tau, orth = directions['axis'], float(directions['tau']), directions['orthogonal_controls']
    split = json.loads((source/'split.json').read_text())
    # Numeric checks on A-side questions, not the196 scoring questions.
    check_ids = [sorted(split[k])[0] for k in ['S_A','N_A']]
    checks=[]
    for sid in check_ids:
        row=cohort[sid]; response=row['response_ids'][:128]
        base,_=logprobs(model,row['prompt_ids'],response)
        identity,_=logprobs(model,row['prompt_ids'],response,lambda h:h)
        np.testing.assert_array_equal(base,identity)
        x=layer_tokens(model,dict(row,response_ids=response))
        for i,v in enumerate(orth):
            op=MatchedTransform(axis,tau,v,cfg,model.device)
            op(torch.as_tensor(x,device=model.device).to(torch.bfloat16))
            checks.append(dict(sample_id=sid,control=i,**op.metrics))
        old,_=logprobs(model,row['prompt_ids'],response,make_torch_transform('clamp',s_hat=axis,tau=tau,device=model.device))
        new,_=logprobs(model,row['prompt_ids'],response,MatchedTransform(axis,tau,None,cfg,model.device))
        np.testing.assert_array_equal(old,new)
    write_json(root/'preflight_SUCCESS.json',dict(cpu=cpu_check,gpu=gpu_check,real_checks=checks,
        identity_logprobs_exact=True,c1_old_new_logprobs_exact=True,gpu_name=torch.cuda.get_device_name(),
        torch=torch.__version__,tokenizer_revision=tok.init_kwargs.get('_commit_hash')))
    if preflight_only:
        return
    cases=json.loads((root/'cases.json').read_text())
    reference=pd.read_parquet(source/'step2_screen_long.parquet').set_index(['question_id','condition'])
    start=time.time();records=[]
    for case in cases:
        sid=case['question_id'];row=cohort[sid]
        for ci,cond in enumerate(cfg['conditions']):
            path=root/'records'/sid/(cond+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                rec=verify_record(path,root);assert sha(path.with_suffix('.npz'))==rec['arrays_sha256']
                item=json.loads(path.read_text())
            else:
                tick=time.time()
                op=None if cond=='NONE' else MatchedTransform(axis,tau,None if cond=='C1' else orth[ci-2],cfg,model.device)
                lp,geometry=logprobs(model,row['prompt_ids'],row['response_ids'],op)
                metric=float(-lp[case['cut']:].mean(dtype=np.float64)) if case['set']=='sink' else float(-lp.mean(dtype=np.float64))
                replay_error=None
                if cond in ['NONE','C1']:
                    col='m_loop' if case['set']=='sink' else 'm_col_nll'
                    replay_error=abs(metric-float(reference.loc[(sid,cond),col]))
                    assert replay_error<=cfg['baseline_replay_nll_tolerance'], (sid,cond,replay_error)
                arrays=dict(logprobs=lp)
                if op is not None:arrays.update(op.arrays)
                write_npz(path.with_suffix('.npz'),**arrays)
                item=dict(**case,condition=cond,nll=metric,seconds=time.time()-tick,geometry=geometry,
                    energy_match=None if op is None else op.metrics,replay_error=replay_error,
                    arrays_sha256=sha(path.with_suffix('.npz')))
                write_json(path,item);receipt(path,root,dict(arrays_sha256=item['arrays_sha256']))
            records.append(item)
            elapsed=time.time()-start
            write_json(root/'status.json',dict(state='running',completed=len(records),expected=plan['expected_records'],
                elapsed_seconds=elapsed,last=sid+'/'+cond,estimated_remaining_seconds=elapsed/max(1,len(records))*(plan['expected_records']-len(records)),
                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,updated_unix=time.time()))
    flat=[{k:v for k,v in x.items() if k not in ['geometry','energy_match']} for x in records]
    pd.DataFrame(flat).to_parquet(root/'scores.parquet',index=False)
    write_json(root/'GPU_COMPLETE.json',dict(records=len(records),seconds=time.time()-start,
        table_sha256=sha(root/'scores.parquet'),plan_sha256=sha(root/'plan.json'),
        source_hashes_unchanged=all(sha(p)==h for p,h in plan['files'].items())))
    assert all(sha(p)==h for p,h in plan['files'].items())
    write_json(root/'status.json',dict(state='complete',completed=len(records),expected=plan['expected_records'],
        elapsed_seconds=time.time()-start,updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--selftest',action='store_true');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    try:
        if a.selftest: print(json.dumps(selftest(json.loads(Path(a.config).read_text())),indent=2))
        else:run(a.config,a.preflight_only)
    except BaseException:
        cfg=json.loads(Path(a.config).read_text());root=Path(cfg['output'])
        if root.exists():write_json(root/'failure.json',dict(traceback=traceback.format_exc(),updated_unix=time.time()))
        raise
