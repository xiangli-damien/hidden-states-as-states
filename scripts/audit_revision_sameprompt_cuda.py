"""Recovery audit: identical raw contracts, exact same-device RMSNorm replay.

The failed CPU audit and its frozen source remain unchanged. This audit loads
only the saved RMSNorm weights, never the language model or a new generation.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer
from openact_eval.evaluators.registry import auto_select_evaluator
from revision_common import sha, write_json
from revision_sameprompt_io import read_inputs, verify_execution
from revision_sameprompt_rms_replay import exact_replay


def check_arrays(path, positions, cfg, model, gamma):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    d, n = model['width'], len(positions)
    assert a['raw'].shape == (len(cfg['layers']), n, d) and a['post'].shape == (n, d)
    np.testing.assert_array_equal(a['positions'], positions)
    np.testing.assert_array_equal(a['layers'], cfg['layers'])
    for k, value in a.items():
        assert np.isfinite(value).all(), k
        if k not in ['positions', 'layers']:
            assert value.dtype == np.float32, k
    for key, source, axis in [('current', a['raw'][:, -1], None),
        ('mean_window', a['raw'][:, -cfg['window']:], 1), ('mean_all', a['raw'], 1),
        ('post_current', a['post'][-1], None), ('post_mean_window', a['post'][-cfg['window']:], 0),
        ('post_mean_all', a['post'], 0)]:
        expected = source if axis is None else np.mean(source.astype(np.float64), axis=axis).astype(np.float32)
        np.testing.assert_array_equal(a[key], expected)
    x = torch.from_numpy(a['raw'][cfg['layers'].index(model['layers'])]).float()
    unit = (x*torch.rsqrt(x.square().mean(-1, keepdim=True)+model['rms_norm_eps'])).to(torch.bfloat16)
    expected = (unit*torch.from_numpy(gamma).to(torch.bfloat16)).float().numpy()
    cpu_bad = np.abs(a['post']-expected) > (.002+np.abs(expected)/128)
    replay = exact_replay(a['raw'][cfg['layers'].index(model['layers'])], a['post'], positions, gamma, model['rms_norm_eps'])
    rms_relative = float(np.linalg.norm(a['post']-expected)/max(np.linalg.norm(expected), 1e-12))
    logits, lp = a['logits'].astype(np.float64), a['logprobs'].astype(np.float64)
    assert logits.shape == lp.shape == (model['vocab_size'],)
    offset = logits.max(); logz = offset+np.log(np.exp(logits-offset).sum())
    np.testing.assert_allclose(lp, logits-logz, rtol=0, atol=1e-5)
    assert abs(np.exp(lp).sum()-1) < 2e-6
    order = np.sort(logits)
    return {'entropy': float(-(np.exp(lp)*lp).sum()), 'margin': float(order[-1]-order[-2]),
            'rms_relative_error': rms_relative, 'cpu_tolerance_violations': int(cpu_bad.sum()),
            'same_device_exact_elements': replay['checked_elements']}


def run(root, stage):
    if (root/stage/'audit.json').exists():
        raise RuntimeError('An audit receipt already exists; preserve it and inspect')
    import subprocess
    assert torch.cuda.is_available(), 'Exact replay requires the original CUDA device'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    plan, rows = read_inputs(root); cfg = plan['config']; verify_execution(root)
    selected = [r for r in rows if r['role'] == 'train'][:2] if stage == 'smoke' else rows
    expected = {t['trajectory_id']: (r, t) for r in selected for t in r['trajectories']}
    dest = root/stage; receipt = json.loads((dest/'_SUCCESS.json').read_text())
    assert receipt['complete'] and receipt['questions'] == len(selected) and receipt['trajectories'] == len(expected)
    assert receipt['execution_plan_sha256'] == sha(root/'execution_plan.json')
    model = json.loads((root/'model_snapshot.json').read_text())
    assert model['rmsnorm_sha256'] == sha(root/'rmsnorm.npz')
    assert model['generation_sha256'] == receipt['generation_config_sha256'] == sha(root/'generation_config.json')
    generation = json.loads((root/'generation_config.json').read_text())
    assert all(generation['resolved'][k] == v for k, v in cfg['generation'].items())
    eos = generation['resolved']['eos_token_id']; eos = set(eos if isinstance(eos, list) else [eos])
    with np.load(root/'rmsnorm.npz') as z:
        gamma = z['weight'].copy()
    tok = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'], local_files_only=True)
    evaluator = auto_select_evaluator('math'); prompts = {}; valid_counts = {str(p): 0 for p in cfg['prefixes']}
    prompt_checks = {}; largest_rms = 0.; correct = 0; length_ended = 0; roles = {}
    exact_elements = 0; cpu_violations = 0; cpu_failure_artifacts = []
    paths = sorted((dest/'samples').glob('*.json'))
    assert {p.stem for p in paths} == set(expected) and set(receipt['records']) == {p.name for p in paths}
    for path in paths:
        assert sha(path) == receipt['records'][path.name]
        r = json.loads(path.read_text()); source, trajectory = expected[path.stem]
        assert r['trajectory'] == trajectory and r['sample_id'] == source['sample_id']
        assert r['role'] == source['role'] and r['question_group'] == source['question_group']
        assert r['execution_plan_sha256'] == receipt['execution_plan_sha256']
        assert r['generation_config_sha256'] == receipt['generation_config_sha256']
        roles.setdefault(r['role'], set()).add(r['question_group'])
        ids = r['generated_ids']; assert ids and len(ids) == r['length'] <= cfg['generation']['max_new_tokens']
        assert all(type(i) is int and 0 <= i < model['vocab_size'] for i in ids)
        assert not eos.intersection(ids[:-1])
        reason = 'eos' if ids[-1] in eos else 'length'; assert r['finish_reason'] == reason
        if reason == 'length':
            assert len(ids) == cfg['generation']['max_new_tokens']; length_ended += 1
        text = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert text == r['response_text'] and r['ground_truth'] == source['ground_truth']
        assert tok(source['model_input_text'], add_special_tokens=False).input_ids == source['prompt_ids']
        score = evaluator.evaluate_sample(SimpleNamespace(sample_idx=source['source_sample_idx'], response_text=text,
            ground_truth=source['ground_truth'], meta={'sample_id': source['sample_id']}))
        assert score.is_correct is not None and not score.error
        assert r['correct'] == bool(score.is_correct) and r['parse_failed'] == bool(score.meta['parse_failed'])
        assert r['parsed_answer'] == score.extracted_answer and r['normalized_answer'] == score.normalized_answer
        correct += int(r['correct']); required = {r['prompt_file'], r['prompt_receipt_file']}
        for filename, checksum in r['files'].items():
            target = (dest/filename).resolve(); assert target.is_relative_to(dest.resolve())
            assert sha(target) == checksum
        sid = source['sample_id']; prompt_hash = r['files'][r['prompt_file']]
        pm = json.loads((dest/r['prompt_receipt_file']).read_text())
        assert pm == {'sample_id': sid, 'prompt_ids': source['prompt_ids'], 'array_sha256': prompt_hash,
                      'execution_plan_sha256': receipt['execution_plan_sha256']}
        if sid in prompts:
            assert prompts[sid] == prompt_hash
        else:
            prompts[sid] = prompt_hash
            prompt_checks[sid] = check_arrays(dest/r['prompt_file'], [len(source['prompt_ids'])-1], cfg, model, gamma)
            pc = prompt_checks[sid]; exact_elements += pc['same_device_exact_elements']
            cpu_violations += pc['cpu_tolerance_violations']; largest_rms = max(largest_rms, pc['rms_relative_error'])
            if pc['cpu_tolerance_violations']:
                cpu_failure_artifacts.append({'file': r['prompt_file'], **pc})
        assert set(r['prefixes']) == set(valid_counts)
        for prefix in cfg['prefixes']:
            entry = r['prefixes'][str(prefix)]
            valid = len(ids) > prefix and not eos.intersection(ids[:prefix])
            assert entry['valid'] == valid
            if not valid:
                assert entry == {'valid': False, 'reason': 'ended_before_boundary'}; continue
            valid_counts[str(prefix)] += 1; required.add(entry['file'])
            positions = list(range(len(source['prompt_ids']), len(source['prompt_ids'])+prefix))
            assert entry['positions'] == positions and entry['reason'] is None
            check = check_arrays(dest/entry['file'], positions, cfg, model, gamma)
            np.testing.assert_allclose([entry['entropy'], entry['margin']], [check['entropy'], check['margin']], atol=1e-10)
            largest_rms = max(largest_rms, check['rms_relative_error'])
            exact_elements += check['same_device_exact_elements']; cpu_violations += check['cpu_tolerance_violations']
            if check['cpu_tolerance_violations']:
                cpu_failure_artifacts.append({'file': entry['file'], **check})
        assert set(r['files']) == required
    assert sum(len(s) for s in roles.values()) == len(set.union(*roles.values())) == len(selected)
    result = {'complete': True, 'stage': stage, 'questions': len(selected), 'trajectories': len(expected),
        'correct': correct, 'length_ended': length_ended, 'valid_prefix_counts': valid_counts,
        'unique_prompt_states': len(prompts), 'largest_rms_relative_error': largest_rms,
        'execution_plan_sha256': receipt['execution_plan_sha256'], 'stage_receipt_sha256': sha(dest/'_SUCCESS.json'),
        'audit_source_sha256': sha(Path(__file__)),
        'rms_replay_source_sha256': sha(Path(__file__).with_name('revision_sameprompt_rms_replay.py')),
        'rms_check': 'exact same-CUDA bf16 replay at original full sequence shape; no tolerance relaxation',
        'exact_rms_elements': exact_elements, 'cpu_tolerance_violations': cpu_violations,
        'cpu_failure_artifacts': cpu_failure_artifacts,
        'preserved_original_failure_log_sha256': sha(root/'full_audit.log'),
        'independent_math_rescoring_means_confidence_rms_ids_roles_and_file_hashes_verified': True}
    write_json(dest/'audit.json', result); print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True, type=Path)
    p.add_argument('--stage', choices=['smoke', 'full'], required=True); a = p.parse_args(); run(a.root, a.stage)
