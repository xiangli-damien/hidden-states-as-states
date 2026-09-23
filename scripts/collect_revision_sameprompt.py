"""Sample complete answers, then independently teacher-force the frozen prefixes."""
import argparse
import copy
import fcntl
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from openact_eval.evaluators.registry import auto_select_evaluator

from extract_revision_prefixes import load_model
from revision_common import freeze, provenance, sha, write_json, write_npz, status
from revision_sameprompt_io import (read_inputs, stage_rows, valid_prefix, summaries, confidence,
                                    verify_record_files, verify_execution)


@torch.inference_mode()
def capture(model, ids, positions, layers, window):
    """Real per-token block outputs; head readout uses the same forward's norm output."""
    if not positions or positions != list(range(positions[0], positions[-1]+1)):
        raise ValueError('Capture positions must form an actual contiguous span')
    if positions[0] < 0 or positions[-1] >= len(ids):
        raise ValueError('Capture outside supplied prefix')
    captured, calls, handles = {}, {}, []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            assert h.shape[:2] == (1, len(ids))
            calls[layer] = calls.get(layer, 0)+1
            captured[layer] = h[0, positions].float().cpu().numpy()
        handles.append(model.model.layers[layer-1].register_forward_hook(hook))
    try:
        output = model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
        normed = output.last_hidden_state
        post = normed[0, positions].float().cpu().numpy()
        logits = model.lm_head(normed[:, positions[-1]:positions[-1]+1]).float()[0, 0]
        lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        logits = logits.cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    assert all(calls.get(layer) == 1 for layer in layers)
    raw = np.stack([captured[layer] for layer in layers])
    return {'raw': raw, 'post': post, 'positions': np.asarray(positions, np.int64),
            'layers': np.asarray(layers, np.int64), 'logits': logits, 'logprobs': lp,
            **summaries(raw, post, window)}


@torch.inference_mode()
def sample(model, ids, seed, gen_config):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tensor = torch.tensor([ids], device=model.device)
    out = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                         generation_config=gen_config, use_cache=True,
                         return_dict_in_generate=False, output_scores=False, output_hidden_states=False)
    assert out[0, :len(ids)].tolist() == ids
    return out[0, len(ids):].tolist()


def freeze_execution(root, cfg, evaluator):
    source = Path(__file__).parent
    files = [root/'plan.json', root/'inputs.json', Path(inspect.getfile(type(evaluator)))]
    files += [source/name for name in ['collect_revision_sameprompt.py', 'revision_sameprompt_io.py',
        'audit_revision_sameprompt.py', 'check_revision_sameprompt_capture.py',
        'extract_revision_prefixes.py', 'revision_common.py']]
    identity = provenance(cfg, files)
    identity['preparation_sha256'] = sha(root/'plan.json')
    identity['software'] = {'torch': str(torch.__version__), 'transformers': transformers.__version__}
    path = root/'execution_plan.json'
    if path.exists():
        old = verify_execution(root)
        assert all(old[k] == identity[k] for k in ['config', 'files', 'preparation_sha256', 'software'])
    else:
        freeze(path, identity)


def publish_snapshot(root, model, tokenizer, cfg):
    generation = copy.deepcopy(model.generation_config)
    for name, value in cfg['generation'].items():
        setattr(generation, name, value)
    generation.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    freeze(root/'generation_config.json', {'resolved': generation.to_dict(), 'attention': 'sdpa',
        'dtype': 'bfloat16', 'model_revision': cfg['revision'], 'overrides': cfg['generation']})
    gamma = model.model.norm.weight.detach().float().cpu().numpy()
    path = root/'rmsnorm.npz'
    if path.exists():
        with np.load(path) as z:
            np.testing.assert_array_equal(z['weight'], gamma)
    else:
        write_npz(path, weight=gamma)
    freeze(root/'model_snapshot.json', {'model': cfg['model'], 'revision': cfg['revision'],
        'layers': model.config.num_hidden_layers, 'width': model.config.hidden_size,
        'vocab_size': model.config.vocab_size, 'rms_norm_eps': float(model.config.rms_norm_eps),
        'rmsnorm_sha256': sha(path), 'generation_sha256': sha(root/'generation_config.json'),
        'torch': str(torch.__version__), 'transformers': transformers.__version__})
    return generation


def transfer_smoke(root, dest):
    source = root/'smoke'
    audit = json.loads((source/'audit.json').read_text())
    assert audit['complete'] and audit['stage_receipt_sha256'] == sha(source/'_SUCCESS.json')
    for path in sorted((source/'samples').glob('*.json')):
        r = json.loads(path.read_text()); verify_record_files(source, r)
        for name, checksum in r['files'].items():
            target = dest/name; target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source/name, target)
            assert sha(target) == checksum
        target = dest/'samples'/path.name; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        assert sha(target) == sha(path)


def run(root, stage):
    lock = (root/'collection.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan, rows = read_inputs(root); cfg = plan['config']; selected = stage_rows(rows, stage)
    dest = root/stage; dest.mkdir(exist_ok=True)
    if (dest/'_SUCCESS.json').exists():
        raise RuntimeError('Already complete; audit without regenerating')
    assert shutil.disk_usage('/home/ubuntu').free >= 50*2**30
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    evaluator = auto_select_evaluator('math'); freeze_execution(root, cfg, evaluator)
    check = json.loads((root/'small_model_audit.json').read_text())
    assert check['passed'] and check['source_sha256'] == sha(Path(__file__))
    if stage == 'full':
        transfer_smoke(root, dest)
    model, tokenizer = load_model(cfg)
    gen = publish_snapshot(root, model, tokenizer, cfg)
    eos = set(gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id])
    expected = sum(len(row['trajectories']) for row in selected); started = time.monotonic(); done = 0
    status(root, 'collection', phase=stage, state='running', completed=0, expected=expected)
    for row in selected:
        ids = row['prompt_ids']; sid = row['sample_id']
        assert tokenizer(row['model_input_text'], add_special_tokens=False).input_ids == ids
        prompt_file = f'prompts/{sid}.npz'; prompt_path = dest/prompt_file
        prompt_receipt_file = f'prompts/{sid}.json'; prompt_receipt = dest/prompt_receipt_file
        if not prompt_receipt.exists():
            if prompt_path.exists():
                raise RuntimeError('Unsealed prompt artifact; inspect interrupted write before resuming')
            prompt = capture(model, ids, [len(ids)-1], cfg['layers'], cfg['window'])
            write_npz(prompt_path, **prompt)
            write_json(prompt_receipt, {'sample_id': sid, 'prompt_ids': ids,
                'array_sha256': sha(prompt_path), 'execution_plan_sha256': sha(root/'execution_plan.json')})
        prompt_meta = json.loads(prompt_receipt.read_text())
        assert prompt_meta == {'sample_id': sid, 'prompt_ids': ids, 'array_sha256': sha(prompt_path),
                               'execution_plan_sha256': sha(root/'execution_plan.json')}
        prompt_sha = sha(prompt_path)
        for trajectory in row['trajectories']:
            tid = trajectory['trajectory_id']; path = dest/'samples'/f'{tid}.json'
            if path.exists():
                record = json.loads(path.read_text()); verify_record_files(dest, record)
                assert record['trajectory'] == trajectory and record['sample_id'] == sid
                assert record['execution_plan_sha256'] == sha(root/'execution_plan.json')
                assert record['files'][prompt_file] == prompt_sha
                done += 1
                continue
            if shutil.disk_usage('/home/ubuntu').free < 50*2**30:
                raise RuntimeError('SSD reserve below 50 GiB; preserving completed records')
            tick = time.monotonic()
            generated = sample(model, ids, trajectory['seed'], gen)
            assert generated and not eos.intersection(generated[:-1])
            reason = 'eos' if generated[-1] in eos else 'length'
            assert reason == 'eos' or len(generated) == cfg['generation']['max_new_tokens']
            text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            score = evaluator.evaluate_sample(SimpleNamespace(sample_idx=row['source_sample_idx'],
                response_text=text, ground_truth=row['ground_truth'], meta={'sample_id': sid}))
            assert score.is_correct is not None and not score.error
            record = {'sample_id': sid, 'question_group': row['question_group'], 'role': row['role'],
                'trajectory': trajectory, 'generated_ids': generated, 'response_text': text,
                'ground_truth': row['ground_truth'], 'finish_reason': reason, 'length': len(generated),
                'correct': bool(score.is_correct), 'parsed_answer': score.extracted_answer,
                'normalized_answer': score.normalized_answer, 'parse_failed': bool(score.meta['parse_failed']),
                'files': {prompt_file: prompt_sha, prompt_receipt_file: sha(prompt_receipt)},
                'prompt_file': prompt_file, 'prompt_receipt_file': prompt_receipt_file, 'prefixes': {},
                'execution_plan_sha256': sha(root/'execution_plan.json'),
                'generation_config_sha256': sha(root/'generation_config.json')}
            for prefix in cfg['prefixes']:
                valid = valid_prefix(generated, prefix, eos)
                entry = {'valid': valid, 'reason': None if valid else 'ended_before_boundary'}
                if valid:
                    positions = list(range(len(ids), len(ids)+prefix))
                    arrays = capture(model, ids+generated[:prefix], positions, cfg['layers'], cfg['window'])
                    filename = f'features/{tid}_p{prefix}.npz'; write_npz(dest/filename, **arrays)
                    record['files'][filename] = sha(dest/filename)
                    entry.update(file=filename, positions=positions, **confidence(arrays['logits'], arrays['logprobs']))
                record['prefixes'][str(prefix)] = entry
            record['seconds'] = time.monotonic()-tick
            write_json(path, record); done += 1
            status(root, 'collection', phase=stage, state='running', completed=done, expected=expected,
                   seconds=time.monotonic()-started, sample_id=sid)
            print(json.dumps({'phase': stage, 'completed': done, 'expected': expected,
                              'tokens': len(generated), 'seconds': record['seconds']}), flush=True)
    paths = sorted((dest/'samples').glob('*.json'))
    assert done == expected and len(paths) == expected
    receipt = {'complete': True, 'stage': stage, 'questions': len(selected), 'trajectories': done,
        'seconds': time.monotonic()-started, 'smoke_reused': stage == 'full',
        'execution_plan_sha256': sha(root/'execution_plan.json'),
        'generation_config_sha256': sha(root/'generation_config.json'),
        'records': {p.name: sha(p) for p in paths}}
    write_json(dest/'_SUCCESS.json', receipt)
    status(root, 'collection', phase=stage, state='complete', completed=done, expected=expected,
           seconds=receipt['seconds'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True, type=Path)
    p.add_argument('--stage', choices=['smoke', 'full'], required=True); a = p.parse_args()
    try:
        run(a.root, a.stage)
    except BaseException:
        status(a.root, 'collection', phase=a.stage, state='failed', traceback=traceback.format_exc()); raise
