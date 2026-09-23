"""Independently re-score saved free generations; never load a decoder model.

Run with the OpenAct environment for its actual MATH evaluator and the pinned
tokenizer. A partial audit only validates execution, not scientific effects.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, GenerationConfig
from openact_eval.evaluators.registry import auto_select_evaluator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, partial=False):
    stage = root/'behavior'
    marker = stage/'_SUCCESS.json'
    complete = json.loads(marker.read_text()) if marker.exists() else None
    if complete is None and not partial:
        raise ValueError('Behavior stage incomplete; --partial is execution-only')
    plan = json.loads((stage/'plan.json').read_text()); cfg = plan['config']
    selected = set(plan['sample_ids']); tokens, rows, validity = {}, {}, {}
    for path in sorted((root/'prefixes').glob('shard_*/_SUCCESS.json')):
        frame = pd.read_parquet(path.parent/'rows.parquet')
        items = json.loads((path.parent/'tokens.json').read_text())
        np.testing.assert_array_equal(frame.sample_id, [i['sample_id'] for i in items])
        for i, item in enumerate(items):
            sid = item['sample_id']
            if sid in selected:
                tokens[sid] = item; rows[sid] = frame.iloc[i]
        for prefix in cfg['functional_prefixes']:
            with np.load(path.parent/f'prefix_{prefix}.npz') as data:
                for i, sid in enumerate(frame.sample_id):
                    if sid in selected: validity[sid, prefix] = bool(data['valid'][i])
    if set(tokens) != selected: raise ValueError('Missing selected question')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'], local_files_only=True)
    gen = GenerationConfig.from_pretrained(cfg['model'], revision=cfg['revision'], local_files_only=True)
    eos = set(gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id])
    evaluator = auto_select_evaluator('math'); expected = {}
    for sid, item in tokens.items():
        for prefix in cfg['functional_prefixes']:
            valid = len(item['response_ids']) > prefix and not eos.intersection(item['response_ids'][:prefix])
            assert validity[sid, prefix] == valid
            if not valid: continue
            length = len(item['prompt_ids'])+prefix
            if length < 16: continue
            for layer in plan['layers']:
                for width in cfg['functional_widths']:
                    for method in plan['methods']:
                        name = f'{sid}_p{prefix}_l{layer}_tokens_w{width}_{method}.json'
                        expected[name] = dict(sample_id=sid, split=rows[sid].split,
                            prefix_tokens=prefix, layer=layer, role='tokens', width=width,
                            method=method, positions=list(range(length-width, length)))
    paths = sorted((stage/'samples').glob('*.json')); identities = {}; hashes = {}
    for path in paths:
        if path.name not in expected: raise ValueError('Unexpected condition '+path.name)
        record = json.loads(path.read_text()); hashes[path.name] = sha(path)
        for key, value in expected[path.name].items():
            if record[key] != value: raise ValueError((path.name, key, record[key], value))
        sid = record['sample_id']; prefix = record['prefix_tokens']; ids = record['generated_ids']
        assert ids and len(ids) == record['length'] and len(ids) <= cfg['generation_budget']-prefix
        assert all(type(t) is int and 0 <= t < len(tokenizer) for t in ids)
        assert not eos.intersection(ids[:-1])
        reason = 'eos' if ids[-1] in eos else 'length'
        assert record['finish_reason'] == reason
        if reason == 'length': assert len(ids) == cfg['generation_budget']-prefix
        assert np.isfinite(record['actual_patch_energy']) and record['actual_patch_energy'] >= 0
        assert record['original_correct'] == int(rows[sid].label)
        assert record['ground_truth'] == str(rows[sid].ground_truth)
        assert record['suffix'] == tokenizer.decode(ids, skip_special_tokens=True)
        text = tokenizer.decode(tokens[sid]['response_ids'][:prefix]+ids, skip_special_tokens=True)
        assert record['response_text'] == text
        score = evaluator.evaluate_sample(SimpleNamespace(sample_idx=int(rows[sid].sample_idx),
            response_text=text, ground_truth=str(rows[sid].ground_truth), meta={'sample_id': sid}))
        assert score.is_correct is not None and not score.error
        assert record['correct'] == bool(score.is_correct)
        assert record['normalized_answer'] == score.normalized_answer
        assert record['parsed_answer'] == score.extracted_answer
        assert record['parse_failed'] == bool(score.meta['parse_failed'])
        if record['method'] == 'identity':
            assert record['actual_patch_energy'] == 0
            key = sid, prefix
            if key in identities: assert identities[key] == ids
            else: identities[key] = ids
    if complete is not None:
        assert {p.name for p in paths} == set(expected)
        assert len(paths) == complete['conditions'] == complete['expected_conditions']
        assert len(selected) == complete['selected_questions']
        saved = pd.read_parquet(stage/'per_question.parquet')
        source = pd.DataFrame([json.loads(p.read_text()) for p in paths])
        sort = ['sample_id', 'prefix_tokens', 'layer', 'role', 'width', 'method']
        for column in ['correct', 'normalized_answer', 'parsed_answer', 'parse_failed', 'response_text',
                       'length', 'finish_reason', 'actual_patch_energy']:
            np.testing.assert_array_equal(source.sort_values(sort)[column], saved.sort_values(sort)[column])
    label = 'execution_audit' if complete is not None else 'execution_audit_partial'
    hash_path = stage/(label+'_input_hashes.json')
    hash_path.write_text(json.dumps(hashes, indent=2)+'\n')
    result = {'stage_complete': complete is not None, 'snapshot_conditions': len(paths),
        'expected_conditions': len(expected), 'selected_questions': len(selected),
        'identity_prefixes': len(identities), 'actual_saved_outputs_rescored_with_OpenAct_MATH': True,
        'positions_token_ids_decode_budget_finish_reason_labels_and_identity_checks_passed': True,
        'scope': 'Execution integrity, not correctness-effect interpretation',
        'plan_sha256': sha(stage/'plan.json'), 'input_hashes_sha256': sha(hash_path),
        'audit_code_sha256': sha(Path(__file__))}
    (stage/(label+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--partial', action='store_true')
    args = parser.parse_args(); run(args.root, args.partial)
