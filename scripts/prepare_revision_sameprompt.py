"""Freeze real prompt IDs and disjoint question roles without generating answers."""
import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from revision_common import config, freeze, provenance, sha, write_json
from revision_sameprompt_common import select_questions, trajectory_seed


def run(cfg, config_path):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        plan = json.loads((root/'plan.json').read_text())
        assert plan['config'] == cfg
        for name, digest in plan['files'].items():
            assert sha(name) == digest, name
        assert sha(root/'inputs.json') == plan['inputs_sha256']
        print(json.dumps({'state': 'existing_preparation_verified', 'questions': plan['questions']}))
        return
    metadata = pd.read_parquet(cfg['rows']).merge(pd.read_parquet(cfg['splits']), on='sample_id', validate='one_to_one')
    selected = select_questions(metadata[['sample_id', 'question_group', 'split']].to_dict('records'),
                                cfg['role_counts'], cfg['selection_seed'])
    by_id = {r['sample_id']: r for r in selected}; original = {}; files = [Path(config_path), Path(__file__),
        Path(__file__).with_name('revision_sameprompt_common.py'), Path(__file__).with_name('revision_common.py'),
        Path(cfg['rows']), Path(cfg['splits']),
        Path(__file__).parents[1]/'docs/revision-sameprompt-plan-20260923.zh-CN.md']
    for marker in sorted(Path(cfg['source']).glob('shard_*/_COPY_VERIFIED.json')):
        path = marker.parent/'data.parquet'
        frame = pd.read_parquet(path)
        subset = frame.loc[frame.sample_id.isin(by_id)]
        if not len(subset):
            continue
        files.extend([marker, path, marker.parent/'manifest.json'])
        for r in subset.to_dict('records'):
            sid = r['sample_id']
            if sid in original:
                raise ValueError(f'Duplicate original row {sid}')
            original[sid] = (r, str(marker.parent))
    if set(original) != set(by_id):
        raise ValueError('Selected source prompts incomplete')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'])
    rows = []; seeds = set(); lengths = []
    for item in selected:
        r, source_shard = original[item['sample_id']]
        ids = json.loads(r['prompt_token_ids_json'])
        if tokenizer(r['model_input_text'], add_special_tokens=False).input_ids != ids:
            raise ValueError(f'Source prompt/tokenizer mismatch {item["sample_id"]}')
        if not ids or len(ids) + cfg['generation']['max_new_tokens'] > cfg['max_context_tokens']:
            raise ValueError('Selected prompt does not fit the fixed context budget')
        source_metadata = metadata.loc[metadata.sample_id.eq(item['sample_id'])].iloc[0]
        assert str(source_metadata.ground_truth) == str(r['ground_truth'])
        repeats = []
        for repetition in range(cfg['repetitions']):
            seed = trajectory_seed(cfg['selection_seed'], item['question_group'], repetition)
            if seed in seeds:
                raise ValueError('Generation seed collision')
            seeds.add(seed)
            repeats.append({'trajectory_id': f'{item["sample_id"]}_r{repetition}', 'repetition': repetition, 'seed': seed})
        rows.append({**item, 'source_shard': source_shard, 'source_sample_idx': int(r['sample_idx']),
            **{name: str(r[name]) for name in ['prompt_text', 'model_input_text', 'ground_truth', 'category', 'level', 'prompt_template_hash']},
            'prompt_ids': ids, 'trajectories': repeats})
        lengths.append(len(ids))
    write_json(root/'inputs.json', rows)
    plan = provenance(cfg, files)
    plan.update(questions=len(rows), trajectories=len(seeds), inputs_sha256=sha(root/'inputs.json'),
        tokenizer_prompts_verified=len(rows), prompt_token_range=[min(lengths), max(lengths)],
        scope=cfg['scope'], generation_started=False)
    freeze(root/'plan.json', plan)
    write_json(root/'preparation.json', {'complete': True, 'questions': len(rows), 'trajectories': len(seeds),
        'role_counts': cfg['role_counts'], 'plan_sha256': sha(root/'plan.json'),
        'prompt_token_range': plan['prompt_token_range'], 'generation_started': False})
    print(json.dumps(json.loads((root/'preparation.json').read_text())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args(); run(config(args.config), args.config)
