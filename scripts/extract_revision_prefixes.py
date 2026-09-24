"""Fresh prompt/prefix activations for matched representation/patch experiments.

Run with OpenAct's pinned GPU Python. Saves actual vectors, never fitting on
labels. Each saved block output is pre-final-RMSNorm, with unpadded batch size 1.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
import zarr
from transformers import AutoModelForCausalLM, AutoTokenizer

from revision_common import config, digest, freeze, provenance, sha, status, write_json, write_npz, OncePatch


def load_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'])
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], revision=cfg['revision'],
        dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
    return model, tokenizer


def question_positions(tokenizer, row):
    text = row.model_input_text
    ids = json.loads(row.prompt_token_ids_json)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded.input_ids != ids:
        raise ValueError('Stored prompt IDs disagree with pinned tokenizer')
    fields = json.loads(row.prompt_fields_json)
    question = fields.get('problem', fields.get('question'))
    # Start searching inside the user message, rather than matching an accidental
    # occurrence in a system instruction. Stored prompt text itself is audited.
    user_start = text.index(row.prompt_text)
    start = text.index(question, user_start)
    end = start + len(question)
    positions = [i for i, (a, b) in enumerate(encoded.offset_mapping) if b > start and a < end]
    if not positions:
        raise ValueError('Question has no token positions')
    return positions


@torch.inference_mode()
def capture(model, ids, layers, positions, mean_positions):
    cache, handles = {}, []
    for layer in layers:
        def hook(module, args, out, layer=layer):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[:2] != (1, len(ids)):
                raise ValueError('Unexpected capture shape')
            cache[layer] = {name: h[0, ix].float().cpu().numpy() for name, ix in positions.items()}
            cache[layer]['all_mean'] = h[0, mean_positions].float().mean(0).cpu().numpy()
        handles.append(model.model.layers[layer-1].register_forward_hook(hook))
    try:
        model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(cache) != set(layers):
        raise ValueError('Incomplete block capture')
    return {name: np.stack([cache[l][name] for l in layers]) for name in cache[layers[0]]}


@torch.inference_mode()
def identity_audit(model, ids, layers, windows):
    inp = torch.tensor([ids], device=model.device)
    baseline = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits.float()
    rows = []
    for layer in layers:
        for positions in windows:
            patch = OncePatch(positions, len(ids), lambda h: h)
            handle = model.model.layers[layer-1].register_forward_hook(patch)
            try:
                value = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits.float()
            finally:
                handle.remove()
            if patch.calls != 1 or not torch.equal(value, baseline):
                raise AssertionError('Identity hook changed logits')
            rows.append({'layer': layer, 'positions': positions, 'calls': patch.calls,
                         'logits_exact': True, 'energy': patch.energy})
    baseline_ids = model.generate(inp, max_new_tokens=8, do_sample=False,
                                 pad_token_id=model.generation_config.pad_token_id)
    patch = OncePatch(windows[-1], len(ids), lambda h: h)
    handle = model.model.layers[layers[-1]-1].register_forward_hook(patch)
    try:
        identity_ids = model.generate(inp, max_new_tokens=8, do_sample=False,
                                     pad_token_id=model.generation_config.pad_token_id)
    finally:
        handle.remove()
    if patch.calls != 1 or not torch.equal(identity_ids, baseline_ids):
        raise AssertionError('Fresh-cache identity generation mismatch')
    return {'logits_checks': rows, 'generation_exact': True,
            'baseline_generated_ids': baseline_ids[0, len(ids):].tolist()}


def run(cfg, limit=None):
    root = Path(cfg['output'])/'prefixes'
    root.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(cfg['source']).glob('shard_*/_COPY_VERIFIED.json'))
    if not shards:
        raise ValueError('No verified source shards')
    # Pin both original data and evaluation/split identities, without hashing TB
    # of activations that this extractor never reads.
    files = [Path(cfg['rows']), Path(cfg['splits']), Path(__file__), Path(__file__).with_name('revision_common.py')]
    for receipt in shards:
        files.extend([receipt, receipt.parent/'manifest.json', receipt.parent/'data.parquet'])
    identity = provenance(cfg, files)
    identity['limit'] = limit
    freeze(root/'plan.json', identity)
    metadata = pd.read_parquet(cfg['rows']).merge(pd.read_parquet(cfg['splits']), on='sample_id', validate='one_to_one')
    if len(metadata) != cfg['expected_rows'] or metadata.question_group.duplicated().any():
        raise ValueError('Expected unique question-level rows')
    model, tokenizer = load_model(cfg)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    done, started = 0, time.monotonic()
    for receipt in shards:
        source = receipt.parent
        frame = pd.read_parquet(source/'data.parquet')
        if limit is not None:
            frame = frame.iloc[:max(0, limit-done)]
        if not len(frame):
            break
        dest = root/source.name
        success = dest/'_SUCCESS.json'
        if success.exists():
            saved = json.loads(success.read_text())
            if saved['plan_key'] != digest(identity):
                raise ValueError('Stale extraction shard')
            for name, value in saved['sha256'].items():
                if sha(dest/name) != value:
                    raise ValueError('Extraction checksum mismatch')
            done += saved['rows']
            continue
        dest.mkdir(parents=True, exist_ok=True)
        z = zarr.open_consolidated(str(source/'tensors.zarr'), mode='r')
        ptr = np.asarray(z['tokens/sample_ptr'])
        tokenids = np.asarray(z['tokens/ids'])
        buffers = {prefix: [] for prefix in cfg['prefixes']}
        manifests = []
        for row in frame.itertuples():
            prompt = json.loads(row.prompt_token_ids_json)
            response = tokenids[ptr[row.sample_idx]:ptr[row.sample_idx+1]].tolist()
            qp = question_positions(tokenizer, row)
            if len(prompt) < cfg['window'] or len(qp) < cfg['window']:
                # Question-tail is padded with NaN, never borrow chat/instruction
                # tokens. Its explicit validity flag drives downstream filtering.
                pass
            if not (root/'identity_audit.json').exists():
                audit = identity_audit(model, prompt, cfg['layers'],
                    [list(range(len(prompt)-n, len(prompt))) for n in (1, 4, 16)])
                write_json(root/'identity_audit.json', audit)
            entry = {'sample_id': row.sample_id, 'source': str(source), 'sample_idx': row.sample_idx,
                     'prompt_ids': prompt, 'response_ids': response, 'question_positions': qp}
            manifests.append(entry)
            for prefix in cfg['prefixes']:
                valid = prefix == 0 or (len(response) > prefix and not eos.intersection(response[:prefix]))
                width = model.config.hidden_size
                blank = lambda n: np.full((len(cfg['layers']), n, width), np.nan, dtype=np.float32)
                if not valid:
                    buffers[prefix].append({'window': blank(cfg['window']), 'question_window': blank(cfg['window']),
                        'all_mean': blank(1)[:, 0], 'valid': False, 'question_valid': False})
                    continue
                ids = prompt + response[:prefix]
                positions = {'window': list(range(len(ids)-cfg['window'], len(ids)))}
                if len(qp) >= cfg['window']:
                    positions['question_window'] = qp[-cfg['window']:]
                result = capture(model, ids, cfg['layers'], positions,
                                 list(range(len(prompt), len(ids))) if prefix else list(range(len(prompt))))
                result.setdefault('question_window', blank(cfg['window']))
                result.update(valid=True, question_valid=len(qp) >= cfg['window'])
                buffers[prefix].append(result)
            done += 1
            if done % 20 == 0:
                status(cfg['output'], 'extract', state='running', completed=done,
                       expected=limit or cfg['expected_rows'], seconds=time.monotonic()-started)
                print(json.dumps({'stage': 'extract', 'rows': done, 'seconds': time.monotonic()-started}), flush=True)
        for prefix, records in buffers.items():
            write_npz(dest/f'prefix_{prefix}.npz', **{k: np.stack([r[k] for r in records]) for k in records[0]})
        write_json(dest/'tokens.json', manifests)
        selected = metadata.set_index('sample_id').loc[[r['sample_id'] for r in manifests]].reset_index()
        selected.to_parquet(dest/'rows.parquet', index=False)
        write_json(success, {'plan_key': digest(identity), 'rows': len(manifests),
                   'sha256': {p.name: sha(p) for p in dest.iterdir() if p.name != '_SUCCESS.json'}})
    expected = limit or cfg['expected_rows']
    if done != expected:
        raise ValueError(f'Incomplete extraction {done}/{expected}')
    write_json(root/'_SUCCESS.json', {'rows': done, 'plan_key': digest(identity), 'elapsed_seconds': time.monotonic()-started})
    status(cfg['output'], 'extract', state='complete', completed=done, expected=expected)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    cfg = config(a.config)
    try:
        run(cfg, a.limit)
    except BaseException:
        status(cfg['output'], 'extract', state='failed', traceback=traceback.format_exc())
        raise
