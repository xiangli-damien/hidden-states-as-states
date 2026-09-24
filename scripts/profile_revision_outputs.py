"""CPU-only, hash-checked profiling of saved locality logits and reference losses.

No model is loaded. The pinned tokenizer is used only to describe saved token IDs.
Source runs are immutable; this writes a separate result namespace.
"""
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from revision_common import sha, write_json, freeze, provenance


def method_key(c):
    return '_'.join(str(c[k]) for k in ('method', 'rank', 'alpha', 'seed') if k in c)


def fragment_category(text, special=False, unused=False):
    if unused:
        return 'unused_vocab_id'
    if special:
        return 'special_token'
    if not text or text.isspace():
        return 'whitespace'
    s = text.strip()
    if re.fullmatch(r'[+−-]?\d+(?:[.,]\d+)*', s):
        return 'number_fragment'
    if re.fullmatch(r'[+−*/=<>^%×÷≤≥≈±]+', s):
        return 'operator_fragment'
    if '\\' in s or '$' in s:
        return 'latex_or_dollar_fragment'
    if s.isalpha():
        return 'alphabetic_fragment'
    if all(not c.isalnum() for c in s):
        return 'punctuation_fragment'
    return 'mixed_or_other_fragment'


def distribution_profile(base, changed):
    """Keep original saved log-softmax values; do not silently renormalize."""
    base, changed = np.asarray(base, dtype=np.float64), np.asarray(changed, dtype=np.float64)
    assert base.ndim == changed.ndim == 1 and base.shape == changed.shape
    assert np.isfinite(base).all() and np.isfinite(changed).all()
    p, q = np.exp(base), np.exp(changed)
    np.testing.assert_allclose([p.sum(), q.sum()], 1, atol=2e-6, rtol=0)
    term = p * (base - changed)
    return p, q, term, float(np.abs(q-p).sum()/2)


def loss_windows(loss):
    a = np.asarray(loss, dtype=np.float64)
    assert a.ndim == 1 and len(a) and np.isfinite(a).all()
    windows = {'first_token': a[:1], 'first16': a[:16], 'positions2_16': a[1:16],
               'after16': a[16:], 'full': a}
    return {name: float(v.mean()) if len(v) else None for name, v in windows.items()}


def source_records(root, cfg):
    stage = root/'functional'
    audit = json.loads((stage/'audit.json').read_text())
    plan = json.loads((stage/'plan.json').read_text())
    assert audit['complete'] and audit['conditions'] == audit['expected']
    assert audit['plan_sha256'] == sha(stage/'plan.json')
    manifest = json.loads((stage/'audit_inputs.json').read_text())
    selected = []
    expected = []
    def include(t):
        return (all(t[k] == v for k, v in cfg['view'].items())
                and str(t['width']) in cfg['methods_by_width']
                and method_key(t['condition']) in cfg['methods_by_width'][str(t['width'])])
    for t in plan['expected']:
        if include(t):
            expected.append(t['name'])
    for name in sorted(expected):
        path = stage/'samples'/f'{name}.json'
        assert sha(path) == manifest[path.name]
        record = json.loads(path.read_text())
        assert include(record['task']) and record['task']['name'] == name
        selected.append((path, record))
    assert len(selected) == len(set(expected))
    return selected, plan


def metadata(root, selected):
    plan = json.loads((root/'plan.json').read_text())
    tokens, rows, used = {}, {}, {}
    wanted = {r['task']['sample_id'] for _, r in selected}
    for marker in sorted((Path(plan['config']['foundation'])/'prefixes').glob('shard_*/_SUCCESS.json')):
        receipt = json.loads(marker.read_text())
        rp, tp = marker.parent/'rows.parquet', marker.parent/'tokens.json'
        # Read only metadata, never touch the many-GB hidden-state arrays.
        assert sha(rp) == plan['files'][str(rp)] == receipt['sha256']['rows.parquet']
        frame = pd.read_parquet(rp)
        subset = frame.loc[frame.sample_id.isin(wanted)]
        if not len(subset):
            continue
        assert sha(tp) == plan['files'][str(tp)] == receipt['sha256']['tokens.json']
        used[str(rp)], used[str(tp)] = sha(rp), sha(tp)
        for r in subset.to_dict('records'):
            assert r['sample_id'] not in rows
            rows[r['sample_id']] = r
        for r in json.loads(tp.read_text()):
            if r['sample_id'] in wanted:
                assert r['sample_id'] not in tokens
                tokens[r['sample_id']] = r
    assert set(rows) == set(tokens) == wanted
    return rows, tokens, used


def run(config_path):
    from transformers import AutoTokenizer
    start = time.time()
    cfg = json.loads(config_path.read_text())
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    if (root/'_SUCCESS.json').exists():
        raise RuntimeError('Completed profile exists; preserve it rather than overwrite.')
    sources = {k: Path(v) for k, v in cfg['sources'].items()}
    files = [config_path, Path(__file__), Path(__file__).with_name('revision_common.py')]
    for source in sources.values():
        files.extend(source/p for p in ['plan.json', 'functional/plan.json', 'functional/audit.json',
                                       'functional/audit_inputs.json', 'report/statistics_audit.json'])
    freeze(root/'plan.json', provenance(cfg, files))
    model_cfg = json.loads((sources['math']/'plan.json').read_text())['config']
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['model'], revision=model_cfg['revision'],
                                               local_files_only=True)
    vocab = tokenizer.batch_decode([[i] for i in range(len(tokenizer))],
                                  skip_special_tokens=False, clean_up_tokenization_spaces=False)
    special = set(tokenizer.all_special_ids)
    vocabulary = [fragment_category(s, i in special) for i, s in enumerate(vocab)]
    write_json(root/'tokenizer.json', {'model': model_cfg['model'], 'revision': model_cfg['revision'],
        'vocabulary': vocab, 'categories': vocabulary, 'special_ids': sorted(special)})
    records, categories, inputs = [], [], {}
    for source_name, source in sources.items():
        selected, _ = source_records(source, cfg)
        rows, tokens, used = metadata(source, selected); inputs.update(used)
        by_question = {}
        for path, r in selected:
            sid = r['task']['sample_id']
            by_question.setdefault(sid, []).append((path, r))
        for sid, group in sorted(by_question.items()):
            token = tokens[sid]; reference = token['response_ids'][16:]
            cohort = source_name + '_' + rows[sid]['split']
            lookup, payload = {}, {'sample_id': sid, 'cohort': cohort,
                'scope': 'exploratory_discovery' if cohort == 'math_validation' else 'descriptive_review_only',
                'prompt': rows[sid]['prompt_text'], 'reference': rows[sid]['response_text'],
                'ground_truth': str(rows[sid]['ground_truth']),
                'prefix_text': tokenizer.decode(token['response_ids'][:16], skip_special_tokens=False),
                'reference_ids': reference,
                'reference_fragments': tokenizer.batch_decode([[i] for i in reference],
                    skip_special_tokens=False, clean_up_tokenization_spaces=False),
                'conditions': {}}
            for path, r in group:
                arr = path.with_suffix('.npz')
                assert sha(arr) == r['arrays_sha256']
                inputs[str(path)], inputs[str(arr)] = sha(path), r['arrays_sha256']
                with np.load(arr, allow_pickle=False) as f:
                    logp, losses = f['logp'].astype(np.float64), f['reference_nll'].astype(np.float64)
                assert len(losses) == len(reference) == r['reference_tokens']
                np.testing.assert_allclose([losses.mean(), losses[:16].mean(), losses[0]],
                    [r['nll'], r['first16_nll'], r['first_token_nll']], atol=1e-12, rtol=1e-10)
                np.testing.assert_allclose(losses[0], -logp[reference[0]], atol=1e-7, rtol=1e-7)
                lookup[(r['task']['width'], method_key(r['task']['condition']))] = (r, logp, losses)
            assert len(lookup) == sum(map(len, cfg['methods_by_width'].values()))
            for (width, method), (r, logp, losses) in sorted(lookup.items()):
                _, base, base_loss = lookup[(width, 'identity')]
                p, q, term, tv = distribution_profile(base, logp)
                np.testing.assert_allclose(term.sum(), r['next_token_kl'], atol=1e-10, rtol=1e-10)
                cats = np.asarray(vocabulary + ['unused_vocab_id']*(len(p)-len(vocab)))
                assert len(cats) == len(p)
                names = sorted(set(cats)); values = loss_windows(losses); baseline = loss_windows(base_loss)
                row = {'source': source_name, 'cohort': cohort, 'sample_id': sid, 'width': width,
                    'method': method, 'reference_tokens': len(reference), 'next_token_kl': float(term.sum()),
                    'total_variation': tv, 'argmax_changed': bool(p.argmax() != q.argmax()),
                    'baseline_entropy': float(-(p*base).sum()),
                    'mse_per_coordinate': sum(r['geometry']['actual']['token_delta_energy'])/(width*3584)}
                for name, value in values.items():
                    row['nll_'+name] = value
                    row['delta_nll_'+name] = value-baseline[name] if value is not None else None
                records.append(row)
                for name in names:
                    ix = cats == name
                    categories.append({k: row[k] for k in ['cohort', 'sample_id', 'width', 'method']} | {
                        'category': name, 'kl_signed_sum': float(term[ix].sum()),
                        'kl_positive_sum': float(np.maximum(term[ix], 0).sum()),
                        'kl_negative_sum': float(np.minimum(term[ix], 0).sum()),
                        'probability_change': float((q-p)[ix].sum()),
                        'total_variation': float(np.abs(q-p)[ix].sum()/2),
                        'baseline_mass': float(p[ix].sum()), 'changed_mass': float(q[ix].sum())})
                if width != 16:
                    continue
                topn = cfg['top_tokens']
                selectors = {'baseline_top': np.argsort(-p)[:topn],
                    'largest_probability_change': np.argsort(-np.abs(q-p))[:topn],
                    'largest_positive_kl_terms': np.argsort(-term)[:topn],
                    'most_negative_kl_terms': np.argsort(term)[:topn]}
                ids = sorted(set(map(int, np.concatenate(list(selectors.values())))))
                details = []
                for i in ids:
                    details.append({'id': i, 'fragment': vocab[i] if i < len(vocab) else f'<unused:{i}>',
                        'category': str(cats[i]), 'p_original': float(p[i]), 'p_changed': float(q[i]),
                        'delta_probability': float(q[i]-p[i]), 'signed_kl_term': float(term[i]),
                        'selected_by': [name for name, ix in selectors.items() if i in ix]})
                payload['conditions'][method] = {'metrics': row, 'top_candidates': details,
                    'reference_nll': losses.tolist(), 'delta_reference_nll': (losses-base_loss).tolist(),
                    'top_abs_probability_change_coverage': float(np.abs(q-p)[selectors['largest_probability_change']].sum()/max(2*tv, 1e-300))}
            write_json(root/'questions'/f'{source_name}_{sid}.json', payload)
            print(f'{source_name} {sid}: {len(lookup)} conditions checked', flush=True)
    pd.DataFrame(records).to_parquet(root/'conditions.parquet', index=False)
    pd.DataFrame(categories).to_parquet(root/'categories.parquet', index=False)
    write_json(root/'input_hashes.json', inputs)
    outputs = list(root.glob('*.parquet')) + list((root/'questions').glob('*.json')) + [root/'tokenizer.json', root/'input_hashes.json', root/'plan.json']
    write_json(root/'_SUCCESS.json', {'complete': True, 'conditions': len(records),
        'questions': len(list((root/'questions').glob('*.json'))), 'seconds': time.time()-start,
        'files': {str(p.relative_to(root)): sha(p) for p in outputs},
        'checks': ['selected source JSON manifests and NPZ SHA', 'exact planned condition coverage',
                   'original reference IDs and length', 'raw loss aggregation and first-token logp',
                   'saved full-vocabulary KL and probability normalization'], 'gpu_calls': 0})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True, type=Path)
    run(ap.parse_args().config)
