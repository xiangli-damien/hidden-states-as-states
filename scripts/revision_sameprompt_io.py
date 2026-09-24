"""Serializable contracts for the frozen same-prompt collection; no model imports."""
import json
from pathlib import Path
import numpy as np
from revision_common import sha


def read_inputs(root):
    root = Path(root)
    plan = json.loads((root/'plan.json').read_text())
    ready = json.loads((root/'preparation.json').read_text())
    assert ready['complete'] and ready['plan_sha256'] == sha(root/'plan.json')
    for name, checksum in plan['files'].items():
        assert sha(name) == checksum, name
    assert sha(root/'inputs.json') == plan['inputs_sha256']
    return plan, json.loads((root/'inputs.json').read_text())


def stage_rows(rows, stage):
    if stage == 'smoke':
        selected = [r for r in rows if r['role'] == 'train'][:2]
        assert len(selected) == 2
        return selected
    if stage == 'full':
        return rows
    raise ValueError(stage)


def valid_prefix(ids, prefix, eos):
    return len(ids) > prefix and not set(ids[:prefix]).intersection(eos)


def summaries(raw, post, window):
    raw, post = np.asarray(raw), np.asarray(post)
    if raw.ndim != 3 or post.shape != raw.shape[1:] or not raw.shape[1]:
        raise ValueError('Invalid layer, token, hidden axes')
    if not np.isfinite(raw).all() or not np.isfinite(post).all():
        raise ValueError('Nonfinite activation')
    return {'current': raw[:, -1].copy(),
            'mean_window': raw[:, -window:].mean(axis=1, dtype=np.float64).astype(np.float32),
            'mean_all': raw.mean(axis=1, dtype=np.float64).astype(np.float32),
            'post_current': post[-1].copy(),
            'post_mean_window': post[-window:].mean(axis=0, dtype=np.float64).astype(np.float32),
            'post_mean_all': post.mean(axis=0, dtype=np.float64).astype(np.float32)}


def confidence(logits, logprobs):
    logits, lp = np.asarray(logits, float), np.asarray(logprobs, float)
    if logits.shape != lp.shape or logits.ndim != 1 or not np.isfinite(lp).all():
        raise ValueError('Invalid next-token distribution')
    if not np.isclose(np.exp(lp).sum(), 1., atol=2e-6):
        raise ValueError('Unnormalized log probabilities')
    top = np.sort(logits)[-2:]
    return {'entropy': float(-np.dot(np.exp(lp), lp)), 'margin': float(top[-1]-top[-2])}


def verify_record_files(folder, record):
    folder = Path(folder).resolve()
    for name, checksum in record['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or sha(path) != checksum:
            raise ValueError('Missing, changed, or unsafe saved artifact: '+name)


def verify_execution(root):
    root = Path(root)
    plan = json.loads((root/'execution_plan.json').read_text())
    assert plan['preparation_sha256'] == sha(root/'plan.json')
    for name, checksum in plan['files'].items():
        assert sha(name) == checksum, name
    return plan
