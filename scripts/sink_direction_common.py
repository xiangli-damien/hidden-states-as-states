"""Preregistered endpoints and policy for an exploratory response-mean direction.

The primary endpoint is complete, nonempty boxed-answer compliance. It is NOT
semantic correctness or an oracle for all possible unboxed final answers.
"""
import hashlib
import re
import numpy as np


def boxed_answer(text):
    """Last balanced nonempty boxed content; never falls back to a number."""
    answers = []
    for match in re.finditer(r'\\boxed\s*\{', text):
        start = match.end(); depth = 1
        for i in range(start, len(text)):
            # Escaped literal braces are not grouping braces.
            if i and text[i-1] == '\\':
                continue
            depth += int(text[i] == '{') - int(text[i] == '}')
            if depth == 0:
                value = text[start:i].strip()
                if value:
                    answers.append(value)
                break
    return answers[-1] if answers else None


def composition(text, finish_reason):
    words = text.split()
    grams = [tuple(words[i:i+12]) for i in range(max(0, len(words)-11))]
    repeated = 1 - len(set(grams))/len(grams) if grams else 0.
    return {'empty': not bool(text.strip()), 'complete_boxed': boxed_answer(text) is not None,
            'truncated': finish_reason == 'length', 'repeated_12gram_fraction': repeated,
            'deferral_phrase': bool(re.search(
                r'beyond the scope|cannot (?:be |provide|determine)|unable to|'
                r'without (?:further|additional)|not (?:detailed|provided) here', text, re.I))}


def ordered(ids, salt):
    return sorted(ids, key=lambda sid: hashlib.sha256((salt+sid).encode()).hexdigest())


def select_alpha(records, alphas):
    """Dev-only boxed compliance; tie -> smaller perturbation. No correctness use."""
    if any(r['split'] != 'dev' for r in records):
        raise ValueError('Policy selection must only see development generations')
    summary = []
    expected = {r['sample_id'] for r in records if r['condition'] == 'zero'}
    if not expected:
        raise ValueError('Missing dev baseline')
    for a in alphas:
        rows = [r for r in records if r['condition'] == f'hss_{a:g}']
        if len(rows) != len(expected) or {r['sample_id'] for r in rows} != expected:
            raise ValueError('Incomplete or duplicate dev conditions')
        summary.append({'alpha': a, 'boxed_count': sum(r['complete_boxed'] for r in rows),
                        'questions': len(rows)})
    selected = sorted(summary, key=lambda r: (-r['boxed_count'], r['alpha']))[0]
    return {'alpha': selected['alpha'], 'dev': summary,
            'criterion': 'Max dev complete-boxed count; ties prefer smallest alpha',
            'test_gate': 'Run held-out comparison even if dev shows no gain; do not search more'}


def paired_binary(a, b, seed=20260925, bootstrap=10000):
    """a minus b, question-level paired bootstrap and exact one-sided McNemar."""
    from math import comb
    a, b = np.asarray(a, int), np.asarray(b, int)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError('Expected paired nonempty vectors')
    d = a-b; wins = int((d > 0).sum()); losses = int((d < 0).sum())
    n = wins+losses
    p = sum(comb(n, k) for k in range(wins, n+1))/2**n if n else 1.
    rng = np.random.default_rng(seed)
    values = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(bootstrap)]
    return {'n': len(d), 'difference': float(d.mean()), 'ci95': np.quantile(values, [.025,.975]).tolist(),
            'wins': wins, 'losses': losses, 'one_sided_exact_p': p}


class GeneratedTokenShift:
    """Batch-one cached decoding. Skip prompt prefill; shift every forwarded new token.

    The first output token is therefore unmodified. The final sampled token/EOS
    is not forwarded by generate(). Prompt states are never overwritten.
    """
    def __init__(self, prompt_length, vector):
        import torch
        self.prompt_length = prompt_length
        self.vector = vector.float()
        if self.vector.ndim != 1 or not torch.isfinite(self.vector).all():
            raise ValueError('Finite one-dimensional shift required')
        self.nonzero = bool(torch.any(self.vector != 0))
        self.calls = 0; self.steps = []
        self.before_sum = torch.zeros_like(self.vector)
        self.after_sum = torch.zeros_like(self.vector)

    def __call__(self, module, args, out):
        import torch
        h = out[0] if isinstance(out, tuple) else out
        expected = self.prompt_length if self.calls == 0 else 1
        if h.shape != (1, expected, len(self.vector)):
            raise ValueError('Unexpected cached decoding shape')
        self.calls += 1
        if self.calls == 1:
            return None
        before = h[0,0].float()
        after = (before+self.vector).to(h.dtype) if self.nonzero else h[0,0]
        actual = after.float()-before
        self.before_sum += before
        self.after_sum += after.float()
        self.steps.append(torch.stack([before.norm(), actual.norm(), (actual-self.vector).norm(),
                                       torch.dot(before, self.vector), torch.dot(actual, self.vector)]))
        if not self.nonzero:
            return None
        replaced = h.clone(); replaced[0,0] = after
        return (replaced, *out[1:]) if isinstance(out, tuple) else replaced

    def arrays(self):
        import torch
        steps = torch.stack(self.steps) if self.steps else self.vector.new_empty((0,5))
        n = max(1, len(self.steps))
        return {'step_geometry': steps.cpu().numpy(),
                'mean_before': (self.before_sum/n).cpu().numpy(),
                'mean_after': (self.after_sum/n).cpu().numpy()}
