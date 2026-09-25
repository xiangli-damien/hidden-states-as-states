"""Portable contracts for the reviewed sink follow-up; no torch/sklearn imports."""
import json
import re
from pathlib import Path
import numpy as np
from revision_common import sha, write_json
from hss_followup_tools import PARAMS


def sentence_boundaries(text, offsets, n_tokens):
    """Exact copy of hss.data.openact.sentence_boundaries; tested against that source.

    This is the current repository rule. The original paper's segmenter was not
    located, so we do not describe this as a verified reproduction of section 5.2.
    """
    valid = np.flatnonzero((offsets[:, 1] > offsets[:, 0]) & (offsets[:, 1] > 0))
    ends = offsets[valid, 1]
    if len(ends) and np.any(np.diff(ends) < 0):
        raise ValueError("Nonmonotonic token offsets cannot define causal sentence boundaries")
    positions = []
    for m in re.finditer(r"[.!?](?=\s|$)|[。！？]|\n+", text):
        pos = int(np.searchsorted(ends, m.end()))
        if pos < len(valid):
            positions.append(int(valid[pos]) + 1)
    return sorted(set(p for p in positions + [n_tokens] if 0 < p <= n_tokens))


def token_text(tok, ids):
    """Offsets without replacing the saved token sequence.

    Fast-tokenizer roundtrip must be exact. Fail rather than silently align a
    different segmentation. Terminal special tokens get empty offsets.
    """
    ids = list(map(int, ids))
    specials = set(tok.all_special_ids)
    core = len(ids)
    while core and ids[core-1] in specials:
        core -= 1
    if any(i in specials for i in ids[:core]):
        raise ValueError('Internal special token needs an explicit segmentation rule')
    text = tok.decode(ids[:core], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != ids[:core]:
        # Byte-BPE alternate segmentations are legitimate. Decode prefixes to
        # align the ORIGINAL tokens instead of retokenizing the forward pass.
        offsets = []
        previous = 0
        for i in range(core):
            prefix = tok.decode(ids[:i+1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            # Incomplete UTF-8 prefixes may end in the replacement character.
            end = len(prefix.rstrip('\ufffd'))
            if end < previous or not text.startswith(prefix.rstrip('\ufffd')):
                raise ValueError('Saved-token character offsets are not prefix consistent')
            offsets.append((previous, end)); previous = end
        if previous != len(text):
            raise ValueError('Incomplete saved-token text coverage')
        rule = 'saved_token_prefix_decode'
    else:
        offsets = list(encoded['offset_mapping']); rule = 'exact_fast_tokenizer_roundtrip'
    offsets.extend([(0, 0)] * (len(ids)-core))
    return text, np.asarray(offsets, dtype=np.int64), core, rule


def boxed_token_indices(text, offsets):
    """Last complete nonempty boxed CONTENT through its closing brace, inclusive."""
    span = None
    for match in re.finditer(r'\\boxed\s*\{', text):
        start = match.end(); depth = 1; i = start
        while i < len(text):
            if text[i] == '\\':
                i += 2; continue
            depth += int(text[i] == '{') - int(text[i] == '}')
            if depth == 0:
                if text[start:i].strip():
                    span = (start, i+1)
                break
            i += 1
    if span is None:
        return np.array([], dtype=np.int64)
    return np.flatnonzero((offsets[:, 0] < span[1]) & (offsets[:, 1] > span[0]))


def gmm_assign(x, means, covariances, weights):
    x = np.atleast_2d(x).astype(np.float64)
    c = np.asarray(means, dtype=np.float64)
    v = np.asarray(covariances, dtype=np.float64)
    if not np.isfinite(x).all() or (v <= 0).any():
        raise ValueError('Invalid GMM input/variance')
    # [N,K,D] avoids cancellation for nearly coincident vectors; N is small.
    scores = -.5 * (np.square(x[:, None]-c[None]) / v[None] + np.log(v)[None]).sum(-1)
    scores += np.log(weights)[None]
    return scores.argmax(-1)


def verify_plan(root):
    root = Path(root)
    plan = json.loads((root/'plan.json').read_text())
    assert plan['params'] == PARAMS, 'Frozen parameters changed'
    for path, digest in plan['files'].items():
        assert sha(path) == digest, f'Frozen input changed: {path}'
    return plan


def receipt(path, root, extra=None):
    write_json(Path(path).with_suffix('.receipt.json'), dict(
        sha256=sha(path), plan_sha256=sha(Path(root)/'plan.json'), **(extra or {})))


def verify_record(path, root):
    r = json.loads(Path(path).with_suffix('.receipt.json').read_text())
    assert sha(path) == r['sha256'] and r['plan_sha256'] == sha(Path(root)/'plan.json')
    return r
