"""Fixed, finite notation-continuation experiment; no model or target fitting.

The variable is the next subscript, not an arithmetic reasoning skill. Inspired
by MATH validation math_2526, all cases below are new controlled synthetic inputs.
"""
import hashlib
import re

import numpy as np


def case_pairs():
    pool = []
    for i in range(13):
        for start in range(1, 6):
            left = {'symbol': chr(97+i), 'start': start}
            right = {'symbol': chr(110+i), 'start': start % 5 + 1}
            key = f'notation-counter-v1/{i}/{start}'
            digest = hashlib.sha256(key.encode()).hexdigest()
            if int(digest[-1], 16) % 2:
                left, right = right, left
            pool.append({'id': 'index_'+digest[:16], 'order': digest,
                         'recipient': left, 'donor': right})
    selected = sorted(pool, key=lambda r: r['order'])[:64]
    result = []
    for i, row in enumerate(selected):
        result.append({k:v for k,v in row.items() if k!='order'} |
                      {'split': 'validation' if i<16 else 'test'})
    clean = [(r[side]['symbol'], r[side]['start']) for r in result for side in ['recipient','donor']]
    assert len(clean) == len(set(clean)) == 128
    return result


def prompt(item):
    return (f'Write exactly seven consecutive indexed terms, starting with {item["symbol"]}_{item["start"]}. '
            'Keep the same lowercase letter in every term and increase the integer subscript by one each time. '
            'Use plain text letter_1 notation, with no dollar signs, braces, or LaTeX. '
            'Separate terms with commas. Output only the seven terms, with no explanation or final period.')


def assistant_prefix(item):
    return ', '.join(f'{item["symbol"]}_{i}' for i in range(item['start'],item['start']+4)) + f', {item["symbol"]}_'


def parse_terms(text):
    if not re.fullmatch(r'\s*[a-z]_\d+(?:\s*,\s*[a-z]_\d+)*\s*', text):
        return None
    return [(s,int(n)) for s,n in re.findall(r'([a-z])_(\d+)',text)]


def ordinary_correct(text, item):
    return parse_terms(text) == [(item['symbol'],i) for i in range(item['start'],item['start']+7)]


def score_suffix(suffix, recipient, donor):
    terms = parse_terms(assistant_prefix(recipient)+suffix)
    valid = terms is not None and len(terms)==7
    original_prefix = [(recipient['symbol'],i) for i in range(recipient['start'],recipient['start']+4)]
    valid = bool(valid and terms[:4]==original_prefix)
    future = terms[4:] if valid else []
    target = bool(valid and [n for _,n in future] == list(range(donor['start']+4, donor['start']+7)))
    # The fifth symbol is in the supplied prefix. Only the sixth and seventh
    # symbols are newly generated; these two form the non-target preservation test.
    preserved = bool(valid and all(s==recipient['symbol'] for s,_ in future[1:]))
    return {'format_valid':valid, 'target_counter_sequence':target,
            'new_symbols_preserved':preserved, 'joint_success':target and preserved,
            'ordinary_recipient_correct':ordinary_correct(assistant_prefix(recipient)+suffix,recipient)}


def transfer(recipient, donor, centers, local_basis, shared_basis, method):
    """Recipient-anchored projected donor difference, not a donor-coordinate mix.

    z = h_R + P_{s_R}(h_D-h_R). Region assignment is from h_R only.
    Shared8 uses the same operation with one frozen shared basis.
    """
    x, d = np.asarray(recipient,dtype=np.float64),np.asarray(donor,dtype=np.float64)
    assert x.shape == d.shape and x.ndim==2 and np.isfinite(x).all() and np.isfinite(d).all()
    if method=='identity':return x.copy()
    if method=='full_donor':return d.copy()
    delta=d-x
    if method=='shared8':
        w=np.asarray(shared_basis,dtype=np.float64)[:8]
        return x+(delta@w.T)@w
    if method!='local8':raise ValueError(method)
    c=np.asarray(centers,dtype=np.float64)
    labels=((x*x).sum(1)[:,None]+(c*c).sum(1)[None,:]-2*x@c.T).argmin(1)
    out=x.copy()
    for k in np.unique(labels):
        select=labels==k;w=np.asarray(local_basis,dtype=np.float64)[k,:8]
        out[select]+=(delta[select]@w.T)@w
    return out


def capability_gate(records):
    """Validation only. Fail closed on missing cases, failure or low coverage."""
    wanted={r['id'] for r in case_pairs() if r['split']=='validation'}
    assert len(records)==len(wanted) and {r['case_id'] for r in records}==wanted
    free=sum(r[s]['free_correct'] for r in records for s in ['recipient','donor'])
    prefixed=sum(r[s]['prefixed_correct'] for r in records for s in ['recipient','donor'])
    eligible=sum(r['aligned'] and all(r[s]['free_correct'] and r[s]['prefixed_correct'] for s in ['recipient','donor']) for r in records)
    return {'pairs':16,'clean_contexts':32,'free_correct':int(free),'prefixed_correct':int(prefixed),
            'eligible_pairs':int(eligible),'passed':bool(free>=29 and prefixed>=29 and eligible>=12)}
