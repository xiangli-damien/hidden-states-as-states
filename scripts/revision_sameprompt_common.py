"""Outcome-independent question selection for a bounded same-prompt study."""
import hashlib


def ordering_key(seed, namespace, value):
    return hashlib.sha256(f'{seed}|{namespace}|{value}'.encode()).hexdigest()


def select_questions(records, counts, seed):
    if set(counts) != {'train', 'tuning', 'calibration', 'test'}:
        raise ValueError('All four question roles are required')
    if any(not isinstance(n, int) or n <= 0 for n in counts.values()):
        raise ValueError('Role counts must be positive integers')
    ids = [r['sample_id'] for r in records]
    groups = [r['question_group'] for r in records]
    if len(ids) != len(set(ids)) or len(groups) != len(set(groups)):
        raise ValueError('Question identity must be unique before splitting')
    if any(r['split'] not in {'train', 'validation', 'test'} for r in records):
        raise ValueError('Unknown inherited question split')
    result = []
    for source_split, roles in [('train', ['train']), ('validation', ['tuning', 'calibration']), ('test', ['test'])]:
        candidates = sorted((r for r in records if r['split'] == source_split),
            key=lambda r: ordering_key(seed, source_split, r['question_group']))
        if len(candidates) < sum(counts[role] for role in roles):
            raise ValueError(f'Insufficient questions in {source_split}')
        offset = 0
        for role in roles:
            for r in candidates[offset:offset + counts[role]]:
                # Do not carry historical outcomes or answers into the selection.
                result.append({'sample_id': r['sample_id'], 'question_group': r['question_group'],
                               'source_split': source_split, 'role': role})
            offset += counts[role]
    return result


def trajectory_seed(seed, question_group, repetition):
    if repetition < 0:
        raise ValueError('Negative repetition')
    return int(ordering_key(seed, f'generation_{repetition}', question_group)[:8], 16)
