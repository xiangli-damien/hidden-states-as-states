import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('sameprompt', Path(__file__).parents[1]/'scripts/revision_sameprompt_common.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def records():
    return [{'sample_id': f'{split}_{i}', 'question_group': f'{split}_group_{i}', 'split': split, 'label': i % 2}
            for split in ['train', 'validation', 'test'] for i in range(12)]


def test_selection_is_order_and_outcome_independent_with_disjoint_roles():
    data = records(); counts = {'train': 5, 'tuning': 3, 'calibration': 4, 'test': 6}
    selected = module.select_questions(data, counts, 42)
    changed = [{**r, 'label': 1-r['label']} for r in reversed(data)]
    assert selected == module.select_questions(changed, counts, 42)
    assert len({r['question_group'] for r in selected}) == 18
    assert {role: sum(r['role'] == role for r in selected) for role in counts} == counts
    assert all(r['source_split'] == ('validation' if r['role'] in ['tuning', 'calibration'] else r['role']) for r in selected)
    assert all('label' not in r for r in selected)


def test_colliding_groups_and_insufficient_pools_are_rejected():
    data = records(); counts = {'train': 5, 'tuning': 3, 'calibration': 4, 'test': 6}
    data[12]['question_group'] = data[0]['question_group']
    with pytest.raises(ValueError, match='identity'):
        module.select_questions(data, counts, 42)
    with pytest.raises(ValueError, match='Insufficient'):
        module.select_questions(records(), counts | {'calibration': 10}, 42)


def test_generation_seeds_are_order_independent_and_separate_repeats():
    values = {(r['question_group'], i): module.trajectory_seed(42, r['question_group'], i)
              for r in records() for i in range(4)}
    assert len(set(values.values())) == len(values)
    assert all(v == module.trajectory_seed(42, group, i) for (group, i), v in reversed(list(values.items())))
