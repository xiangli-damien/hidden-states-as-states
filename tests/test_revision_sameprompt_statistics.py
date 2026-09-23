from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from revision_sameprompt_statistics import auc, mean_interval, within_question, far_threshold, alarms


def test_static_prompt_score_has_no_within_question_discrimination():
    result = within_question(['easy']*4+['hard']*4, [0, 0, 0, 1, 0, 1, 1, 1],
        [.1]*4+[.9]*4, [.5]*8)
    assert result['paired_delta']['estimate'] == 0
    assert result['macro_first']['estimate'] == .5
    assert auc([0, 1, 0, 1], [0., 0., 1., 1.]) == .5


def test_question_macro_does_not_weight_by_number_of_trajectory_pairs():
    q = ['a']*2+['b']*6; y = [0, 1, 0, 0, 0, 1, 1, 1]
    first = [0, 1, 1, 1, 1, 0, 0, 0]; second = [.5]*8
    result = within_question(q, y, first, second)
    assert result['macro_first']['estimate'] == .5
    assert result['paired_delta']['n_questions'] == 2
    order = np.array([7, 0, 6, 1, 5, 2, 4, 3])
    reordered = within_question(np.array(q)[order], np.array(y)[order], np.array(first)[order], np.array(second)[order])
    assert result == reordered


def test_single_class_is_not_fabricated_as_chance_or_zero():
    assert auc([1, 1], [.1, .9]) is None
    assert mean_interval([])['estimate'] is None
    assert within_question(['q', 'q'], [0, 0], [.1, .2], [.1, .9])['paired_delta']['n_questions'] == 0


def test_response_maximum_and_strict_ties_control_calibration_far():
    scores = np.zeros((10, 2)); scores[0, 0] = .9; scores[1, 1] = .8
    maximum = scores.max(1); threshold = far_threshold(maximum, .1)
    assert threshold == .8 and np.sum(maximum > threshold) == 1
    assert np.sum(np.ones(10) > far_threshold(np.ones(10), .1)) == 0
    assert far_threshold([]) is None
    assert far_threshold([-np.inf, -np.inf]) == -np.inf


def test_first_alarm_and_saved_tokens_only_use_existing_boundaries():
    scores = np.array([[.7, .9], [.1, .9], [-np.inf, -np.inf]])
    r = alarms(scores, [16, 64], [100, 80, 10], .5)
    np.testing.assert_array_equal(r['first_boundary'], [16, 64, -1])
    np.testing.assert_array_equal(r['potential_saved_tokens'], [84, 16, 0])
    with pytest.raises(ValueError, match='termination'):
        alarms([[.9, .8]], [16, 64], [40], .5)
