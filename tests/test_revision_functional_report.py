"""Statistical contracts: pair by question and avoid weighting longer answers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from report_revision_functional import interval, paired_difference


def test_pairing_aligns_ids_and_resamples_differences():
    frame = pd.DataFrame({'sample_id': ['b', 'a', 'a', 'b'],
                          'method': ['A', 'A', 'B', 'B'], 'nll': [12., 2., 1., 11.]})
    result = paired_difference(frame, 'A', 'B', 'nll')
    assert result == {'estimate': 1., 'low': 1., 'high': 1., 'n': 2}


def test_pairing_refuses_missing_or_duplicate_questions():
    frame = pd.DataFrame({'sample_id': ['a', 'b', 'a'], 'method': ['A', 'A', 'B'], 'nll': [1., 2., 3.]})
    with pytest.raises(ValueError): paired_difference(frame, 'A', 'B', 'nll')
    frame.loc[1, 'sample_id'] = 'a'
    with pytest.raises(ValueError): paired_difference(frame, 'A', 'B', 'nll')


def test_question_means_are_not_token_pooled():
    # Ten tokens at NLL .1, 1000 tokens at NLL .9: question mean is .5.
    means = np.array([1., 900.])/np.array([10, 1000])
    assert interval(means)['estimate'] == .5
    assert not np.isclose(interval(means)['estimate'], 901/1010)
    with pytest.raises(ValueError): interval([np.nan])
