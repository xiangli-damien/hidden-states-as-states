from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from fit_revision_sameprompt import candidates, design


def test_candidate_selection_uses_tuning_and_saves_reusable_weights():
    rng = np.random.default_rng(9); x = rng.normal(size=(60, 4)); y = (x[:, 0]+.3*x[:, 1] > 0).astype(int)
    selected, trials, models = candidates(x[:30], y[:30], x[30:40], y[30:40], x, [.001, .01, .1], 200)
    best = min((r for r in trials if r['converged']), key=lambda r: (r['tuning_log_loss'], r['C']))
    assert selected['C'] == best['C']
    for model in models.values():
        np.testing.assert_allclose(expit(x@model['coef'][0]+model['intercept'][0]), model['probability'], atol=1e-12)
    # Replacing held-out feature rows cannot affect fitting or C selection.
    changed = x.copy(); changed[40:] = rng.normal(size=(20, 4))*100
    again, _, fitted = candidates(x[:30], y[:30], x[30:40], y[30:40], changed, [.001, .01, .1], 200)
    assert again['C'] == selected['C']
    np.testing.assert_array_equal(fitted[again['C']]['coef'], models[selected['C']]['coef'])


def test_baseline_and_duplicate_control_have_explicit_feature_blocks():
    arrays = {'prompt': np.full((2, 3), 1), 'current': np.full((2, 3), 2),
              'controls': np.full((2, 2), 3), 'mean16': np.full((2, 3), 4), 'allmean': np.full((2, 3), 5),
              'failure': np.ones(2), 'observed_length': np.ones(2)*999}
    base = design(arrays, 'baseline'); mean = design(arrays, 'mean16'); duplicate = design(arrays, 'duplicate_current')
    assert base.shape == (2, 8) and mean.shape == duplicate.shape == (2, 11)
    np.testing.assert_array_equal(mean[:, :8], base)
    np.testing.assert_array_equal(duplicate[:, 8:], arrays['current'])
    assert not np.any(base == 999)
