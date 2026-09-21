import numpy as np
from sklearn.metrics import roc_auc_score

from hss.analysis.change_bayes import (
    LocalMarkovBayes, MonotonePlatt, bootstrap_auc, far_threshold,
    fit_bayes, log_odds, pair_counts,
)


def test_transition_evidence_can_differ_when_node_marginals_are_identical():
    x = np.tile([[0, 0], [1, 1], [0, 1], [1, 0]], (20, 1))
    y = np.tile([0, 0, 1, 1], 20)
    nb = fit_bayes("categorical", x, y, [2, 2])
    markov = fit_bayes("markov", x, y, [2, 2])
    assert roc_auc_score(y, log_odds(nb, x)) == .5
    assert roc_auc_score(y, log_odds(markov, x)) == 1.
    # Independent renumbering at each layer must not change predictions.
    flipped = x.copy(); flipped[:, 1] = 1 - flipped[:, 1]
    other = fit_bayes("markov", flipped, y, [2, 2])
    np.testing.assert_allclose(log_odds(markov, x), log_odds(other, flipped))


def test_unseen_transitions_and_unequal_vocabularies_remain_finite():
    x = np.array([[0, 0, 0], [0, 1, 0], [1, 2, 0], [1, 1, 0]])
    model = LocalMarkovBayes([2, 3, 2]).fit(x, np.array([0, 0, 1, 1]))
    score = model.predict_log_proba(np.array([[0, 2, 1], [1, 0, 1]]))
    assert np.isfinite(score).all()
    np.testing.assert_allclose(np.exp(score).sum(1), 1)
    assert model.transitions[0].shape == (2, 2, 3)


def test_count_nb_retains_extreme_log_odds():
    x = np.vstack([np.zeros((100, 400), int), np.ones((100, 400), int)])
    model = fit_bayes("categorical", x, np.repeat([0, 1], 100), [2] * 400)
    score = log_odds(model, x)
    assert np.isfinite(score).all() and score[-1] > 1000
    np.testing.assert_allclose(model.predict_proba(x).sum(1), 1)


def test_pair_count_grouping_and_far_with_ties():
    counts = pair_counts(np.array([0, 1, 1, 2]), np.array([0, 0, 1, 1]), 2, 3)
    np.testing.assert_array_equal(counts, [[1, 1, 0], [0, 1, 1]])
    s = np.array([0.] * 8 + [1.] * 2)
    t = far_threshold(s, .1)
    assert np.mean(s > t) <= .1


def test_bootstrap_matches_sklearn_and_calibration_cannot_flip_direction():
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0., 0., 1., 1., 2., 3.])
    indices = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 1, 3, 5, 5]])
    np.testing.assert_allclose(bootstrap_auc(y, s, indices),
                               [roc_auc_score(y[i], s[i]) for i in indices])
    c = MonotonePlatt().fit(s, y)
    assert c.slope >= 0 and np.isfinite(c.predict_proba(s)).all()
