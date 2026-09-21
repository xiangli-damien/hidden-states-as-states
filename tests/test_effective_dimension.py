import numpy as np
import pytest

from hss.analysis.effective_dimension import (
    covariance_dimensions, empirical_dimensions, matched_size_dimensions, weighted_quantile,
)


def test_low_rank_trace_identity_matches_dense_covariance():
    rng = np.random.default_rng(7)
    w, psi = rng.normal(size=(17, 4)), rng.uniform(.1, 3, 17)
    actual = covariance_dimensions(w, psi)
    covariance = w @ w.T + np.diag(psi)
    eigen = np.linalg.eigvalsh(covariance)
    np.testing.assert_allclose(actual['model_pr'], eigen.sum() ** 2 / (eigen @ eigen), rtol=1e-12)
    np.testing.assert_allclose(actual['total_square_trace'], np.trace(covariance @ covariance), rtol=1e-12)


def test_factor_rotation_and_global_scale_do_not_change_dimensions():
    rng = np.random.default_rng(8)
    w, psi = rng.normal(size=(19, 5)), rng.uniform(.1, 2, 19)
    rotation = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    a = covariance_dimensions(w, psi)
    for b in [covariance_dimensions(w @ rotation, psi), covariance_dimensions(w * 7, psi * 49)]:
        for key in ['factor_pr', 'model_pr', 'noise_pr', 'factor_fraction']:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12)


def test_factor_rank_is_not_total_dimension():
    w = np.zeros((20, 2)); w[0, 0] = 1; w[1, 1] = 1
    a = covariance_dimensions(w, np.ones(20))
    assert a['factor_pr'] == 2 and a['model_pr'] > 2
    w[0, 0] = 100
    assert covariance_dimensions(w, np.ones(20))['factor_pr'] < 1.01
    assert covariance_dimensions(np.zeros((20, 2)), np.ones(20))['model_pr'] == 20


def test_empirical_dimension_is_centered_and_bounded_by_sample_rank():
    rng = np.random.default_rng(9); x = rng.normal(size=(9, 20))
    a = empirical_dimensions(x)
    assert 1 <= a['empirical_pr'] <= 8 and a['empirical_rank_ceiling'] == 8
    np.testing.assert_allclose(a['empirical_pr'], empirical_dimensions(x * 7 + 200)['empirical_pr'])
    eigen = np.linalg.eigvalsh(np.cov(x, rowvar=False))
    np.testing.assert_allclose(a['empirical_pr'], eigen.sum() ** 2 / (eigen @ eigen))
    assert empirical_dimensions(np.ones((5, 3)))['empirical_pr'] == 0


def test_matched_sample_size_is_reproducible_and_reports_insufficient_size():
    x = np.random.default_rng(9).normal(size=(50, 7))
    a = matched_size_dimensions(x, 20, 5, 4)
    np.testing.assert_array_equal(a, matched_size_dimensions(x, 20, 5, 4))
    assert len(matched_size_dimensions(x, 100, 5, 4)) == 0
    np.testing.assert_allclose(matched_size_dimensions(x, 50, 3, 4), empirical_dimensions(x)['empirical_pr'])


def test_cluster_equal_and_question_weighted_medians_differ():
    x = np.array([1., 3., 10.])
    assert weighted_quantile(x, np.ones(3), [.5])[0] == 3
    assert weighted_quantile(x, np.array([1, 1, 100]), [.5])[0] == 10
    with pytest.raises(ValueError): weighted_quantile(x, np.zeros(3), [.5])


def test_layer_summary_preserves_view_and_weighting():
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from analyze_mfa_dimensions import summarize
    data = pd.DataFrame(dict(unit=['L01'] * 3, position=[1] * 3, view=['post'] * 3,
        layer=[1] * 3, k=[3] * 3, rank=[16] * 3, n=[10, 10, 100],
        factor_pr=[1., 3., 10.], factor_fraction=[.8] * 3))
    row = summarize(data, ['factor_pr']).iloc[0]
    assert row['view'] == 'post' and row.n_questions == 120
    assert row.factor_pr_p50 == 3 and row.factor_pr_weighted_median == 10
