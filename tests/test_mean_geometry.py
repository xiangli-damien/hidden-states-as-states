import numpy as np
import pytest
from hss.analysis.mean_geometry import trajectory_scores, within_length_auc


def test_paper_equations_hand_trajectory():
    # Two unit vectors at right angles: NDR=1, M/ZM=1, A/ZA=1.
    x = np.array([[[1.,0.],[0.,1.]]])
    s = trajectory_scores(x)
    assert s['ndr'][0] == pytest.approx(1.)
    assert s['coe_r'][0] == pytest.approx(0.)
    assert s['coe_c'][0] == pytest.approx(1.)
    assert s['last_update_cosine'][0] == pytest.approx(-1/np.sqrt(2))


def test_metric_rotation_and_uniform_scale_invariance():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(8,6,10))
    rotation, _ = np.linalg.qr(rng.normal(size=(10,10)))
    a, b = trajectory_scores(x), trajectory_scores(7*x@rotation)
    for name in ['ndr','coe_r','coe_c','last_to_previous_norm_ratio','last_update_cosine']:
        np.testing.assert_allclose(a[name], b[name], atol=1e-12)


def test_final_replacement_changes_denominator_and_layer_mean():
    x = np.array([[[1.,0.],[1.,1.],[0.,4.]]])
    a = trajectory_scores(x)
    x[:,-1] /= 2
    b = trajectory_scores(x)
    assert a['ndr'][0] == pytest.approx((1+np.sqrt(2)+4)/12)
    assert b['ndr'][0] == pytest.approx((1+np.sqrt(2)+2)/6)
    with pytest.raises(ValueError, match='Degenerate'):
        trajectory_scores(np.ones((1,3,4)))


def test_length_conditional_auc_does_not_compare_across_bins():
    # Pooled AUC differs, but ordering is perfect inside each stratum.
    y = np.array([0,1,0,1])
    score = np.array([0.,1.,100.,101.])
    assert within_length_auc(y, score, np.array([0,0,1,1])) == 1.

