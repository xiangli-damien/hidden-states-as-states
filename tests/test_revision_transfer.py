from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from sklearn.mixture import GaussianMixture
from revision_transfer import density,nb_fit,nb_predict,fit_map


def test_density_is_real_diagonal_mixture_likelihood():
    rng=np.random.default_rng(1);x=rng.normal(size=(80,4));x[:40]+=4
    g=GaussianMixture(2,covariance_type='diag',random_state=1).fit(x)
    model={'centers':g.means_,'variances':g.covariances_,'weights':g.weights_}
    np.testing.assert_allclose(density(x,model),g.score_samples(x),rtol=1e-10,atol=1e-10)


def test_target_calibration_changes_readout_without_changing_assignments():
    codes=np.array([0,0,0,1,1,1]);y=np.array([0,0,1,1,1,0])
    a=nb_fit(codes,y,3);b=nb_fit(codes,1-y,3)
    np.testing.assert_allclose(nb_predict(a,np.array([0,1,2])),1-nb_predict(b,np.array([0,1,2])))
    assert np.isfinite(nb_predict(a,np.array([2]))).all()


def test_target_map_selection_has_no_label_argument_and_saves_all_candidates(tmp_path):
    rng=np.random.default_rng(4);x=np.r_[rng.normal(size=(60,3)),rng.normal(size=(60,3))+5]
    model=fit_map(x,tmp_path,[1,2],[42])
    assert set(model)=={'centers','variances','weights','train_mean'}
    assert (tmp_path/'k1_seed42.npz').exists() and (tmp_path/'k2_seed42.npz').exists()
    assert (tmp_path/'selection.json').exists()
