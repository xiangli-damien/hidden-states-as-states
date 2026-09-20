import numpy as np
import pytest
pytest.importorskip("torch")
from hss.cluster.mfa import fit_mfa
from hss.cluster.mfa_fast import BatchedMFAEM


@pytest.mark.parametrize("rank", [0, 3])
def test_batched_em_matches_reference_and_dense_scores(rank):
    rng = np.random.default_rng(92)
    X = rng.normal(size=(110, 13)) * np.linspace(.2, 2, 13)
    X[:55] += 4
    initial = fit_mfa(X, 2, rank=rank, n_init=1, max_iter=3, tol=0, init_method="svd")
    reference = fit_mfa(X, 2, rank=rank, n_init=1, max_iter=9, tol=0, initial_model=initial)
    engine = BatchedMFAEM(X, device="cpu", chunk_size=37, component_batch=1)
    model, audit = engine.fit(initial, max_steps=6, tol=0)
    np.testing.assert_allclose(model.score_samples(X), reference.score_samples(X), atol=1e-8)
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))
    ll, _, prob = engine.evaluate(engine.parameters(model), probabilities=True)
    np.testing.assert_allclose(prob, model.predict_proba(X), atol=1e-9)
    assert abs(ll-model.score(X)) < 1e-9
    assert np.min(np.diff(model.history_)) >= -1e-9


def test_squarem_is_monotone_and_checkpoint_matches_parameters():
    rng = np.random.default_rng(71)
    X = rng.normal(size=(170, 3)) @ rng.normal(size=(3, 15)) + .2*rng.normal(size=(170,15))
    initial = fit_mfa(X, 2, rank=3, n_init=1, max_iter=1, tol=0, init_method="svd")
    engine = BatchedMFAEM(X, device="cpu", chunk_size=60)
    saved = []
    model, audit = engine.fit(initial, max_steps=40, accelerator="squarem",
                              checkpoint=lambda m,a:saved.append(m))
    assert np.min(np.diff(model.history_)) >= -1e-8
    assert abs(model.score(X)-model.history_[-1]) < 1e-8
    assert np.all(model.noise_ >= model.reg_covar)
    assert saved[-1].converged_ == model.converged_
    assert audit["squarem_accepted"] > 0


def test_deadline_does_not_claim_convergence():
    import time
    X = np.random.default_rng(4).normal(size=(40, 7))
    initial = fit_mfa(X, 2, rank=2, n_init=1, max_iter=1)
    model, audit = BatchedMFAEM(X,device="cpu").fit(initial,deadline=time.time()-1)
    assert not model.converged_ and audit["stopped_by_deadline"]
