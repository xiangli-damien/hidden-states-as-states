"""Supervised Bayesian readouts of frozen, layer-local change clusters.

The prediction target is failure (1), success (0). Cluster IDs have independent
vocabularies at each layer; neither matching nor correctness enters clustering.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import rankdata
from sklearn.naive_bayes import MultinomialNB

from hss.experiments.evaluate import CountNB


class LocalMarkovBayes:
    """Class-conditional, depth-dependent first-order Markov probabilities."""

    def __init__(self, cardinalities, alpha=1.0):
        self.cardinalities = list(cardinalities)
        self.alpha = alpha

    def fit(self, states, y):
        states = np.asarray(states, dtype=int)
        y = np.asarray(y, dtype=int)
        if self.alpha <= 0 or set(np.unique(y)) != {0, 1}:
            raise ValueError("Positive smoothing and both classes are required")
        if states.shape[1] != len(self.cardinalities):
            raise ValueError("One cardinality is required per layer")
        for j, k in enumerate(self.cardinalities):
            if np.any((states[:, j] < 0) | (states[:, j] >= k)):
                raise ValueError("Cluster ID outside layer-local vocabulary")
        self.log_prior = np.log(np.bincount(y, minlength=2) / len(y))
        initial = np.full((2, self.cardinalities[0]), self.alpha, dtype=float)
        np.add.at(initial, (y, states[:, 0]), 1)
        self.initial = np.log(initial / initial.sum(1, keepdims=True))
        self.transitions = []
        for j, (a, b) in enumerate(zip(self.cardinalities, self.cardinalities[1:])):
            counts = np.full((2, a, b), self.alpha, dtype=float)
            np.add.at(counts, (y, states[:, j], states[:, j + 1]), 1)
            self.transitions.append(np.log(counts / counts.sum(2, keepdims=True)))
        return self

    def predict_log_proba(self, states):
        states = np.asarray(states, dtype=int)
        scores = self.log_prior + self.initial[:, states[:, 0]].T
        for j, table in enumerate(self.transitions):
            scores = scores + table[:, states[:, j], states[:, j + 1]].T
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def predict_proba(self, states):
        return np.exp(self.predict_log_proba(states))


def fit_bayes(kind, x, y, cardinalities, alpha=1.0):
    if kind == "categorical":
        model = CountNB([np.arange(k) for k in cardinalities], alpha=alpha)
    elif kind == "markov":
        model = LocalMarkovBayes(cardinalities, alpha=alpha)
    elif kind == "multinomial":
        model = MultinomialNB(alpha=alpha, fit_prior=True)
    else:
        raise ValueError(kind)
    return model.fit(x, y)


def log_odds(model, x):
    logp = model.predict_log_proba(x)
    return logp[:, 1] - logp[:, 0]


def pair_counts(assignments, owners, n_questions, k):
    """Counts preserve question grouping; scattered pairs are not a sequence."""
    counts = np.zeros((n_questions, k), dtype=np.int64)
    np.add.at(counts, (owners, assignments), 1)
    return counts


def far_threshold(success_scores, target=0.1):
    """Strict > alert; empirical validation FAR does not exceed target."""
    scores = np.sort(np.asarray(success_scores, float))
    if not len(scores) or not np.isfinite(scores).all() or not 0 <= target < 1:
        raise ValueError("Finite success scores and 0 <= FAR < 1 required")
    allowed = int(np.floor(len(scores) * target))
    return float(scores[len(scores) - allowed - 1])


def bootstrap_auc(y, scores, indices):
    """Question bootstrap; average ranks handle tied categorical scores."""
    values = []
    for ii in np.array_split(indices, max(1, int(np.ceil(len(indices) / 100)))):
        yy = y[ii]
        n1 = yy.sum(1)
        n0 = yy.shape[1] - n1
        ranks = rankdata(scores[ii], axis=1)
        values.append(((ranks * yy).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0))
    return np.concatenate(values)


class MonotonePlatt:
    """Validation-only calibration cannot reverse the prespecified score sign."""

    def fit(self, score, y):
        self.center = float(np.mean(score))
        self.scale = max(float(np.std(score)), 1e-8)
        z = (score - self.center) / self.scale
        def objective(ab):
            logits = ab[0] * z + ab[1]
            loss = np.mean(np.logaddexp(0, logits) - y * logits)
            error = expit(logits) - y
            return loss + 1e-4 * ab[0] ** 2, np.array([
                np.mean(error * z) + 2e-4 * ab[0], np.mean(error)])
        opt = minimize(objective, [1., 0.], jac=True, method="L-BFGS-B",
                       bounds=[(0, None), (None, None)])
        if not opt.success:
            raise RuntimeError(f"Calibration failed: {opt.message}")
        self.slope, self.intercept = map(float, opt.x)
        return self

    def predict_proba(self, score):
        return expit(self.slope * (score - self.center) / self.scale + self.intercept)
