from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class NaiveBayesClassifier:
    alpha: float = 1.0
    n_classes_: int = 0
    class_log_prior_: Optional[np.ndarray] = field(default=None, repr=False)
    emission_log_prob_: Optional[List[np.ndarray]] = field(default=None, repr=False)
    n_layers_: int = 0
    n_states_: int = 0

    def fit(
        self, global_labels: np.ndarray, y: np.ndarray
    ) -> NaiveBayesClassifier:
        y = np.asarray(y, dtype=np.int32).ravel()
        n, L = global_labels.shape
        classes = np.unique(y)
        C = int(classes.max()) + 1
        self.n_classes_ = C
        self.n_layers_ = L
        S = int(global_labels.max()) + 1
        self.n_states_ = S
        class_counts = np.zeros(C, dtype=np.float64)
        for c in range(C):
            class_counts[c] = float((y == c).sum())
        self.class_log_prior_ = np.log(
            (class_counts + self.alpha) / (n + C * self.alpha)
        )
        self.emission_log_prob_ = []
        for li in range(L):
            count_table = np.full((C, S), self.alpha, dtype=np.float64)
            for c in range(C):
                mask = y == c
                col = global_labels[mask, li]
                col = col[col >= 0]
                if len(col) > 0:
                    bc = np.bincount(col, minlength=S).astype(np.float64)
                    count_table[c] += bc
            row_sums = count_table.sum(axis=1, keepdims=True)
            self.emission_log_prob_.append(np.log(count_table / row_sums))
        return self

    def predict_log_proba(self, global_labels: np.ndarray) -> np.ndarray:
        if self.class_log_prior_ is None or self.emission_log_prob_ is None:
            raise RuntimeError("Not fitted")
        n, L = global_labels.shape
        C = self.n_classes_
        log_prob = np.tile(self.class_log_prior_, (n, 1))
        for li in range(min(L, self.n_layers_)):
            col = global_labels[:, li]
            for c in range(C):
                valid = (col >= 0) & (col < self.n_states_)
                log_prob[valid, c] += self.emission_log_prob_[li][c, col[valid]]
        return log_prob

    def predict_proba(self, global_labels: np.ndarray) -> np.ndarray:
        log_prob = self.predict_log_proba(global_labels)
        log_prob -= log_prob.max(axis=1, keepdims=True)
        prob = np.exp(log_prob)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def predict(self, global_labels: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_log_proba(global_labels), axis=1).astype(np.int32)


@dataclass
class MarkovClassifier:
    alpha: float = 1.0
    n_classes_: int = 0
    class_log_prior_: Optional[np.ndarray] = field(default=None, repr=False)
    init_log_prob_: Optional[np.ndarray] = field(default=None, repr=False)
    trans_log_prob_: Optional[List[np.ndarray]] = field(default=None, repr=False)
    n_layers_: int = 0
    n_states_: int = 0

    def fit(
        self, global_labels: np.ndarray, y: np.ndarray
    ) -> MarkovClassifier:
        y = np.asarray(y, dtype=np.int32).ravel()
        n, L = global_labels.shape
        classes = np.unique(y)
        C = int(classes.max()) + 1
        self.n_classes_ = C
        self.n_layers_ = L
        S = int(global_labels.max()) + 1
        self.n_states_ = S
        class_counts = np.zeros(C, dtype=np.float64)
        for c in range(C):
            class_counts[c] = float((y == c).sum())
        self.class_log_prior_ = np.log(
            (class_counts + self.alpha) / (n + C * self.alpha)
        )
        init_counts = np.full((C, S), self.alpha, dtype=np.float64)
        for c in range(C):
            mask = y == c
            col = global_labels[mask, 0]
            col = col[col >= 0]
            if len(col) > 0:
                init_counts[c] += np.bincount(col, minlength=S).astype(np.float64)
        init_sums = init_counts.sum(axis=1, keepdims=True)
        self.init_log_prob_ = np.log(init_counts / init_sums)
        self.trans_log_prob_ = []
        for li in range(L - 1):
            trans_counts = np.full((C, S, S), self.alpha, dtype=np.float64)
            for c in range(C):
                mask = y == c
                s_from = global_labels[mask, li]
                s_to = global_labels[mask, li + 1]
                valid = (s_from >= 0) & (s_to >= 0) & (s_from < S) & (s_to < S)
                sf = s_from[valid]
                st = s_to[valid]
                np.add.at(trans_counts[c], (sf, st), 1.0)
            row_sums = trans_counts.sum(axis=2, keepdims=True)
            self.trans_log_prob_.append(np.log(trans_counts / row_sums))
        return self

    def predict_log_proba(self, global_labels: np.ndarray) -> np.ndarray:
        if (
            self.class_log_prior_ is None
            or self.init_log_prob_ is None
            or self.trans_log_prob_ is None
        ):
            raise RuntimeError("Not fitted")
        n, L = global_labels.shape
        C = self.n_classes_
        S = self.n_states_
        log_prob = np.tile(self.class_log_prior_, (n, 1))
        s0 = global_labels[:, 0]
        for c in range(C):
            valid = (s0 >= 0) & (s0 < S)
            log_prob[valid, c] += self.init_log_prob_[c, s0[valid]]
        for li in range(min(L - 1, len(self.trans_log_prob_))):
            sf = global_labels[:, li]
            st = global_labels[:, li + 1]
            valid = (sf >= 0) & (st >= 0) & (sf < S) & (st < S)
            for c in range(C):
                log_prob[valid, c] += self.trans_log_prob_[li][c, sf[valid], st[valid]]
        return log_prob

    def predict_proba(self, global_labels: np.ndarray) -> np.ndarray:
        log_prob = self.predict_log_proba(global_labels)
        log_prob -= log_prob.max(axis=1, keepdims=True)
        prob = np.exp(log_prob)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def predict(self, global_labels: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_log_proba(global_labels), axis=1).astype(np.int32)


def evaluate_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Binary classification metrics. y_score: probability of positive class (or score)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    y_pred = (y_score >= threshold).astype(np.int32)
    acc = float((y_pred == y_true).mean())
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")
    return {"accuracy": acc, "auroc": auroc}