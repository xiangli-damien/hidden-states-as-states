from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import hss
import numpy as np
import pandas as pd
from hss.types import StateProvider
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import PredictionConfig
from .io_utils import ExperimentStore

_log = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    summary_df: pd.DataFrame
    fold_df: pd.DataFrame
    config: Dict[str, Any]


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float('nan')


def _eval_fold(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'auroc': _safe_auroc(y_true, y_score),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _run_hss_nb(gl_train: np.ndarray, y_train: np.ndarray, gl_test: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    model = hss.NaiveBayesClassifier(alpha=alpha).fit(gl_train, y_train)
    proba = model.predict_proba(gl_test)
    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0], model.predict(gl_test)


def _run_hss_markov(gl_train: np.ndarray, y_train: np.ndarray, gl_test: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    model = hss.MarkovClassifier(alpha=alpha).fit(gl_train, y_train)
    proba = model.predict_proba(gl_test)
    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0], model.predict(gl_test)


def _collect_states(provider: StateProvider, layer: int, indices: np.ndarray) -> np.ndarray:
    chunks = []
    for batch in provider.iter_batches(layer=layer, indices=indices, batch_size=4096):
        chunks.append(np.asarray(batch.states, dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, provider.state_dim()), dtype=np.float32)


def _get_continuous_features(provider: StateProvider, layers: List[int], indices: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'last_layer':
        return _collect_states(provider, layers[-1], indices)
    if mode == 'all_layers':
        parts = [_collect_states(provider, layer, indices) for layer in layers]
        return np.concatenate(parts, axis=1)
    raise ValueError(f'Unknown continuous feature mode: {mode}')


def _run_sklearn(method: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, cfg: PredictionConfig) -> tuple[np.ndarray, np.ndarray]:
    builders = {
        'LDA': lambda: LinearDiscriminantAnalysis(),
        'GaussianNB': lambda: GaussianNB(),
        'Logistic': lambda: LogisticRegression(max_iter=1000, solver='lbfgs'),
        'LinearSVM': lambda: LinearSVC(max_iter=2000, dual='auto'),
        'MLP': lambda: MLPClassifier(hidden_layer_sizes=cfg.mlp_hidden, max_iter=cfg.mlp_max_iter, random_state=cfg.seed),
    }
    model = builders[method]()
    model.fit(X_train, y_train)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(X_test)
    else:
        score = model.predict(X_test).astype(np.float64)
    return np.asarray(score, dtype=np.float64), model.predict(X_test)


def run_prediction(provider: StateProvider, global_labels: np.ndarray, y: np.ndarray, layers: List[int], cfg: PredictionConfig, *, store: Optional[ExperimentStore] = None) -> PredictionResult:
    import warnings
    warnings.warn('Legacy prediction cross-validates a supplied map and cannot verify map-training isolation. '
                  'For paper reproduction use hss run with evaluation.mode=prediction; it fits the map on train only.',
                  FutureWarning, stacklevel=2)
    if len(y) == 0:
        empty = pd.DataFrame(columns=['method', 'fold', 'auroc', 'accuracy', 'f1'])
        summary = pd.DataFrame(columns=['method', 'mean_auroc', 'std_auroc', 'mean_accuracy', 'std_accuracy', 'mean_f1', 'std_f1'])
        return PredictionResult(summary_df=summary, fold_df=empty, config=cfg.to_dict())
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    hss_methods = {'HSS-NB', 'HSS-Markov'}
    need_cont = bool(set(cfg.methods) - hss_methods)
    fold_records = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        y_tr, y_val = y[train_idx], y[val_idx]
        gl_tr, gl_val = global_labels[train_idx], global_labels[val_idx]
        X_tr = X_val = None
        if need_cont:
            X_tr = _get_continuous_features(provider, layers, train_idx, cfg.continuous_feature_mode)
            X_val = _get_continuous_features(provider, layers, val_idx, cfg.continuous_feature_mode)
            if cfg.continuous_standardize:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_val = scaler.transform(X_val)
        for method in cfg.methods:
            try:
                if method == 'HSS-NB':
                    score, pred = _run_hss_nb(gl_tr, y_tr, gl_val, cfg.alpha)
                elif method == 'HSS-Markov':
                    score, pred = _run_hss_markov(gl_tr, y_tr, gl_val, cfg.alpha)
                else:
                    score, pred = _run_sklearn(method, X_tr, y_tr, X_val, cfg)
                fold_records.append({'method': method, 'fold': fold_idx, **_eval_fold(y_val, score, pred)})
            except Exception as exc:
                _log.warning('prediction fold=%d method=%s failed: %s', fold_idx, method, exc)
                fold_records.append({'method': method, 'fold': fold_idx, 'auroc': np.nan, 'accuracy': np.nan, 'f1': np.nan})
    fold_df = pd.DataFrame(fold_records)
    summary_rows = []
    for method in cfg.methods:
        sub = fold_df[fold_df['method'] == method]
        row = {'method': method}
        for metric in ['auroc', 'accuracy', 'f1']:
            values = sub[metric].dropna()
            row[f'mean_{metric}'] = float(values.mean()) if len(values) else np.nan
            row[f'std_{metric}'] = float(values.std()) if len(values) else np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    if store is not None:
        store.save_csv('prediction/fold_results.csv', fold_df)
        store.save_csv('prediction/summary.csv', summary_df)
        store.save_json('prediction/prediction_config.json', cfg.to_dict())
    return PredictionResult(summary_df=summary_df, fold_df=fold_df, config=cfg.to_dict())
