"""Label-safe holdouts, categorical counts, and response-level early warning."""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def grouped_split(meta, config):
    groups = meta.groupby("group_id", sort=True).label
    if meta.label.isna().any() or groups.nunique().max() != 1:
        raise ValueError(
            "Supervised experiments require one known, consistent binary label per sample"
        )
    labels = groups.first().astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Both classes are required for stratified evaluation")
    train, hold = train_test_split(
        labels.index.to_numpy(),
        train_size=config.train_fraction,
        random_state=config.split_seed,
        stratify=labels.to_numpy(),
    )
    val = np.array([], dtype=int)
    test = hold
    if config.mode == "monitoring":
        val, test = train_test_split(
            hold,
            train_size=config.validation_fraction / (1 - config.train_fraction),
            random_state=config.split_seed + 1,
            stratify=labels.loc[hold].to_numpy(),
        )
    result = {
        key: np.flatnonzero(meta.group_id.isin(ids).to_numpy())
        for key, ids in [("train", train), ("validation", val), ("test", test)]
    }
    assert not (set(train) & set(test) or set(val) & set(test) or set(train) & set(val))
    return result


class CountNB:
    """Per-layer vocabulary with Laplace emissions and empirical class priors."""

    def __init__(self, vocabularies, alpha=1.0):
        self.vocabularies = [np.asarray(v) for v in vocabularies]
        self.alpha = alpha

    def fit(self, states, y):
        if self.alpha <= 0 or set(np.unique(y)) != {0, 1}:
            raise ValueError("CountNB needs positive smoothing and both classes")
        self.log_prior = np.log(np.bincount(y.astype(int), minlength=2) / len(y))
        self.tables = []
        for j, vocab in enumerate(self.vocabularies):
            table = np.full((2, len(vocab)), self.alpha)
            for pos, state in enumerate(vocab):
                for label in (0, 1):
                    table[label, pos] += np.sum((states[:, j] == state) & (y == label))
            self.tables.append(np.log(table / table.sum(axis=1, keepdims=True)))
        return self

    def predict_proba(self, states):
        scores = np.tile(self.log_prior, (len(states), 1))
        for j, (vocab, table) in enumerate(zip(self.vocabularies, self.tables)):
            mapping = {int(v): i for i, v in enumerate(vocab)}
            positions = np.array([mapping[int(s)] for s in states[:, j]])
            scores += table[:, positions].T
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


def binary_metrics(y, score, *, threshold=0.5):
    return {
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None,
        "accuracy": float(accuracy_score(y, np.asarray(score) >= threshold)),
        "n": len(y),
    }


def calibrate_threshold(meta, scores, far, far_scope="all_boundaries"):
    """Threshold is selected only from maximum score of each successful response.

    Alerts use strict >, so ties are conservative. The empirical validation FAR
    is <= the requested rate, including very small validation sets.
    """
    if far_scope not in ("all_boundaries", "nonfinal"):
        raise ValueError("Unknown FAR boundary scope")
    if not 0 <= far < 1 or not np.isfinite(scores).all():
        raise ValueError("FAR must be in [0,1); scores must be finite")
    frame = meta.copy()
    frame["score"] = scores
    success_ids = frame.loc[frame.label == 1, "group_id"].unique()
    eligible = (
        frame if far_scope == "all_boundaries" else frame[~frame.is_final.astype(bool)]
    )
    # Responses with no eligible boundary remain in the denominator and cannot
    # alarm. A finite floor keeps their threshold serializable and reproducible.
    floor = float(np.nextafter(np.min(scores), -np.inf))
    successful = (
        eligible[eligible.label == 1]
        .groupby("group_id")
        .score.max()
        .reindex(success_ids, fill_value=floor)
        .to_numpy()
    )
    if not len(successful):
        raise ValueError("FAR calibration needs successful validation responses")
    ordered = np.sort(successful)
    allowed = int(np.floor(far * len(ordered)))
    threshold = float(ordered[max(0, len(ordered) - allowed - 1)])
    return threshold, float((successful > threshold).mean())


def monitoring_metrics(meta, scores, threshold, far_scope="all_boundaries"):
    if far_scope not in ("all_boundaries", "nonfinal"):
        raise ValueError("Unknown FAR boundary scope")
    frame = meta.copy()
    frame["score"] = scores
    cases = []
    for group, part in frame.groupby("group_id", sort=True):
        part = part.sort_values("token_end")
        failed = int(part.label.iloc[0] == 0)
        alarms = part[part.score > threshold]
        early = alarms[~alarms.is_final.astype(bool)]
        total = int(part.n_tokens.iloc[0])
        # Last available boundary no later than 50% of tokens; never peek forward.
        half = part[part.token_end <= 0.5 * total]
        cases.append(
            {
                "group_id": int(group),
                "failed": failed,
                "alarm": len(alarms if far_scope == "all_boundaries" else early) > 0,
                "early_detected": len(early) > 0,
                "saved_fraction": (1 - int(early.token_end.iloc[0]) / total)
                if len(early)
                else 0.0,
                "score_half": float(half.score.iloc[-1]) if len(half) else np.nan,
                "score_final": float(part.score.iloc[-1]),
            }
        )
    cases = pd.DataFrame(cases)
    failure, success = cases[cases.failed == 1], cases[cases.failed == 0]
    half = cases.dropna(subset=["score_half"])
    result = {
        "threshold": threshold,
        "far_scope": far_scope,
        "n_responses": len(cases),
        "test_far": float(success.alarm.mean()) if len(success) else None,
        "early_detection_rate": float(failure.early_detected.mean())
        if len(failure)
        else None,
        "saved_token_fraction_failed": float(failure.saved_fraction.mean())
        if len(failure)
        else None,
        "saved_token_fraction_all": float(cases.saved_fraction.mean()),
        "auroc_half": binary_metrics(half.failed, half.score_half)["auroc"]
        if len(half)
        else None,
        "half_coverage": len(half) / len(cases),
        "auroc_final": binary_metrics(cases.failed, cases.score_final)["auroc"],
    }
    return result, cases


def characterize(states, meta, layers):
    associations, tags, transitions = [], [], []
    global_accuracy = (
        float(meta.label.dropna().mean()) if meta.label.notna().any() else None
    )
    for j, layer in enumerate(layers):
        for axis in ("category", "level", "subject", "language", "label"):
            valid = meta[axis].notna().to_numpy()
            table = pd.crosstab(states[valid, j], meta.loc[valid, axis].to_numpy())
            v = chi = p_value = dof = expected_min = sparse_fraction = None
            independent = not meta.loc[valid, "group_id"].duplicated().any()
            if min(table.shape) > 1:
                chi, p_value, dof, expected = chi2_contingency(table, correction=False)
                expected_min = float(expected.min())
                sparse_fraction = float((expected < 5).mean())
                v = float(
                    np.sqrt(chi / (table.to_numpy().sum() * (min(table.shape) - 1)))
                )
            associations.append(
                {
                    "layer": layer,
                    "axis": axis,
                    "cramers_v": v,
                    "chi_square": chi,
                    "p_value": p_value if independent else None,
                    "degrees_of_freedom": dof,
                    "n_rows": int(valid.sum()),
                    "independent_responses": independent,
                    "expected_min": expected_min,
                    "expected_below_5_fraction": sparse_fraction,
                }
            )
        for state in np.unique(states[:, j]):
            part = meta.loc[states[:, j] == state]
            types = part.category.dropna().value_counts(normalize=True)
            accuracy = (
                float(part.label.dropna().mean()) if part.label.notna().any() else None
            )
            tags.append(
                {
                    "layer": layer,
                    "state": int(state),
                    "n": len(part),
                    "dominant_type": str(types.index[0]) if len(types) else None,
                    "dominant_type_fraction": float(types.iloc[0])
                    if len(types)
                    else None,
                    "type_tag": ("single" if types.iloc[0] > 0.6 else "mixed")
                    if len(types)
                    else None,
                    "accuracy": accuracy,
                    "accuracy_delta": accuracy - global_accuracy
                    if accuracy is not None
                    else None,
                    "correctness_tag": (
                        "high"
                        if accuracy - global_accuracy > 0.3
                        else "low"
                        if accuracy - global_accuracy < -0.3
                        else "mixed"
                    )
                    if accuracy is not None
                    else None,
                }
            )
        if j:
            for label in (None, 0, 1):
                mask = (
                    np.ones(len(meta), bool)
                    if label is None
                    else meta.label.to_numpy() == label
                )
                pairs, counts = np.unique(
                    states[mask, j - 1 : j + 1], axis=0, return_counts=True
                )
                from_counts = dict(
                    zip(*np.unique(states[mask, j - 1], return_counts=True))
                )
                for pair, count in zip(pairs, counts):
                    transitions.append(
                        {
                            "from_layer": layers[j - 1],
                            "to_layer": layer,
                            "from_state": int(pair[0]),
                            "to_state": int(pair[1]),
                            "label": label,
                            "count": int(count),
                            "probability": float(count / from_counts[pair[0]]),
                        }
                    )
    return pd.DataFrame(associations), pd.DataFrame(tags), pd.DataFrame(transitions)
