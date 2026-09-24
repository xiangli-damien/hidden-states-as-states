"""Exploratory, label-free route scores on already fitted state maps.

Cross-fitting excludes each row from transition counts, but the input clustering
was fitted on all rows. This is NOT an inductive held-out detector evaluation.
Correctness labels are read only after the fixed scores have been computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def route_scores(states, folds):
    """Laplace alpha=1; use decoder layers, excluding embedding position 0."""
    z = np.column_stack([np.unique(col, return_inverse=True)[1] for col in states.T])
    n, depth = z.shape
    sizes = z.max(axis=0) + 1
    scores = {key: np.zeros(n) for key in ("node_rarity", "transition_surprise", "transition_excess", "final_node_rarity")}
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        marginals = []
        for layer in range(depth):
            counts = np.bincount(z[train, layer], minlength=sizes[layer]) + 1.0
            marginals.append(counts / counts.sum())
            scores["node_rarity"][test] -= np.log(marginals[-1][z[test, layer]]) / depth
        scores["final_node_rarity"][test] = -np.log(marginals[-1][z[test, -1]])
        for layer in range(depth - 1):
            counts = np.ones((sizes[layer], sizes[layer + 1]))
            np.add.at(counts, (z[train, layer], z[train, layer + 1]), 1)
            probs = counts / counts.sum(axis=1, keepdims=True)
            conditional = probs[z[test, layer], z[test, layer + 1]]
            marginal = marginals[layer + 1][z[test, layer + 1]]
            scores["transition_surprise"][test] -= np.log(conditional) / (depth - 1)
            scores["transition_excess"][test] -= np.log(conditional / marginal) / (depth - 1)
    return scores


def auc_interval(y, score, seed=831, repeats=300):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        idx = rng.integers(len(y), size=len(y))
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], score[idx]))
    return [float(x) for x in np.quantile(values, [.025, .975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmm", type=Path, required=True)
    parser.add_argument("--mfa", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {"gmm": args.gmm, "mfa": args.mfa}
    rows = {name: pd.read_parquet(path / "rows.parquet") for name, path in sources.items()}
    assert rows["gmm"].sample_id.tolist() == rows["mfa"].sample_id.tolist()
    assert rows["gmm"].sample_id.is_unique
    meta = rows["mfa"]
    group = meta.group_id.fillna(meta.sample_id).astype(str)
    folds = np.array([int(hashlib.sha256(("route-preflight-v1:" + x).encode()).hexdigest(), 16) % 5 for x in group])
    results = pd.DataFrame({"sample_id": meta.sample_id, "fold": folds})
    model_scores = {}
    for method, path in sources.items():
        full = np.load(path / "states.npy")
        assert full.shape == (len(meta), 29) and np.all(full >= 0)
        scores = route_scores(full[:, 1:], folds)
        # The route scores must not depend on names assigned to layer-local states.
        relabeled = np.zeros_like(full[:, 1:])
        rng = np.random.default_rng(209)
        for layer in range(relabeled.shape[1]):
            _, inv = np.unique(full[:, layer + 1], return_inverse=True)
            relabeled[:, layer] = rng.permutation(inv.max() + 1)[inv] + 300 * layer
        for key, value in route_scores(relabeled, folds).items():
            np.testing.assert_allclose(value, scores[key], atol=1e-12)
        scores["global_id_switch_rate"] = (full[:, 1:-1] != full[:, 2:]).mean(axis=1)
        model_scores[method] = scores
        for key, score in scores.items():
            results[method + "__" + key] = score
    # Scores are frozen and saved before using correctness in evaluation.
    results.to_parquet(args.out / "unlabeled_scores.parquet", index=False)
    assert np.array_equal(rows["gmm"].label, rows["mfa"].label)
    assert meta.label.isin([0, 1]).all()
    failure = 1 - meta.label.to_numpy(dtype=int)
    evaluations = []
    for method, scores in model_scores.items():
        for key, score in scores.items():
            evaluations.append(dict(method=method, score=key, failure_auroc=float(roc_auc_score(failure, score)),
                                    bootstrap_ci=auc_interval(failure, score),
                                    mean_correct=float(score[failure == 0].mean()),
                                    mean_incorrect=float(score[failure == 1].mean())))
    length = meta.n_tokens.to_numpy(dtype=float)
    evaluations.append(dict(method="shared", score="response_length", failure_auroc=float(roc_auc_score(failure, length)),
                            bootstrap_ci=auc_interval(failure, length),
                            mean_correct=float(length[failure == 0].mean()), mean_incorrect=float(length[failure == 1].mean())))
    pd.DataFrame(evaluations).to_csv(args.out / "score_evaluation.csv", index=False)
    report = dict(samples=len(meta), correct=int((failure == 0).sum()), incorrect=int(failure.sum()),
                  decoder_layers=28, alignment="Existing cosine Hungarian eta=.6 unchanged",
                  score_orientation="Higher score predeclared as greater failure risk; never reversed after evaluation",
                  smoothing="Laplace alpha=1 over actual next-layer states; no globally inactive columns",
                  folds="5 deterministic group-hash folds; no correctness labels used for splits or score fitting",
                  evaluations=evaluations,
                  limitations=["Exploratory: original clusters and K/rank were selected on all 5000 samples.",
                               "Cross-fitting isolates transition counts only, not clustering or model selection.",
                               "Bootstrap intervals condition on frozen clusters and scores; no refitting uncertainty included.",
                               "Completed-answer means are retrospective and cannot establish early failure prediction.",
                               "No causal claim, no calibrated correctness probabilities, no false-alarm guarantee.",
                               "The data have been examined in prior studies; this is not a previously untouched test set."],
                  sources={name: dict(path=str(path), hashes={f: hashlib.sha256((path / f).read_bytes()).hexdigest()
                                                           for f in ["states.npy", "rows.parquet", "config.json"]})
                           for name, path in sources.items()})
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(evaluations)[["method", "score", "failure_auroc", "mean_correct", "mean_incorrect"]].to_string(index=False))


if __name__ == "__main__":
    main()
