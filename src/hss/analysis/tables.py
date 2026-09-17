"""Deterministic derived tables. No raw activation access and no model fitting."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform


def state_map(result):
    states, meta, layers = result.states, result.rows, result.layers
    nodes, edges = [], []
    labels = pd.to_numeric(meta.label, errors="coerce").to_numpy()
    baseline = float(np.nanmean(labels)) if np.isfinite(labels).any() else np.nan
    for j, layer in enumerate(layers):
        unique, counts = np.unique(states[:, j], return_counts=True)
        for rank, (state, count) in enumerate(zip(unique, counts)):
            selected = states[:, j] == state
            entropy = np.nan
            if j + 1 < len(layers):
                next_states, transition_counts = np.unique(
                    states[selected, j + 1], return_counts=True
                )
                probabilities = transition_counts / count
                entropy = float(-np.sum(probabilities * np.log2(probabilities)))
                for next_state, n, prob in zip(
                    next_states, transition_counts, probabilities
                ):
                    edges.append(
                        dict(
                            from_layer=layer,
                            to_layer=layers[j + 1],
                            from_state=int(state),
                            to_state=int(next_state),
                            count=int(n),
                            probability=float(prob),
                            frequency=float(n / len(meta)),
                        )
                    )
            y = labels[selected]
            accuracy = float(np.nanmean(y)) if np.isfinite(y).any() else np.nan
            nodes.append(
                dict(
                    layer=layer,
                    position=j,
                    state=int(state),
                    y=rank - (len(unique) - 1) / 2,
                    count=int(count),
                    frequency=float(count / len(meta)),
                    entropy=entropy,
                    accuracy=accuracy,
                    accuracy_delta=accuracy - baseline,
                )
            )
    return pd.DataFrame(nodes), pd.DataFrame(
        edges,
        columns=[
            "from_layer",
            "to_layer",
            "from_state",
            "to_state",
            "count",
            "probability",
            "frequency",
        ],
    )


def dynamics(result):
    counts = pd.Series(np.asarray(result.states).ravel()).value_counts()
    marginal = counts.rename_axis("state").reset_index(name="count")
    marginal["frequency"] = marginal["count"] / marginal["count"].sum()
    marginal["rank"] = np.arange(1, len(marginal) + 1)
    marginal["cumulative_mass"] = marginal.frequency.cumsum()
    profile = pd.DataFrame(result.summary["profile"])
    vocab = result.json("alignment.json")["local_to_global"]
    profile["active_k"] = [
        len(np.unique(result.states[:, j])) for j in range(len(profile))
    ]
    profile["uniform_self_transition"] = 1 / profile.k
    profile.loc[0, "uniform_self_transition"] = np.nan
    profile["inherited_fraction"] = [np.nan] + [
        len(set(vocab[j]) & set(vocab[j - 1])) / len(vocab[j])
        for j in range(1, len(vocab))
    ]
    return marginal, profile


def trajectory_similarity(result, max_rows=1000, seed=42, linkage_method="average"):
    """Bound O(N^2) storage; record every sampled row and hierarchical order.

    Ward is intentionally excluded: it requires Euclidean distances, whereas
    dissimilarity here is Hamming distance between categorical state trajectories.
    """
    if max_rows < 2 or linkage_method not in ("average", "complete", "single"):
        raise ValueError("Need max_rows>=2 and average/complete/single linkage")
    if len(result.states) < 2:
        raise ValueError("Need two trajectories")
    selected = np.sort(
        np.random.default_rng(seed).choice(
            len(result.states), min(max_rows, len(result.states)), replace=False
        )
    )
    distance = pdist(result.states[selected], metric="hamming")
    order = leaves_list(linkage(distance, method=linkage_method))
    rows = result.rows.iloc[selected[order]][["sample_id", "token_end"]].copy()
    rows["result_row"] = selected[order]
    rows["display_position"] = np.arange(len(rows))
    matrix = (1 - squareform(distance))[np.ix_(order, order)].astype("float32")
    return matrix, rows.reset_index(drop=True)


def selection_surface(result):
    rows = []
    for layer in result.json("selection.json"):
        candidates = layer["candidates"]
        finite = [
            c["criterion"]
            for c in candidates
            if c.get("criterion") is not None and np.isfinite(c["criterion"])
        ]
        best = min(finite) if finite else np.nan
        for candidate in candidates:
            value = candidate.get("criterion")
            relative = (
                (value - best) / max(abs(best), 1) if value is not None else np.nan
            )
            rows.append(
                {
                    "layer": layer["layer"],
                    **candidate,
                    "relative_criterion": relative,
                    "selected": candidate["k"] == layer["selected"]["k"],
                    "near_optimal": bool(
                        relative <= result.config["cluster"]["parsimony_tolerance"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def collect_tables(results):
    tables = {
        key: []
        for key in (
            "profiles",
            "evaluation",
            "diagnostics",
            "associations",
            "state_tags",
            "global_layer_purity",
        )
    }
    for result in results:
        fields = result.metadata()
        fields["n_global_states"] = result.summary["n_global_states"]
        _, profile = dynamics(result)
        frames = {
            "profiles": profile,
            "evaluation": pd.DataFrame(result.summary["evaluation"]),
            "diagnostics": pd.DataFrame(result.json("diagnostics.json")),
            **{
                key: result.table(key + ".csv")
                for key in ("associations", "state_tags", "global_layer_purity")
            },
        }
        for name, frame in frames.items():
            if name == "evaluation" and len(frame):
                frame = frame.rename(columns={"method": "predictor"})
            for row in frame.to_dict("records"):
                tables[name].append({**fields, **row})
    return {key: pd.DataFrame(rows) for key, rows in tables.items()}


def compare_trials(results, all_pairs=False):
    """Controls vs one baseline; fixed-K refits additionally compared pairwise.

    Groups never cross dataset snapshots, model revisions or representations.
    Unmatched component counts are exported; equal-K comparisons are identifiable.
    """
    from ..experiments.diagnostics import compare_results

    groups = {}
    for r in results:
        c = r.config
        key = (
            tuple(r.summary["model"]),
            r.summary["snapshot"],
            c["data"]["representation"],
            c["data"]["final_norm"],
            c["evaluation"]["mode"],
        )
        groups.setdefault(key, []).append(r)
    output = []
    for group in groups.values():
        group = sorted(
            group,
            key=lambda r: (
                bool(r.config["evaluation"]["fixed_k_map"]),
                r.config["seed"],
                r.summary["trial_id"],
            ),
        )
        if len(group) < 2:
            continue
        pairs = (
            set(combinations(range(len(group)), 2))
            if all_pairs
            else {(0, j) for j in range(1, len(group))}
        )
        fixed = [
            j for j, r in enumerate(group) if r.config["evaluation"]["fixed_k_map"]
        ]
        pairs.update(combinations(fixed, 2))
        for i, j in sorted(pairs):
            a, b = group[i], group[j]
            for row in compare_results(a.path, b.path):
                output.append(
                    {
                        "reference": a.summary["trial_id"],
                        "target": b.summary["trial_id"],
                        "reference_seed": a.config["seed"],
                        "target_seed": b.config["seed"],
                        "reference_fit_fraction": a.config["evaluation"][
                            "fit_fraction"
                        ],
                        "target_fit_fraction": b.config["evaluation"]["fit_fraction"],
                        "fixed_k_refits": bool(
                            a.config["evaluation"]["fixed_k_map"]
                            and b.config["evaluation"]["fixed_k_map"]
                        ),
                        "equal_k": row["k_reference"] == row["k_target"],
                        **row,
                    }
                )
    return pd.DataFrame(output)


def bootstrap_predictions(result, repeats=1000, seed=42):
    """Stratified response bootstrap CI on saved held-out scores, no refitting.

    This measures test-sample uncertainty, not variability of map training.
    Monitoring is excluded: its boundary rows are not independent responses.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    if result.config["evaluation"]["mode"] != "prediction":
        return pd.DataFrame()
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    frame = result.table("predictions.parquet")
    records = []
    for method, part in frame.groupby("method", sort=True):
        if part.sample_id.duplicated().any():
            raise ValueError("Bootstrap needs one prediction per response")
        y = part.label.to_numpy()
        score = part.score.to_numpy()
        classes = [np.flatnonzero(y == v) for v in (0, 1)]
        if any(not len(x) for x in classes):
            continue
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(repeats):
            idx = np.concatenate(
                [rng.choice(ids, len(ids), replace=True) for ids in classes]
            )
            values.append(
                [
                    roc_auc_score(y[idx], score[idx]),
                    accuracy_score(
                        y[idx], score[idx] >= (0 if method == "LinearSVM" else 0.5)
                    ),
                ]
            )
        for j, metric in enumerate(("auroc", "accuracy")):
            lo, hi = np.quantile(np.asarray(values)[:, j], [0.025, 0.975])
            records.append(
                {
                    "trial_id": result.summary["trial_id"],
                    "predictor": method,
                    "metric": metric,
                    "lower_95": float(lo),
                    "upper_95": float(hi),
                    "repeats": repeats,
                    "seed": seed,
                    "unit": "held-out response",
                    "uncertainty": "stratified test-sample bootstrap, fixed fitted model",
                }
            )
    return pd.DataFrame(records)
