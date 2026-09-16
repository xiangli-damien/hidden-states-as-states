"""Portable CSV tables and scientific plots from completed experiment artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import save_json
from .diagnostics import compare_results


def report(output_root, destination=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    root = Path(output_root).resolve()
    target = Path(destination).resolve() if destination else root / "report"
    target.mkdir(parents=True, exist_ok=True)
    paths = (
        [root]
        if (root / "_SUCCESS.json").exists()
        else sorted(p.parent for p in root.glob("*/_SUCCESS.json"))
    )
    if not paths:
        raise ValueError("No completed experiment outputs to report")
    profiles, metrics, trials = [], [], []
    for path in paths:
        summary = json.loads((path / "summary.json").read_text())
        cfg = json.loads((path / "config.json").read_text())
        fields = {
            "trial_id": summary["trial_id"],
            "name": summary["name"],
            "model": summary["model"][0],
            "method": cfg["cluster"]["method"],
            "representation": cfg["data"]["representation"],
            "final_norm": cfg["data"]["final_norm"],
            "seed": cfg["seed"],
            "snapshot": summary["snapshot"],
            "model_revision": summary["model"][1],
        }
        profiles.extend([{**fields, **row} for row in summary["profile"]])
        metrics.extend(
            [
                {
                    **fields,
                    "predictor": row["method"],
                    **{k: v for k, v in row.items() if k != "method"},
                }
                for row in summary["evaluation"]
            ]
        )
        trials.append(
            {
                "path": str(path),
                **fields,
                "rows": summary["n_rows"],
                "seconds": summary["seconds"],
                "unavailable_baselines": summary["unavailable_baselines"],
            }
        )
        folder = target / summary["trial_id"]
        folder.mkdir(exist_ok=True)
        frame = pd.DataFrame(summary["profile"])
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        axs[0].plot(frame.relative_depth, frame.k, "o-", markersize=3)
        axs[0].set(xlabel="Relative layer depth", ylabel="Selected components K")
        axs[1].plot(frame.relative_depth, frame.self_transition, "o-", markersize=3)
        axs[1].set(
            xlabel="Relative layer depth",
            ylabel="Self-transition probability",
            ylim=(0, 1),
        )
        fig.savefig(folder / "profile.png", dpi=160)
        plt.close(fig)
        selection = json.loads((path / "selection.json").read_text())
        scan = pd.DataFrame(
            [{"layer": s["layer"], **c} for s in selection for c in s["candidates"]]
        )
        scan.to_csv(folder / "selection_surface.csv", index=False)
        surface = scan.pivot(index="k", columns="layer", values="criterion")
        denominator = surface.min().abs().clip(lower=1.0)
        relative = (surface - surface.min()) / denominator
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        im = ax.imshow(relative, origin="lower", aspect="auto", interpolation="nearest")
        ax.set(
            xlabel="Layer",
            ylabel="Candidate K",
            title="Relative model-selection criterion",
        )
        ax.set_xticks(
            range(len(surface.columns)), surface.columns, rotation=90, fontsize=7
        )
        stride = max(1, len(surface.index) // 15)
        ax.set_yticks(range(0, len(surface.index), stride), surface.index[::stride])
        fig.colorbar(im, ax=ax)
        fig.savefig(folder / "selection_surface.png", dpi=160)
        plt.close(fig)
        states = np.load(path / "states.npy", mmap_mode="r")
        meta = pd.read_parquet(path / "rows.parquet")
        graph_nodes, edges = [], []
        positions = {}
        for layer in range(states.shape[1]):
            unique, counts = np.unique(states[:, layer], return_counts=True)
            for i, (state, count) in enumerate(zip(unique, counts)):
                pos = (layer, i / max(1, len(unique) - 1))
                positions[(layer, int(state))] = pos
                accuracy = meta.loc[states[:, layer] == state, "label"].mean()
                graph_nodes.append(
                    (
                        *pos,
                        int(state),
                        count,
                        float(accuracy) if pd.notna(accuracy) else 0.5,
                    )
                )
            if layer:
                pairs, counts = np.unique(
                    states[:, layer - 1 : layer + 1], axis=0, return_counts=True
                )
                edges.extend(
                    (positions[(layer - 1, int(a))], positions[(layer, int(b))], int(n))
                    for (a, b), n in zip(pairs, counts)
                )
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        ax.add_collection(
            LineCollection(
                [[a, b] for a, b, n in edges],
                colors="gray",
                alpha=0.2,
                linewidths=[0.2 + 3 * n / len(states) for a, b, n in edges],
            )
        )
        node = np.asarray(graph_nodes)
        points = ax.scatter(
            node[:, 0],
            node[:, 1],
            s=8 + 180 * node[:, 3] / len(states),
            c=node[:, 4],
            cmap="coolwarm_r",
            vmin=0,
            vmax=1,
            zorder=3,
        )
        ax.set(
            xlabel="Layer position",
            ylabel="State position within layer",
            title=summary["name"],
        )
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-0.5, states.shape[1] - 0.5)
        fig.colorbar(points, ax=ax, label="Positive-label frequency")
        fig.savefig(folder / "state_graph.png", dpi=160)
        plt.close(fig)
    pd.DataFrame(profiles).to_csv(target / "profiles.csv", index=False)
    pd.DataFrame(metrics).to_csv(target / "evaluation.csv", index=False)
    save_json(target / "experiments.json", trials)
    # Compare same-model/representation trials on common rows (seeds, preprocessing,
    # alignment, MFA controls). Cross-model profiles are compared on relative depth.
    comparisons = []
    baselines = {}
    for trial in sorted(
        trials,
        key=lambda t: (
            t["name"] not in ("qwen_math_map", "default_map"),
            t["seed"],
            t["trial_id"],
        ),
    ):
        group = (
            trial["model"],
            trial["model_revision"],
            trial["representation"],
            trial["final_norm"],
            trial["snapshot"],
        )
        if group not in baselines:
            baselines[group] = trial
        else:
            reference = baselines[group]
            for row in compare_results(reference["path"], trial["path"]):
                comparisons.append(
                    {
                        "reference": reference["trial_id"],
                        "target": trial["trial_id"],
                        **row,
                    }
                )
    pd.DataFrame(comparisons).to_csv(target / "reliability.csv", index=False)
    result = {
        "experiments": len(paths),
        "path": str(target),
        "tables": ["profiles.csv", "evaluation.csv", "reliability.csv"],
    }
    save_json(target / "index.json", result)
    return result
