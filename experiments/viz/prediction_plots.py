from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import COLORS, save_figure


_METHOD_STYLE = {
    "HSS-NB":     {"color": COLORS["blue"],   "hatch": ""},
    "HSS-Markov": {"color": COLORS["cyan"],   "hatch": ""},
    "LDA":        {"color": COLORS["gray"],   "hatch": "//"},
    "GaussianNB": {"color": COLORS["olive"],  "hatch": "//"},
    "Logistic":   {"color": COLORS["orange"], "hatch": "\\\\"},
    "LinearSVM":  {"color": COLORS["brown"],  "hatch": "\\\\"},
    "MLP":        {"color": COLORS["red"],    "hatch": "xx"},
}


def plot_prediction_comparison(
    summary_df: pd.DataFrame, metric: str = "auroc",
    figsize: Tuple[float, float] = (8, 4.5), out_path=None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    methods = summary_df["method"].unique().tolist()
    x = np.arange(len(methods))
    width = 0.55

    means, stds, colors, hatches = [], [], [], []
    for m in methods:
        sub = summary_df[summary_df["method"] == m]
        col = f"mean_{metric}" if f"mean_{metric}" in sub.columns else metric
        means.append(float(sub[col].iloc[0]) if len(sub) > 0 else 0.0)
        std_col = f"std_{metric}"
        stds.append(float(sub[std_col].iloc[0]) if std_col in sub.columns and len(sub) > 0 else 0.0)
        style = _METHOD_STYLE.get(m, {"color": COLORS["gray"], "hatch": ""})
        colors.append(style["color"])
        hatches.append(style["hatch"])

    bars = ax.bar(x, means, width,
                  yerr=stds if any(s > 0 for s in stds) else None,
                  color=colors, edgecolor="black", linewidth=0.6, alpha=0.85,
                  capsize=3, error_kw={"lw": 1.0})
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)

    hss_set = {"HSS-NB", "HSS-Markov"}
    best_hss = max((means[i] for i, m in enumerate(methods) if m in hss_set), default=None)
    if best_hss is not None:
        ax.axhline(best_hss, color=COLORS["blue"], linestyle="--", lw=0.8, alpha=0.5)

    label_map = {"auroc": "AUROC", "accuracy": "Accuracy", "f1": "F1"}
    ax.set_ylabel(label_map.get(metric, metric), fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    for i, (mv, sv) in enumerate(zip(means, stds)):
        ax.text(i, mv + sv + 0.005, f"{mv:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="medium")

    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_fold_distribution(
    fold_df: pd.DataFrame, metric: str = "auroc",
    figsize: Tuple[float, float] = (8, 5), out_path=None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    methods = fold_df["method"].unique().tolist()
    data_list, colors_list = [], []
    for m in methods:
        data_list.append(fold_df[fold_df["method"] == m][metric].to_numpy())
        colors_list.append(_METHOD_STYLE.get(m, {"color": COLORS["gray"]})["color"])

    bp = ax.boxplot(data_list, labels=methods, patch_artist=True, widths=0.5,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markersize=4))
    for patch, c in zip(bp["boxes"], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    label_map = {"auroc": "AUROC", "accuracy": "Accuracy", "f1": "F1"}
    ax.set_ylabel(label_map.get(metric, metric), fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig

