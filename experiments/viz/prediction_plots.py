from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import COLORS, save_figure


_METHOD_STYLE = {
    "HSS-NB": {"color": COLORS["blue"], "marker": "o", "hatch": ""},
    "HSS-Markov": {"color": COLORS["cyan"], "marker": "s", "hatch": ""},
    "LDA": {"color": COLORS["gray"], "marker": "^", "hatch": "//"},
    "GaussianNB": {"color": COLORS["olive"], "marker": "D", "hatch": "//"},
    "Logistic": {"color": COLORS["orange"], "marker": "v", "hatch": "\\\\"},
    "LinearSVM": {"color": COLORS["brown"], "marker": "<", "hatch": "\\\\"},
    "MLP": {"color": COLORS["red"], "marker": ">", "hatch": "xx"},
}


def plot_prediction_comparison(
    summary_df: pd.DataFrame,
    metric: str = "auroc",
    figsize: Tuple[float, float] = (8, 4.5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    methods = summary_df["method"].unique().tolist()
    x = np.arange(len(methods))
    width = 0.55

    means = []
    stds = []
    colors = []
    hatches = []
    for m in methods:
        sub = summary_df[summary_df["method"] == m]
        col = f"mean_{metric}" if f"mean_{metric}" in sub.columns else metric
        means.append(float(sub[col].iloc[0]) if len(sub) > 0 else 0.0)
        std_col = f"std_{metric}"
        if std_col in sub.columns:
            stds.append(float(sub[std_col].iloc[0]))
        else:
            stds.append(0.0)
        style = _METHOD_STYLE.get(m, {"color": COLORS["gray"], "hatch": ""})
        colors.append(style["color"])
        hatches.append(style["hatch"])

    bars = ax.bar(x, means, width, yerr=stds if any(s > 0 for s in stds) else None,
                  color=colors, edgecolor="black", linewidth=0.6, alpha=0.85,
                  capsize=3, error_kw={"lw": 1.0})

    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)

    hss_methods = {"HSS-NB", "HSS-Markov"}
    best_hss = max(
        (means[i] for i, m in enumerate(methods) if m in hss_methods),
        default=None,
    )
    if best_hss is not None:
        ax.axhline(best_hss, color=COLORS["blue"], linestyle="--", lw=0.8,
                   alpha=0.5)

    metric_label = {"auroc": "AUROC", "accuracy": "Accuracy", "f1": "F1"}
    ax.set_ylabel(metric_label.get(metric, metric), fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    for i, (m_val, s_val) in enumerate(zip(means, stds)):
        label = f"{m_val:.1f}"
        if metric in ("accuracy", "f1"):
            label = f"{m_val:.1%}"
        ax.text(i, m_val + s_val + 0.005, label, ha="center", va="bottom",
                fontsize=7, fontweight="medium")

    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_fold_distribution(
    fold_df: pd.DataFrame,
    metric: str = "auroc",
    figsize: Tuple[float, float] = (8, 5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    methods = fold_df["method"].unique().tolist()
    data_list = []
    colors_list = []
    for m in methods:
        sub = fold_df[fold_df["method"] == m]
        data_list.append(sub[metric].to_numpy())
        style = _METHOD_STYLE.get(m, {"color": COLORS["gray"]})
        colors_list.append(style["color"])

    bp = ax.boxplot(data_list, labels=methods, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black",
                                   markersize=4))

    for patch, c in zip(bp["boxes"], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    metric_label = {"auroc": "AUROC", "accuracy": "Accuracy", "f1": "F1"}
    ax.set_ylabel(metric_label.get(metric, metric), fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig