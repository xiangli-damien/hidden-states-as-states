from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import COLORS, save_figure


def plot_trend_stability(
    baseline_k_curve: np.ndarray, layers: np.ndarray,
    seed_k_curves: Optional[np.ndarray] = None,
    subsample_k_curves: Optional[Dict[float, np.ndarray]] = None,
    kmax_k_curves: Optional[Dict[int, np.ndarray]] = None,
    n_sigma: float = 2.0, figsize: Tuple[float, float] = (9, 5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(layers, baseline_k_curve, color="black", lw=2.0, marker="o",
            markersize=4, label="Baseline", zorder=5)
    if seed_k_curves is not None and seed_k_curves.ndim == 2:
        m = np.mean(seed_k_curves, axis=0)
        s = np.std(seed_k_curves, axis=0)
        ax.fill_between(layers, m - n_sigma * s, m + n_sigma * s,
                        alpha=0.20, color=COLORS["blue"],
                        label=f"Seed var. (\u00b1{n_sigma:.0f}\u03c3)")
    if subsample_k_curves is not None:
        cmap = plt.cm.viridis
        fracs = sorted(subsample_k_curves.keys())
        for i, frac in enumerate(fracs):
            if abs(frac - 1.0) < 1e-6:
                continue
            curve = subsample_k_curves[frac]
            ax.plot(layers[:len(curve)], curve, color=cmap(i / max(1, len(fracs) - 1)),
                    lw=1.2, alpha=0.7, label=f"f={frac:.1f}")
    if kmax_k_curves is not None:
        for km, curve in sorted(kmax_k_curves.items()):
            ax.plot(layers[:len(curve)], curve, lw=1.0, linestyle="--",
                    alpha=0.6, label=f"kmax={km}")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Number of Clusters", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25, linestyle=":")
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_centroid_persistence(
    layers: np.ndarray,
    seed_distances: np.ndarray,
    random_distances: np.ndarray,
    sample_distances: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (9, 4.5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if seed_distances.ndim == 2:
        m = np.mean(seed_distances, axis=0)
        s = np.std(seed_distances, axis=0)
        ax.plot(layers[:len(m)], m, color=COLORS["blue"], lw=1.8, marker="o",
                markersize=3, label="Seed Variation")
        ax.fill_between(layers[:len(m)], m - s, m + s, alpha=0.15, color=COLORS["blue"])
    else:
        ax.plot(layers[:len(seed_distances)], seed_distances, color=COLORS["blue"],
                lw=1.8, marker="o", markersize=3, label="Seed Variation")
    if sample_distances is not None:
        if sample_distances.ndim == 2:
            md = np.mean(sample_distances, axis=0)
        else:
            md = sample_distances
        ax.plot(layers[:len(md)], md, color=COLORS["orange"], lw=1.5, marker="s",
                markersize=3, label="Sample Variation")
    if random_distances.ndim == 2:
        mr = np.mean(random_distances, axis=0)
        sr = np.std(random_distances, axis=0)
        ax.plot(layers[:len(mr)], mr, color=COLORS["gray"], lw=1.5, linestyle="--",
                label="Random Baseline")
        ax.fill_between(layers[:len(mr)], mr - sr, mr + sr, alpha=0.10, color=COLORS["gray"])
    else:
        ax.plot(layers[:len(random_distances)], random_distances, color=COLORS["gray"],
                lw=1.5, linestyle="--", label="Random Baseline")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Bipartite Cosine Distance", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=":")
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_tolerance_sweep(
    tolerance_df: pd.DataFrame, target_avg_k: float = 8.0,
    figsize: Tuple[float, float] = (8, 4.5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=figsize)
    tols = tolerance_df["tolerance"].to_numpy()
    avg_ks = tolerance_df["avg_k"].to_numpy()
    ax1.plot(tols, avg_ks, color=COLORS["blue"], lw=2.0, marker="o", markersize=5, label="avg(k)")
    ax1.axhline(target_avg_k, color="black", linestyle="--", lw=1.0, alpha=0.6,
                label=f"target={target_avg_k:.0f}")
    ax1.set_xlabel("Tolerance", fontsize=10)
    ax1.set_ylabel("Average k", color=COLORS["blue"], fontsize=10)
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])
    if "std_k" in tolerance_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(tols, tolerance_df["std_k"].to_numpy(), color=COLORS["orange"],
                 lw=1.2, marker="s", markersize=3, alpha=0.7, label="std(k)")
        ax2.set_ylabel("Std(k)", color=COLORS["orange"], fontsize=10)
        ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle=":")
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig

