from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from .style import (
    CMAP_STABILITY, CMAP_PROB, CMAP_DISC,
    COLOR_EDGE, text_color_for_stability, text_color_for_bg, save_figure,
)
from .utils import (
    add_curved_edge, quantile_scale, scale_linear, scale_linear_vec,
    compute_positions, adaptive_figsize, adaptive_radius, filter_edges,
    add_colorbar,
)


def plot_dynamics_unlabeled(
    node_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    layers: np.ndarray,
    N_total: int,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    positions = compute_positions(node_df, layers)
    n_layers = len(layers)
    max_clusters = int(node_df.groupby("layer").size().max())
    figsize = adaptive_figsize(n_layers, max_clusters)
    min_r, max_r = adaptive_radius(n_layers, max_clusters)

    fig, ax = plt.subplots(figsize=figsize)
    df = filter_edges(flow_df)

    if len(df) > 0:
        cnt = df["count"].to_numpy(dtype=np.float64)
        w_lo, w_hi = quantile_scale(cnt, 0.10, 0.90, 1.0, 10.0)
    else:
        w_lo, w_hi = 1.0, 10.0

    for r in df.itertuples(index=False):
        lf, lt = int(r.layer_from), int(r.layer_to)
        cf, ct = int(r.cluster_from), int(r.cluster_to)
        if (lf, cf) not in positions or (lt, ct) not in positions:
            continue
        p1, p2 = positions[(lf, cf)], positions[(lt, ct)]
        lw = scale_linear(float(r.count), w_lo, w_hi, 0.5, 3.5)
        alpha = scale_linear(float(r.count), w_lo, w_hi, 0.08, 0.32)
        if cf == ct:
            lw *= 1.10
            alpha = min(0.40, alpha + 0.06)
        add_curved_edge(ax, p1, p2, color=COLOR_EDGE, lw=lw, alpha=alpha)

    sizes = node_df["count"].to_numpy(dtype=np.float64)
    s_t = np.sqrt(sizes)
    s_lo, s_hi = quantile_scale(s_t, 0.05, 0.95, 1.0, 10.0)
    radii = scale_linear_vec(s_t, s_lo, s_hi, min_r, max_r)
    r_lookup = {}
    for idx, (_, row) in enumerate(node_df.iterrows()):
        r_lookup[(int(row["layer"]), int(row["cluster"]))] = float(radii[idx])

    ent = node_df["out_entropy"].to_numpy(dtype=np.float64) if "out_entropy" in node_df.columns else np.zeros(len(node_df))
    ent_v = ent[np.isfinite(ent)]
    vmin = float(np.nanmin(ent_v)) if ent_v.size else 0.0
    vmax = float(np.nanmax(ent_v)) if ent_v.size else 1.0
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))

    for r in node_df.itertuples(index=False):
        layer, cluster = int(r.layer), int(r.cluster)
        if (layer, cluster) not in positions:
            continue
        x, y = positions[(layer, cluster)]
        radius = r_lookup.get((layer, cluster), min_r)
        val = float(getattr(r, "out_entropy", np.nan))
        face = CMAP_STABILITY(norm(val)) if np.isfinite(val) else (0.85, 0.85, 0.85, 1.0)
        circ = plt.Circle((x, y), radius, facecolor=face, edgecolor="black", lw=0.5, alpha=0.95, zorder=3)
        ax.add_patch(circ)
        if len(node_df) <= 400:
            tc = text_color_for_stability(val, vmin, vmax) if np.isfinite(val) else "black"
            ax.text(x, y, str(cluster), ha="center", va="center", fontsize=6.5, color=tc, fontweight="medium", zorder=4)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Layer", fontsize=10)
    xt = np.arange(n_layers)
    ax.set_xticks(xt)
    if n_layers <= 20:
        ax.set_xticklabels([f"L{int(l)}" for l in layers], fontsize=8)
    else:
        ax.set_xticklabels([f"{int(l)}" for l in layers], rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle=":", alpha=0.20, zorder=0)
    ax.set_axisbelow(True)

    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        xpad = max_r + 0.15
        ypad = max_r + 0.15
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    sm = mpl.cm.ScalarMappable(cmap=CMAP_STABILITY, norm=norm)
    sm.set_array([])
    add_colorbar(fig, ax, sm, "Transition Entropy")
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_dynamics_absolute(
    node_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    layers: np.ndarray,
    N_total: int,
    baseline: float,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    positions = compute_positions(node_df, layers)
    n_layers = len(layers)
    max_clusters = int(node_df.groupby("layer").size().max())
    figsize = adaptive_figsize(n_layers, max_clusters)
    min_r, max_r = adaptive_radius(n_layers, max_clusters)

    fig, ax = plt.subplots(figsize=figsize)
    df = filter_edges(flow_df)
    norm_prob = Normalize(vmin=0.0, vmax=1.0)

    if len(df) > 0:
        cnt = df["count"].to_numpy(dtype=np.float64)
        w_lo, w_hi = quantile_scale(cnt, 0.10, 0.90, 1.0, 10.0)
    else:
        w_lo, w_hi = 1.0, 10.0

    for r in df.itertuples(index=False):
        lf, lt = int(r.layer_from), int(r.layer_to)
        cf, ct = int(r.cluster_from), int(r.cluster_to)
        if (lf, cf) not in positions or (lt, ct) not in positions:
            continue
        p1, p2 = positions[(lf, cf)], positions[(lt, ct)]
        lw = scale_linear(float(r.count), w_lo, w_hi, 0.5, 3.5)
        alpha = scale_linear(float(r.count), w_lo, w_hi, 0.23, 0.57)
        alpha = min(0.85, alpha)
        pr = float(getattr(r, "pos_rate", baseline))
        color = CMAP_PROB(norm_prob(pr))
        if cf == ct:
            lw *= 1.10
            alpha = min(0.90, alpha + 0.10)
        add_curved_edge(ax, p1, p2, color=color, lw=lw, alpha=alpha)

    sizes = node_df["count"].to_numpy(dtype=np.float64)
    s_t = np.sqrt(sizes)
    s_lo, s_hi = quantile_scale(s_t, 0.05, 0.95, 1.0, 10.0)
    radii = scale_linear_vec(s_t, s_lo, s_hi, min_r, max_r)
    r_lookup = {}
    for idx, (_, row) in enumerate(node_df.iterrows()):
        r_lookup[(int(row["layer"]), int(row["cluster"]))] = float(radii[idx])

    for r in node_df.itertuples(index=False):
        layer, cluster = int(r.layer), int(r.cluster)
        if (layer, cluster) not in positions:
            continue
        x, y = positions[(layer, cluster)]
        radius = r_lookup.get((layer, cluster), min_r)
        pr = float(getattr(r, "pos_rate", baseline))
        face = CMAP_PROB(norm_prob(pr))
        circ = plt.Circle((x, y), radius, facecolor=face, edgecolor="black", lw=0.6, alpha=0.95, zorder=3)
        ax.add_patch(circ)
        if len(node_df) <= 500:
            tc = "white" if (pr < 0.30 or pr > 0.70) else "black"
            ax.text(x, y, str(cluster), ha="center", va="center", fontsize=6.0, color=tc, fontweight="medium", zorder=4)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Layer", fontsize=10)
    xt = np.arange(n_layers)
    ax.set_xticks(xt)
    if n_layers <= 20:
        ax.set_xticklabels([f"L{int(l)}" for l in layers], fontsize=8)
    else:
        ax.set_xticklabels([f"{int(l)}" for l in layers], rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle=":", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        ax.set_xlim(min(xs) - max_r - 0.15, max(xs) + max_r + 0.15)
        ax.set_ylim(min(ys) - max_r - 0.15, max(ys) + max_r + 0.15)

    sm = mpl.cm.ScalarMappable(cmap=CMAP_PROB, norm=norm_prob)
    sm.set_array([])
    cbar = add_colorbar(fig, ax, sm, "P(correct | cluster)", pad=0.12)
    cbar.ax.plot([0, 1], [baseline, baseline], color="black", lw=2.0, transform=cbar.ax.get_yaxis_transform(), clip_on=False)
    cbar.ax.plot(-0.3, baseline, marker=">", markersize=5, color="black", transform=cbar.ax.get_yaxis_transform(), clip_on=False)
    cbar.ax.text(0.5, -0.08, f"baseline={baseline:.0%}", transform=cbar.ax.transAxes, fontsize=7, ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="gray", alpha=0.9))
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_dynamics_relative(
    node_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    layers: np.ndarray,
    N_total: int,
    baseline: float,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    positions = compute_positions(node_df, layers)
    n_layers = len(layers)
    max_clusters = int(node_df.groupby("layer").size().max())
    figsize = adaptive_figsize(n_layers, max_clusters)
    min_r, max_r = adaptive_radius(n_layers, max_clusters)

    fig, ax = plt.subplots(figsize=figsize)
    df = filter_edges(flow_df)

    disc_vals = []
    if "disc" in node_df.columns:
        disc_vals.extend(node_df["disc"].dropna().tolist())
    if "disc" in df.columns:
        disc_vals.extend(df["disc"].dropna().tolist())
    disc_arr = np.array([v for v in disc_vals if np.isfinite(v)], dtype=np.float64)
    vmax = float(np.nanmax(np.abs(disc_arr))) if disc_arr.size else 0.1
    vmax = max(vmax, 0.05)
    vmax_adj = vmax * 0.7
    norm_disc = TwoSlopeNorm(vmin=-vmax_adj, vcenter=0.0, vmax=vmax_adj)

    if len(df) > 0:
        cnt = df["count"].to_numpy(dtype=np.float64)
        w_lo, w_hi = quantile_scale(cnt, 0.10, 0.90, 1.0, 10.0)
    else:
        w_lo, w_hi = 1.0, 10.0

    for r in df.itertuples(index=False):
        lf, lt = int(r.layer_from), int(r.layer_to)
        cf, ct = int(r.cluster_from), int(r.cluster_to)
        if (lf, cf) not in positions or (lt, ct) not in positions:
            continue
        p1, p2 = positions[(lf, cf)], positions[(lt, ct)]
        lw = scale_linear(float(r.count), w_lo, w_hi, 0.5, 3.5)
        alpha = scale_linear(float(r.count), w_lo, w_hi, 0.23, 0.57)
        alpha = min(0.85, alpha)
        d = float(getattr(r, "disc", 0.0))
        d_c = np.clip(d, -vmax_adj, vmax_adj)
        color = CMAP_DISC(norm_disc(d_c))
        if cf == ct:
            lw *= 1.10
            alpha = min(0.90, alpha + 0.10)
        add_curved_edge(ax, p1, p2, color=color, lw=lw, alpha=alpha)

    sizes = node_df["count"].to_numpy(dtype=np.float64)
    s_t = np.sqrt(sizes)
    s_lo, s_hi = quantile_scale(s_t, 0.05, 0.95, 1.0, 10.0)
    radii = scale_linear_vec(s_t, s_lo, s_hi, min_r, max_r)
    r_lookup = {}
    for idx, (_, row) in enumerate(node_df.iterrows()):
        r_lookup[(int(row["layer"]), int(row["cluster"]))] = float(radii[idx])

    for r in node_df.itertuples(index=False):
        layer, cluster = int(r.layer), int(r.cluster)
        if (layer, cluster) not in positions:
            continue
        x, y = positions[(layer, cluster)]
        radius = r_lookup.get((layer, cluster), min_r)
        d = float(getattr(r, "disc", 0.0))
        d_c = np.clip(d, -vmax_adj, vmax_adj)
        face = CMAP_DISC(norm_disc(d_c))
        circ = plt.Circle((x, y), radius, facecolor=face, edgecolor="black", lw=0.6, alpha=0.95, zorder=3)
        ax.add_patch(circ)
        if len(node_df) <= 500:
            tc = "white" if abs(d) > vmax_adj * 0.4 else "black"
            ax.text(x, y, str(cluster), ha="center", va="center", fontsize=6.0, color=tc, fontweight="medium", zorder=4)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Layer", fontsize=10)
    xt = np.arange(n_layers)
    ax.set_xticks(xt)
    if n_layers <= 20:
        ax.set_xticklabels([f"L{int(l)}" for l in layers], fontsize=8)
    else:
        ax.set_xticklabels([f"{int(l)}" for l in layers], rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle=":", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        ax.set_xlim(min(xs) - max_r - 0.15, max(xs) + max_r + 0.15)
        ax.set_ylim(min(ys) - max_r - 0.15, max(ys) + max_r + 0.15)

    sm = mpl.cm.ScalarMappable(cmap=CMAP_DISC, norm=norm_disc)
    sm.set_array([])
    cbar = add_colorbar(fig, ax, sm, "P(correct|cluster) - P(correct)")
    cbar.ax.axhline(0.0, color="black", lw=1.5, linestyle="-", alpha=0.85)
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig