from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from .style import (
    CMAP_STABILITY, CMAP_PROB, CMAP_DISC,
    COLOR_EDGE, text_color_for_stability, save_figure,
)
from .utils import (
    add_curved_edge, quantile_scale, scale_linear, scale_linear_vec,
    compute_positions, adaptive_figsize, adaptive_radius, filter_edges,
    add_colorbar,
)


def _draw_edges(ax, df, positions, w_lo, w_hi, color_fn):
    for r in df.itertuples(index=False):
        lf, lt = int(r.layer_from), int(r.layer_to)
        cf, ct = int(r.cluster_from), int(r.cluster_to)
        if (lf, cf) not in positions or (lt, ct) not in positions:
            continue
        p1, p2 = positions[(lf, cf)], positions[(lt, ct)]
        lw = scale_linear(float(r.count), w_lo, w_hi, 0.5, 3.5)
        alpha = scale_linear(float(r.count), w_lo, w_hi, 0.08, 0.32)
        color = color_fn(r)
        if cf == ct:
            lw *= 1.10
            alpha = min(0.40, alpha + 0.06)
        add_curved_edge(ax, p1, p2, color=color, lw=lw, alpha=alpha)


def _draw_nodes(ax, node_df, positions, r_lookup, min_r, val_fn, cmap, norm, text_fn):
    for r in node_df.itertuples(index=False):
        layer, cluster = int(r.layer), int(r.cluster)
        if (layer, cluster) not in positions:
            continue
        x, y = positions[(layer, cluster)]
        radius = r_lookup.get((layer, cluster), min_r)
        val = val_fn(r)
        face = cmap(norm(val)) if np.isfinite(val) else (0.85, 0.85, 0.85, 1.0)
        circ = plt.Circle((x, y), radius, facecolor=face, edgecolor="black",
                           lw=0.5, alpha=0.95, zorder=3)
        ax.add_patch(circ)
        if len(node_df) <= 400:
            tc = text_fn(val)
            ax.text(x, y, str(cluster), ha="center", va="center",
                    fontsize=6.5, color=tc, fontweight="medium", zorder=4)


def _setup_axes(ax, layers, positions, max_r):
    n_layers = len(layers)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_xticks(np.arange(n_layers))
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
        pad = max_r + 0.15
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)


def _build_radii(node_df, min_r, max_r):
    sizes = node_df["count"].to_numpy(dtype=np.float64)
    s_t = np.sqrt(sizes)
    s_lo, s_hi = quantile_scale(s_t, 0.05, 0.95, 1.0, 10.0)
    radii = scale_linear_vec(s_t, s_lo, s_hi, min_r, max_r)
    r_lookup = {}
    for idx, (_, row) in enumerate(node_df.iterrows()):
        r_lookup[(int(row["layer"]), int(row["cluster"]))] = float(radii[idx])
    return r_lookup


def plot_dynamics_unlabeled(
    node_df: pd.DataFrame, flow_df: pd.DataFrame,
    layers: np.ndarray, N_total: int,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    positions = compute_positions(node_df, layers)
    n_layers = len(layers)
    max_clusters = int(node_df.groupby("layer").size().max())
    figsize = adaptive_figsize(n_layers, max_clusters)
    min_r, max_r = adaptive_radius(n_layers, max_clusters)

    fig, ax = plt.subplots(figsize=figsize)
    df = filter_edges(flow_df)
    w_lo, w_hi = (quantile_scale(df["count"].to_numpy(dtype=np.float64), 0.10, 0.90, 1.0, 10.0)
                  if len(df) > 0 else (1.0, 10.0))
    _draw_edges(ax, df, positions, w_lo, w_hi, lambda r: COLOR_EDGE)

    r_lookup = _build_radii(node_df, min_r, max_r)
    ent = node_df["out_entropy"].to_numpy(dtype=np.float64) if "out_entropy" in node_df.columns else np.zeros(len(node_df))
    ent_v = ent[np.isfinite(ent)]
    vmin = float(np.nanmin(ent_v)) if ent_v.size else 0.0
    vmax = float(np.nanmax(ent_v)) if ent_v.size else 1.0
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))

    _draw_nodes(
        ax, node_df, positions, r_lookup, min_r,
        lambda r: float(getattr(r, "out_entropy", np.nan)),
        CMAP_STABILITY, norm,
        lambda v: text_color_for_stability(v, vmin, vmax) if np.isfinite(v) else "black",
    )

    _setup_axes(ax, layers, positions, max_r)
    sm = mpl.cm.ScalarMappable(cmap=CMAP_STABILITY, norm=norm)
    sm.set_array([])
    add_colorbar(fig, ax, sm, "Transition Entropy")
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_dynamics_absolute(
    node_df: pd.DataFrame, flow_df: pd.DataFrame,
    layers: np.ndarray, N_total: int, baseline: float,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    positions = compute_positions(node_df, layers)
    n_layers = len(layers)
    max_clusters = int(node_df.groupby("layer").size().max())
    figsize = adaptive_figsize(n_layers, max_clusters)
    min_r, max_r = adaptive_radius(n_layers, max_clusters)
    norm_prob = Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=figsize)
    df = filter_edges(flow_df)
    w_lo, w_hi = (quantile_scale(df["count"].to_numpy(dtype=np.float64), 0.10, 0.90, 1.0, 10.0)
                  if len(df) > 0 else (1.0, 10.0))

    for r in df.itertuples(index=False):
        lf, lt = int(r.layer_from), int(r.layer_to)
        cf, ct = int(r.cluster_from), int(r.cluster_to)
        if (lf, cf) not in positions or (lt, ct) not in positions:
            continue
        p1, p2 = positions[(lf, cf)], positions[(lt, ct)]
        lw = scale_linear(float(r.count), w_lo, w_hi, 0.5, 3.5)
        alpha = min(0.85, scale_linear(float(r.count), w_lo, w_hi, 0.23, 0.57))
        pr = float(getattr(r, "pos_rate", baseline))
        if cf == ct:
            lw *= 1.10
            alpha = min(0.90, alpha + 0.10)
        add_curved_edge(ax, p1, p2, color=CMAP_PROB(norm_prob(pr)), lw=lw, alpha=alpha)

    r_lookup = _build_radii(node_df, min_r, max_r)
    _draw_nodes(
        ax, node_df, positions, r_lookup, min_r,
        lambda r: float(getattr(r, "pos_rate", baseline)),
        CMAP_PROB, norm_prob,
        lambda v: "white" if (v < 0.30 or v > 0.70) else "black",
    )

    _setup_axes(ax, layers, positions, max_r)
    sm = mpl.cm.ScalarMappable(cmap=CMAP_PROB, norm=norm_prob)
    sm.set_array([])
    cbar = add_colorbar(fig, ax, sm, "P(correct | cluster)", pad=0.12)
    cbar.ax.plot([0, 1], [baseline, baseline], color="black", lw=2.0,
                 transform=cbar.ax.get_yaxis_transform(), clip_on=False)
    cbar.ax.text(0.5, -0.08, f"baseline={baseline:.0%}", transform=cbar.ax.transAxes,
                 fontsize=7, ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="gray", alpha=0.9))
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_dynamics_relative(
    node_df: pd.DataFrame, flow_df: pd.DataFrame,
    layers: np.ndarray, N_total: int, baseline: float,
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
    vmax = max(float(np.nanmax(np.abs(disc_arr))) if disc_arr.size else 0.1, 0.05)
    vmax_adj = vmax * 0.7
    norm_disc = TwoSlopeNorm(vmin=-vmax_adj, vcenter=0.0, vmax=vmax_adj)

    w_lo, w_hi = (quantile_scale(df["count"].to_numpy(dtype=np.float64), 0.10, 0.90, 1.0, 10.0)
                  if len(df) > 0 else (1.0, 10.0))

    for r in df.itertuples(index=False):
        lf, lt = int(r.layer_from), int(r.layer_to)
        cf, ct = int(r.cluster_from), int(r.cluster_to)
        if (lf, cf) not in positions or (lt, ct) not in positions:
            continue
        p1, p2 = positions[(lf, cf)], positions[(lt, ct)]
        lw = scale_linear(float(r.count), w_lo, w_hi, 0.5, 3.5)
        alpha = min(0.85, scale_linear(float(r.count), w_lo, w_hi, 0.23, 0.57))
        d = np.clip(float(getattr(r, "disc", 0.0)), -vmax_adj, vmax_adj)
        if cf == ct:
            lw *= 1.10
            alpha = min(0.90, alpha + 0.10)
        add_curved_edge(ax, p1, p2, color=CMAP_DISC(norm_disc(d)), lw=lw, alpha=alpha)

    r_lookup = _build_radii(node_df, min_r, max_r)
    _draw_nodes(
        ax, node_df, positions, r_lookup, min_r,
        lambda r: np.clip(float(getattr(r, "disc", 0.0)), -vmax_adj, vmax_adj),
        CMAP_DISC, norm_disc,
        lambda v: "white" if abs(v) > vmax_adj * 0.4 else "black",
    )

    _setup_axes(ax, layers, positions, max_r)
    sm = mpl.cm.ScalarMappable(cmap=CMAP_DISC, norm=norm_disc)
    sm.set_array([])
    cbar = add_colorbar(fig, ax, sm, "P(correct|cluster) - P(correct)")
    cbar.ax.axhline(0.0, color="black", lw=1.5, linestyle="-", alpha=0.85)
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig

