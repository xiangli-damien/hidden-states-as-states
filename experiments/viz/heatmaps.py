from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .style import (
    COLORS, COLOR_BIRTH, COLOR_DEATH,
    CMAP_DIVERGING, CMAP_PROBABILITY, save_figure,
)


def plot_heatmap_diverging(
    data: np.ndarray,
    layers: np.ndarray,
    cluster_order: List[int],
    birth: Dict[int, int],
    death: Dict[int, int],
    cbar_label: str,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (7, 5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    sub = data[:, cluster_order]
    finite = sub[np.isfinite(sub)]
    if vmax is None:
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 0.1
        vmax = max(vmax, 1e-3)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sub, cmap=CMAP_DIVERGING, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xlabel("Global Cluster ID", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    nC = len(cluster_order)
    step = max(1, nC // 12)
    ax.set_xticks(list(range(0, nC, step)))
    ax.set_xticklabels([str(cluster_order[i]) for i in range(0, nC, step)], rotation=45, ha="right", fontsize=8)
    L = len(layers)
    ystep = max(1, L // 12)
    ax.set_yticks(list(range(0, L, ystep)))
    ax.set_yticklabels([str(int(layers[i])) for i in range(0, L, ystep)], fontsize=8)

    layer_to_row = {int(layers[i]): i for i in range(L)}
    has_b, has_d = False, False
    for xi, gid in enumerate(cluster_order):
        b = birth.get(int(gid), int(layers[0]))
        d = death.get(int(gid), int(layers[-1]))
        if b in layer_to_row and layer_to_row[b] > 0:
            ax.plot(xi, layer_to_row[b], marker="v", markersize=4, color=COLOR_BIRTH, markeredgecolor="white", markeredgewidth=0.4, alpha=0.9)
            has_b = True
        if d in layer_to_row and layer_to_row[d] < L - 1:
            ax.plot(xi, layer_to_row[d], marker="^", markersize=4, color=COLOR_DEATH, markeredgecolor="white", markeredgewidth=0.4, alpha=0.9)
            has_d = True

    if has_b or has_d:
        elems = []
        if has_b:
            elems.append(Line2D([0], [0], marker="v", color="w", markerfacecolor=COLOR_BIRTH, markersize=6, label="Birth", linestyle="None"))
        if has_d:
            elems.append(Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_DEATH, markersize=6, label="Death", linestyle="None"))
        ax.legend(handles=elems, loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray", fontsize=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_heatmap_probability(
    data: np.ndarray,
    layers: np.ndarray,
    cluster_order: List[int],
    birth: Dict[int, int],
    death: Dict[int, int],
    baseline: float,
    cbar_label: str,
    figsize: Tuple[float, float] = (7, 5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    sub = data[:, cluster_order]
    norm = Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sub, cmap=CMAP_PROBABILITY, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xlabel("Global Cluster ID", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    nC = len(cluster_order)
    step = max(1, nC // 12)
    ax.set_xticks(list(range(0, nC, step)))
    ax.set_xticklabels([str(cluster_order[i]) for i in range(0, nC, step)], rotation=45, ha="right", fontsize=8)
    L = len(layers)
    ystep = max(1, L // 12)
    ax.set_yticks(list(range(0, L, ystep)))
    ax.set_yticklabels([str(int(layers[i])) for i in range(0, L, ystep)], fontsize=8)

    layer_to_row = {int(layers[i]): i for i in range(L)}
    for xi, gid in enumerate(cluster_order):
        b = birth.get(int(gid), int(layers[0]))
        d = death.get(int(gid), int(layers[-1]))
        if b in layer_to_row and layer_to_row[b] > 0:
            ax.plot(xi, layer_to_row[b], marker="v", markersize=4, color=COLOR_BIRTH, markeredgecolor="white", markeredgewidth=0.4)
        if d in layer_to_row and layer_to_row[d] < L - 1:
            ax.plot(xi, layer_to_row[d], marker="^", markersize=4, color=COLOR_DEATH, markeredgecolor="white", markeredgewidth=0.4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.plot([0, 1], [baseline, baseline], color="black", lw=2.0, transform=cbar.ax.get_yaxis_transform(), clip_on=False)
    cbar.ax.text(0.5, -0.08, f"baseline={baseline:.0%}", transform=cbar.ax.transAxes, fontsize=7, ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="gray", alpha=0.9))
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_dual_axis_information(
    df_info: pd.DataFrame,
    figsize: Tuple[float, float] = (9, 4.5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=figsize)
    layer = df_info["layer"].to_numpy()
    H_C = df_info["H_C"].to_numpy()
    I_CY = df_info["I_CY"].to_numpy()
    HY = float(df_info["H_Y"].iloc[0]) if df_info["H_Y"].notna().any() else 1.0

    c_comp = COLORS["blue"]
    c_pred = COLORS["red"]

    ax1.set_xlabel("Layer", fontsize=10)
    ax1.set_ylabel(r"$H(C_\ell)$ (bits)", color=c_comp, fontsize=10)
    l1, = ax1.plot(layer, H_C, marker="o", markersize=5, color=c_comp, lw=1.5, label=r"$H(C_\ell)$")
    ax1.tick_params(axis="y", labelcolor=c_comp)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$I(C_\ell; Y)$ (bits)", color=c_pred, fontsize=10)
    l2, = ax2.plot(layer, I_CY, marker="s", markersize=5, color=c_pred, lw=1.5, label=r"$I(C_\ell; Y)$")
    ax2.tick_params(axis="y", labelcolor=c_pred)

    i_valid = I_CY[np.isfinite(I_CY)]
    if i_valid.size > 0 and float(np.max(i_valid)) > HY * 0.5:
        ax2.axhline(HY, color=c_pred, linestyle="--", alpha=0.5, lw=1.0)
        ax2.text(layer[-1], HY * 1.02, f"H(Y)={HY:.3f}", ha="right", va="bottom", color=c_pred, fontsize=8, alpha=0.7)

    ax1.grid(True, alpha=0.25, axis="both", linestyle=":")
    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center", ncol=2, frameon=True, fontsize=8, framealpha=0.9)
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_information_bottleneck(
    df_info: pd.DataFrame,
    figsize: Tuple[float, float] = (5.5, 5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    x = df_info["H_C"].to_numpy(dtype=np.float64)
    yv = df_info["I_CY"].to_numpy(dtype=np.float64)
    l = df_info["layer"].to_numpy(dtype=np.int32)
    m = np.isfinite(x) & np.isfinite(yv)
    x, yv, l = x[m], yv[m], l[m]

    if x.size == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return fig

    norm = Normalize(vmin=float(l.min()), vmax=float(l.max()))
    sc = ax.scatter(x, yv, c=l, cmap="viridis", norm=norm, s=50, edgecolors="black", linewidths=0.4, alpha=0.95, zorder=3)
    ax.plot(x, yv, lw=1.0, alpha=0.5, color="gray", zorder=1)

    for i in range(len(x)):
        if i % max(1, len(x) // 5) == 0:
            ax.annotate(f"L{l[i]}", (x[i], yv[i]), textcoords="offset points", xytext=(5, 3), fontsize=7, alpha=0.8)

    ax.set_xlabel(r"$H(C_\ell)$ (bits)", fontsize=10)
    ax.set_ylabel(r"$I(C_\ell; Y)$ (bits)", fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Layer", fontsize=9)
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_effective_rank(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (8, 4),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    fracs = sorted(df["frac"].unique())
    cmap = plt.cm.viridis
    for i, frac in enumerate(fracs):
        sub = df[df["frac"] == frac]
        agg = sub.groupby("layer")["effective_rank"].agg(["mean", "std"]).reset_index()
        color = cmap(i / max(1, len(fracs) - 1))
        n_sub = sub["n_samples"].iloc[0] if len(sub) > 0 else 0
        ax.plot(agg["layer"], agg["mean"], marker="o", markersize=3, color=color, lw=1.2, label=f"{frac:.0%} (n={n_sub})")
        ax.fill_between(agg["layer"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.15, color=color)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Effective Rank (90%)", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=":")
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_rel_icl_surface(
    rel_mat: np.ndarray,
    layers: np.ndarray,
    k_values: np.ndarray,
    k_sel_by_layer: Optional[Dict[int, int]] = None,
    rel_pct: float = 0.02,
    figsize: Tuple[float, float] = (10, 5),
    out_path: Optional[Path] = None,
) -> plt.Figure:
    L, K = rel_mat.shape
    fig, ax = plt.subplots(figsize=figsize)

    finite = rel_mat[np.isfinite(rel_mat)]
    vmax = float(np.quantile(finite, 0.95)) if finite.size else 0.2
    vmax = max(vmax, rel_pct * 2.0, 0.05)

    im = ax.imshow(rel_mat, aspect="auto", interpolation="nearest", cmap="magma_r", vmin=0.0, vmax=vmax, origin="lower")
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)

    k_step = max(1, K // 10)
    ax.set_xticks(list(range(0, K, k_step)))
    ax.set_xticklabels([str(k_values[j]) for j in range(0, K, k_step)], fontsize=8)
    l_step = max(1, L // 10)
    ax.set_yticks(list(range(0, L, l_step)))
    ax.set_yticklabels([str(int(layers[i])) for i in range(0, L, l_step)], fontsize=8)

    if k_sel_by_layer:
        k_to_j = {int(k): j for j, k in enumerate(k_values)}
        xs, ys = [], []
        for i, layer in enumerate(layers):
            ks = k_sel_by_layer.get(int(layer))
            if ks is None:
                continue
            j = k_to_j.get(int(ks))
            if j is None:
                continue
            xs.append(j)
            ys.append(i)
        if xs:
            ax.plot(xs, ys, color="white", lw=2.5, alpha=0.95, zorder=5)
            ax.plot(xs, ys, color="black", lw=1.0, alpha=0.85, zorder=6)

    try:
        X, Y = np.meshgrid(np.arange(K), np.arange(L))
        cs = ax.contour(X, Y, rel_mat, levels=[rel_pct], colors=["cyan"], linewidths=1.2, alpha=0.9)
        ax.clabel(cs, inline=True, fontsize=7, fmt={rel_pct: f"tau={rel_pct:.2%}"})
    except Exception:
        pass

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("rel-ICL", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    if out_path:
        save_figure(fig, out_path)
    return fig