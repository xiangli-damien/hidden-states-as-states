from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_curved_edge(
    ax: plt.Axes, p1: Tuple[float, float], p2: Tuple[float, float],
    color, lw: float, alpha: float, curve: float = 0.35, zorder: int = 1,
):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    path_data = [
        (MplPath.MOVETO, (x1, y1)),
        (MplPath.CURVE4, (x1 + curve * dx, y1)),
        (MplPath.CURVE4, (x2 - curve * dx, y2)),
        (MplPath.CURVE4, (x2, y2)),
    ]
    codes, verts = zip(*path_data)
    patch = PathPatch(
        MplPath(verts, codes), facecolor="none", edgecolor=color,
        lw=lw, alpha=alpha, zorder=zorder, capstyle="round",
    )
    ax.add_patch(patch)


def quantile_scale(
    values: np.ndarray, qlo: float, qhi: float, vmin: float, vmax: float,
) -> Tuple[float, float]:
    if values.size == 0:
        return vmin, vmax
    lo, hi = np.quantile(values, [qlo, qhi])
    lo = max(float(lo), vmin)
    hi = max(float(hi), lo + 1e-9)
    return float(lo), float(hi)


def scale_linear(x: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return 0.5 * (out_lo + out_hi)
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return out_lo + t * (out_hi - out_lo)


def scale_linear_vec(
    x: np.ndarray, lo: float, hi: float, out_lo: float, out_hi: float,
) -> np.ndarray:
    if hi <= lo:
        return np.full_like(x, 0.5 * (out_lo + out_hi), dtype=np.float64)
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return out_lo + t * (out_hi - out_lo)


def compute_positions(
    node_df, layers: np.ndarray,
    x_spacing: float = 1.0, y_spacing: float = 0.75,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    clusters_by_layer: Dict[int, List[int]] = {}
    for layer, g in node_df.groupby("layer"):
        clusters_by_layer[int(layer)] = sorted(g["cluster"].astype(int).tolist())

    positions: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for li, layer in enumerate(int(l) for l in layers):
        clusters = clusters_by_layer.get(layer, [])
        n = len(clusters)
        if n == 0:
            continue
        y0 = -0.5 * (n - 1) * y_spacing
        for i, c in enumerate(sorted(clusters)):
            positions[(layer, c)] = (float(li * x_spacing), float(y0 + i * y_spacing))
    return positions


def adaptive_figsize(
    n_layers: int, max_clusters: int,
    ref_width: float = 14.0, ref_height: float = 6.0,
    ref_layers: int = 32, ref_clusters: int = 10,
) -> Tuple[float, float]:
    w = ref_width * (n_layers / ref_layers) ** 0.75 + 1.2
    h = ref_height * (max_clusters / ref_clusters) ** 0.70
    return (float(np.clip(w, 5.0, 16.0)), float(np.clip(h, 3.5, 9.0)))


def adaptive_radius(
    n_layers: int, max_clusters: int,
    min_r: float = 0.15, max_r: float = 0.38, y_spacing: float = 0.75,
) -> Tuple[float, float]:
    layer_f = min(1.0, n_layers / 20)
    cluster_f = min(1.0, max_clusters / 8)
    density = (layer_f + cluster_f) / 2
    scale = 0.7 + 0.3 * np.sqrt(density)
    mr = min_r * scale
    xr = min(max_r * scale, 0.40, y_spacing * 0.45)
    mr = min(mr, xr * 0.5)
    return (float(mr), float(xr))


def filter_edges(
    flow_df, min_count: int = 10, min_prob: float = 0.02, max_per_source: int = 5,
):
    df = flow_df.copy()
    if "prob_out" in df.columns and len(df) > 0:
        df = df[df["prob_out"] >= min_prob]
    if len(df) > 0:
        df = df[df["count"] >= min_count]
    if max_per_source > 0 and len(df) > 0:
        df = (
            df.sort_values("count", ascending=False)
            .groupby(["layer_from", "cluster_from"], as_index=False)
            .head(max_per_source)
        )
    return df


def add_colorbar(fig, ax, mappable, label, size="3%", pad=0.08):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return cbar

