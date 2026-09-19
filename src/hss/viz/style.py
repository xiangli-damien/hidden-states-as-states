"""Render-only settings. These never participate in fitting or trial identity."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class PublicationStyle:
    graph_width: float = 13.6
    layer_spacing: float = 1.25
    state_spacing: float = 0.50
    min_radius: float = 0.105
    max_radius: float = 0.215
    node_fontsize: float = 6.5
    edge_min_count: int = 10
    edge_min_probability: float = 0.02
    edge_max_per_source: int = 5  # zero displays every eligible edge
    edge_min_width: float = 0.35
    edge_max_width: float = 2.8
    edge_min_alpha: float = 0.08
    edge_max_alpha: float = 0.44
    curve: float = 0.35
    correctness_limit: float = 0.75


def publication_settings():
    path = os.environ.get("HSS_FIGURE_STYLE")
    values = (
        tomllib.loads(Path(path).read_text()).get("publication", {}) if path else {}
    )
    cfg = PublicationStyle(**values)
    if not (cfg.graph_width > 0 and cfg.layer_spacing > 0 and cfg.state_spacing > 0):
        raise ValueError("Figure dimensions and spacing must be positive")
    if not (0 < cfg.min_radius <= cfg.max_radius < cfg.state_spacing / 2):
        raise ValueError(
            "Node radii must be positive and smaller than half the state spacing"
        )
    if not (
        0 <= cfg.edge_min_probability <= 1
        and cfg.edge_min_count >= 0
        and cfg.edge_max_per_source >= 0
    ):
        raise ValueError("Invalid display edge thresholds")
    if not (
        0 <= cfg.edge_min_alpha <= cfg.edge_max_alpha <= 1
        and 0 < cfg.edge_min_width <= cfg.edge_max_width
    ):
        raise ValueError("Invalid edge alpha or width")
    if not (
        0 <= cfg.curve <= 0.5
        and 0 < cfg.correctness_limit <= 1
        and cfg.node_fontsize > 0
    ):
        raise ValueError("Invalid curve, correctness range or label size")
    return asdict(cfg)


STYLE = {
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.4,
    "savefig.dpi": 240,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}

METHOD_NAMES = {
    "gmm": "GMM",
    "kmeans": "KMeans",
    "minibatch_kmeans": "MiniBatchKMeans",
    "mfa": "MFA",
}
METHOD_COLORS = {
    "gmm": "#245b82",
    "kmeans": "#d36625",
    "minibatch_kmeans": "#249477",
    "mfa": "#8561a3",
}
