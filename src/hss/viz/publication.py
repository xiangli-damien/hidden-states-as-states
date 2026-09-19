"""Paper-oriented plotting of saved measurements; no fitting or data selection.

Layout references: LMD cluster/plotting_pub.py and refracluster/experiments/
clusterlens_paper_figures.py, 02_sta.py, 01_main_figure_stability.py. Numerical
inputs come exclusively from current HSS trials, never the reference notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .style import METHOD_COLORS, METHOD_NAMES, STYLE, publication_settings

ENTROPY = colors.LinearSegmentedColormap.from_list(
    "hss_entropy", ["#fff3b0", "#ffcb69", "#e8871e", "#c9462a", "#4a1a2a"]
)
CORRECTNESS = colors.LinearSegmentedColormap.from_list(
    "hss_correctness", ["#1e4a7a", "#80afd4", "#faf8f4", "#dfa19a", "#8b1a1a"]
)


def _grid(ax):
    ax.set_axisbelow(True)
    ax.grid(alpha=0.18, linewidth=0.5, linestyle=":")


def _panel(ax, title):
    ax.set_title(title, loc="left", pad=11)


def graph_display(nodes, edges):
    """Exportable display geometry and mask; no state or transition is relabeled."""
    cfg = publication_settings()
    nd, ed = nodes.copy(), edges.copy()
    nd["plot_x"] = nd.position * cfg["layer_spacing"]
    nd["plot_y"] = nd.y * cfg["state_spacing"]
    # Radius squared is affine in count. A small floor keeps every ID readable.
    nd["plot_radius"] = np.sqrt(
        cfg["min_radius"] ** 2
        + (cfg["max_radius"] ** 2 - cfg["min_radius"] ** 2)
        * nd.frequency
        / nd.frequency.max()
    )
    outgoing = ed.groupby(["from_layer", "from_state"]).size()

    def normalized(r):
        k = int(outgoing.get((r.layer, r.state), 0))
        return np.nan if k == 0 else (float(r.entropy / np.log2(k)) if k > 1 else 0.0)

    nd["entropy_normalized"] = [normalized(r) for r in nd.itertuples()]
    ed["displayed"] = (ed["count"] >= cfg["edge_min_count"]) & (
        ed.probability >= cfg["edge_min_probability"]
    )
    if cfg["edge_max_per_source"]:
        keep = (
            ed[ed.displayed]
            .sort_values(["count", "to_state"], ascending=[False, True])
            .groupby(["from_layer", "from_state"], sort=False)
            .head(cfg["edge_max_per_source"])
            .index
        )
        ed["displayed"] = ed.index.isin(keep)
    return nd, ed


def state_graph(nodes, edges, color="entropy", title="State map"):
    cfg = publication_settings()
    nd, ed = graph_display(nodes, edges)
    shown = ed[ed.displayed]
    max_k = nd.groupby("layer").size().max()
    width = cfg["graph_width"]
    height = max(
        3.4,
        min(
            9.5,
            width
            * ((max_k - 1) * cfg["state_spacing"] + 1.1)
            / (max(1, nd.position.max()) * cfg["layer_spacing"] + 1.2)
            + 0.7,
        ),
    )
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0.035, right=0.89, bottom=0.15, top=0.91)
        positions = {
            (int(r.layer), int(r.state)): (r.plot_x, r.plot_y) for r in nd.itertuples()
        }
        if color == "entropy":
            values, cmap, norm = nd.entropy_normalized, ENTROPY, colors.Normalize(0, 1)
            label = "Normalized outgoing entropy"
        elif color in ("accuracy", "accuracy_delta"):
            values, cmap = nd[color], CORRECTNESS
            limit = cfg["correctness_limit"]
            norm = (
                colors.Normalize(0, 1)
                if color == "accuracy"
                else colors.TwoSlopeNorm(0, -limit, limit)
            )
            label = (
                r"$P(\mathrm{correct}\mid s_l)-P(\mathrm{correct})$"
                if color == "accuracy_delta"
                else r"$P(\mathrm{correct}\mid s_l)$"
            )
        elif color == "none":
            values, cmap, norm, label = (
                pd.Series(0.4, index=nd.index),
                plt.get_cmap("Blues"),
                colors.Normalize(0, 1),
                "",
            )
        else:
            raise ValueError(f"Unsupported state color: {color}")
        maximum = max(1, shown["count"].max()) if len(shown) else 1
        for r in shown.sort_values("count").itertuples():
            x0, y0 = positions[(r.from_layer, r.from_state)]
            x1, y1 = positions[(r.to_layer, r.to_state)]
            t = np.sqrt(r.count / maximum)
            edge_color = "#4a4a4a"
            if (
                color in ("accuracy", "accuracy_delta")
                and hasattr(r, color)
                and np.isfinite(getattr(r, color))
            ):
                edge_color = cmap(norm(getattr(r, color)))
            path = Path(
                [
                    (x0, y0),
                    (x0 + cfg["curve"] * (x1 - x0), y0),
                    (x1 - cfg["curve"] * (x1 - x0), y1),
                    (x1, y1),
                ],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
            )
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor="none",
                    edgecolor=edge_color,
                    lw=cfg["edge_min_width"]
                    + t * (cfg["edge_max_width"] - cfg["edge_min_width"]),
                    alpha=cfg["edge_min_alpha"]
                    + t * (cfg["edge_max_alpha"] - cfg["edge_min_alpha"]),
                    zorder=1,
                )
            )
        for (_, r), value in zip(nd.iterrows(), values):
            face = (
                cmap(norm(value)) if np.isfinite(value) else colors.to_rgba("#d7d7d7")
            )
            ax.add_patch(
                Circle(
                    (r.plot_x, r.plot_y),
                    r.plot_radius,
                    facecolor=face,
                    edgecolor="#292929",
                    linewidth=0.55,
                    zorder=3,
                )
            )
            luminance = np.dot(face[:3], [0.2126, 0.7152, 0.0722])
            ax.text(
                r.plot_x,
                r.plot_y,
                str(int(r.state)),
                ha="center",
                va="center",
                fontsize=cfg["node_fontsize"],
                color="white" if luminance < 0.36 else "#202020",
                zorder=4,
            )
        ticks = nd[["layer", "plot_x"]].drop_duplicates()
        ax.set_xticks(ticks.plot_x, [f"L{v}" for v in ticks.layer])
        ax.set(
            xlabel="Layer",
            yticks=[],
            xlim=(nd.plot_x.min() - 0.55, nd.plot_x.max() + 0.55),
            ylim=(nd.plot_y.min() - 0.5, nd.plot_y.max() + 0.5),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.22, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        if color != "none":
            cax = fig.add_axes([0.915, 0.25, 0.014, 0.54])
            finite = np.asarray(values)[np.isfinite(values)]
            extend = (
                "both"
                if len(finite)
                and (finite.min() < norm.vmin or finite.max() > norm.vmax)
                else "neither"
            )
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax,
                extend=extend,
                label=label,
            )
        coverage = float(shown["count"].sum() / ed["count"].sum()) if len(ed) else 1.0
        caption = f"All {len(nd)} nodes shown; IDs are global within this trial. Displayed edges retain {coverage:.1%} of transition mass."
        if color == "entropy":
            caption += "  Gray: terminal layer."
        fig.text(0.035, 0.032, caption, fontsize=8, color="#444444")
        fig._hss_encoding = {
            "settings": cfg,
            "node_size": "radius squared affine in frequency; floor for readable IDs",
            "entropy": "bits / log2(number of observed outgoing destinations); terminal undefined",
            "color": color,
            "color_limits": [float(norm.vmin), float(norm.vmax)],
            "displayed_edges": len(shown),
            "all_edges": len(ed),
            "transition_mass_retained": coverage,
        }
        return fig


def geometry(marginal, profile, similarity):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(
            1, 3, figsize=(13, 3.9), layout="constrained", gridspec_kw={"wspace": 0.23}
        )
        ax = axs[0]
        ax.bar(marginal["rank"], marginal.frequency, color="#77a8d8", width=0.8)
        ax.set(
            xlabel="Global state rank",
            ylabel="Frequency",
            xlim=(0.1, len(marginal) + 0.9),
        )
        ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
        _grid(ax)
        _panel(ax, "(a) Marginal state frequency")
        twin = ax.twinx()
        twin.plot(
            marginal["rank"],
            marginal.cumulative_mass,
            ".-",
            color="#df5149",
            markersize=3,
        )
        twin.set(ylim=(0, 1.03), ylabel="Cumulative mass")
        twin.tick_params(axis="y", colors="#cf443e")
        twin.yaxis.label.set_color("#cf443e")
        im = axs[1].imshow(
            similarity,
            vmin=0,
            vmax=1,
            cmap="Blues",
            interpolation="nearest",
            rasterized=True,
        )
        axs[1].set(xlabel="Samples", ylabel="Samples", xticks=[], yticks=[])
        _panel(axs[1], "(b) Trajectory similarity")
        fig.colorbar(
            im, ax=axs[1], fraction=0.046, pad=0.03, label="Hamming similarity"
        )
        ax = axs[2]
        bars = ax.twinx()
        bars.bar(
            profile.layer, profile.active_k, color="#eeb889", alpha=0.48, width=0.82
        )
        bars.set(ylabel="Active states", ylim=(0, max(profile.active_k) * 1.15))
        bars.yaxis.set_major_locator(MaxNLocator(integer=True))
        bars.tick_params(axis="y", colors="#c96b25")
        bars.yaxis.label.set_color("#c96b25")
        ax.set_zorder(bars.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(
            profile.layer,
            profile.self_transition,
            ".-",
            color="#458aca",
            label="Self-transition",
            markersize=4,
        )
        ax.plot(
            profile.layer,
            profile.uniform_self_transition,
            "--",
            color="#555555",
            lw=0.9,
            label="Uniform 1/K",
        )
        ax.set(xlabel="Layer", ylabel="P(stay)", ylim=(0, 1.04))
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.tick_params(axis="y", colors="#458aca")
        ax.yaxis.label.set_color("#458aca")
        _grid(ax)
        _panel(ax, "(c) Layer-wise dynamics")
        ax.legend(loc="lower left", fontsize=7)
        fig._hss_encoding = {
            "similarity_rows": len(similarity),
            "similarity": "Hamming, stored hierarchical order",
            "all_states_in_frequency": True,
        }
        return fig


def icl_surface(scan, criterion="Model-selection criterion", tolerance=None):
    surface = scan.pivot(index="layer", columns="k", values="relative_criterion")
    matrix = surface.to_numpy()
    finite = matrix[np.isfinite(matrix)]
    vmax = max(float(np.quantile(finite, 0.95)), 0.05) if len(finite) else 0.05
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap=plt.get_cmap("magma_r").with_extremes(bad="#d9d9d9"),
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        xs = {k: i for i, k in enumerate(surface.columns)}
        ys = {v: i for i, v in enumerate(surface.index)}
        selected = scan[scan.selected].sort_values("layer")
        ax.plot(selected.k.map(xs), selected.layer.map(ys), color="white", lw=3.3)
        ax.plot(
            selected.k.map(xs),
            selected.layer.map(ys),
            "o-",
            color="#262626",
            lw=1.1,
            ms=3,
            label="Selected K",
        )
        if (
            tolerance is not None
            and len(finite)
            and min(matrix.shape) > 1
            and finite.min() < tolerance < finite.max()
        ):
            ax.contour(matrix, levels=[tolerance], colors=["#3fbfc5"], linewidths=1.0)
            ax.plot([], [], color="#3fbfc5", label=f"Tolerance {tolerance:.0%}")
        for values, setter in [
            (surface.columns, ax.set_xticks),
            (surface.index, ax.set_yticks),
        ]:
            ix = np.unique(
                np.linspace(0, len(values) - 1, min(12, len(values))).astype(int)
            )
            setter(ix, values[ix])
        ax.set(xlabel="Candidate K", ylabel="Layer")
        _panel(ax, f"Relative {criterion}")
        ax.legend(loc="upper right", facecolor="white", frameon=True, framealpha=0.92)
        fig.colorbar(
            im,
            ax=ax,
            pad=0.025,
            fraction=0.035,
            extend="max" if len(finite) and finite.max() > vmax else "neither",
            label="(criterion - layer minimum) / max(|minimum|, 1)",
        )
        fig._hss_encoding = {
            "color_vmax": vmax,
            "upper_color_quantile": 0.95,
            "full_values_in_csv": True,
            "tolerance": tolerance,
        }
        return fig


def associations(frame):
    palette = {
        "category": ("Problem type", "#d64843", "o"),
        "correctness": ("Correctness", "#2b9360", "^"),
        "label": ("Correctness", "#2b9360", "^"),
        "level": ("Difficulty", "#eb8b2d", "s"),
    }
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
        for axis, part in frame.groupby("axis", sort=True):
            part = part.dropna(subset=["cramers_v"]).sort_values("layer")
            if len(part):
                label, color, marker = palette.get(axis, (axis, "#577ba1", "o"))
                ax.plot(
                    part.layer,
                    part.cramers_v,
                    label=label,
                    color=color,
                    marker=marker,
                    ms=3,
                    lw=1.2,
                )
        ax.set(xlabel="Layer", ylabel="Cramér's V", ylim=(0, 1))
        ax.xaxis.set_major_locator(MaxNLocator(9, integer=True))
        _grid(ax)
        if ax.lines:
            ax.legend(loc="upper left")
        return fig


def correctness_bands(tags):
    frame = tags.pivot(index="layer", columns="state", values="accuracy_delta")
    limit = publication_settings()["correctness_limit"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
        im = ax.imshow(
            frame,
            cmap=CORRECTNESS.with_extremes(bad="white"),
            vmin=-limit,
            vmax=limit,
            aspect="auto",
            interpolation="nearest",
        )
        stride = max(1, int(np.ceil(len(frame.columns) / 30)))
        ax.set_xticks(
            range(0, len(frame.columns), stride), frame.columns[::stride], rotation=45
        )
        ix = np.unique(np.linspace(0, len(frame) - 1, min(17, len(frame))).astype(int))
        ax.set_yticks(ix, frame.index[ix])
        for x, state in enumerate(frame.columns):
            active = np.flatnonzero(frame[state].notna().to_numpy())
            if len(active):
                ax.scatter(
                    [x],
                    [active[0] - 0.16],
                    marker="v",
                    s=24,
                    c="#278c50",
                    edgecolors="white",
                    linewidths=0.6,
                )
                ax.scatter(
                    [x],
                    [active[-1] + 0.16],
                    marker="^",
                    s=24,
                    c="#c62b36",
                    edgecolors="white",
                    linewidths=0.6,
                )
        ax.set(xlabel="Global state ID", ylabel="Layer")
        handles = [
            Line2D([], [], ls="", marker=m, color=c, label=t)
            for m, c, t in [
                ("v", "#278c50", "First appearance"),
                ("^", "#c62b36", "Last appearance"),
            ]
        ]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
        finite = frame.to_numpy()[np.isfinite(frame.to_numpy())]
        fig.colorbar(
            im,
            ax=ax,
            pad=0.025,
            fraction=0.03,
            label=r"$P(\mathrm{correct}\mid s_l)-P(\mathrm{correct})$",
            extend="both"
            if len(finite) and np.abs(finite).max() > limit
            else "neither",
        )
        fig._hss_encoding = {
            "missing_state": "white; not zero correctness deviation",
            "color_limits": [-limit, limit],
        }
        return fig


def refit_summary(comparisons, baselines):
    """Compare each refit only with its declared method baseline, not all pairs."""
    parts = []
    for method, baseline in baselines.items():
        rows = comparisons[
            (comparisons.method == method)
            & ((comparisons.reference == baseline) | (comparisons.target == baseline))
        ].copy()
        if rows.empty:
            continue
        forward = rows.reference.eq(baseline)
        rows["refit_seed"] = np.where(forward, rows.target_seed, rows.reference_seed)
        rows["refit_fraction"] = np.where(
            forward, rows.target_fit_fraction, rows.reference_fit_fraction
        )
        rows["perturbation"] = np.where(rows.refit_fraction < 1, "Subsample", "Seed")
        parts.append(rows)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stability_summary(frame):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2), layout="constrained")
        for col, kind in enumerate(["Seed", "Subsample"]):
            for row, (field, ylabel) in enumerate(
                [
                    ("ari", "Adjusted Rand index"),
                    ("centroid_cosine_distance", "Matched-center cosine distance"),
                ]
            ):
                ax = axs[row, col]
                part = frame[frame.perturbation == kind]
                for method, g in part.groupby("method", sort=False):
                    stats = g.groupby("layer")[field].agg(["mean", "min", "max"])
                    ax.plot(
                        stats.index,
                        stats["mean"],
                        color=METHOD_COLORS[method],
                        label=METHOD_NAMES[method],
                    )
                    ax.fill_between(
                        stats.index,
                        stats["min"],
                        stats["max"],
                        color=METHOD_COLORS[method],
                        alpha=0.12,
                        lw=0,
                    )
                ax.set(xlabel="Layer", ylabel=ylabel)
                if row == 0:
                    ax.set_ylim(0, 1.03)
                else:
                    ax.set_ylim(bottom=0)
                ax.xaxis.set_major_locator(MaxNLocator(9, integer=True))
                _grid(ax)
                values = (
                    sorted(part.refit_seed.unique())
                    if kind == "Seed"
                    else sorted(part.refit_fraction.unique())
                )
                subtitle = (
                    ("Seeds " + ", ".join(str(int(x)) for x in values))
                    if kind == "Seed"
                    else ("Fit fractions " + ", ".join(f"{x:.0%}" for x in values))
                )
                _panel(
                    ax,
                    f"({chr(97 + row * 2 + col)}) {subtitle}",
                )
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4)
        fig.suptitle(
            "Fixed-K refits: mean and range across recorded perturbations", fontsize=12
        )
        fig._hss_encoding = {
            "comparison": "each refit versus method baseline",
            "band": "min-max across recorded perturbations, not a confidence interval",
            "k_selection_stability_tested": False,
        }
        return fig


def separation(diagnostics, comparisons):
    """GMM geometry and center reproducibility: supported portion of paper Fig. 5."""
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for kind, c in [("Seed", "#5089c4"), ("Subsample", "#23a89b")]:
            part = comparisons[
                (comparisons.method == "gmm") & (comparisons.perturbation == kind)
            ]
            if len(part):
                stats = part.groupby("layer").centroid_cosine_distance.agg(
                    ["mean", "min", "max"]
                )
                axs[0].plot(stats.index, stats["mean"], ".-", color=c, label=kind, ms=4)
                axs[0].fill_between(
                    stats.index, stats["min"], stats["max"], color=c, alpha=0.15, lw=0
                )
        if "uniform_assignment_centroid_distance" in diagnostics:
            axs[0].plot(
                diagnostics.layer,
                diagnostics.uniform_assignment_centroid_distance,
                "x--",
                color="#777777",
                ms=3,
                label="Uniform assignment",
            )
        for field, label, c in [
            ("between_cluster_symmetric_kl", "Between clusters", "#e76f51"),
            ("within_fit_symmetric_kl", "Fitted vs empirical", "#5b9ed8"),
        ]:
            if field in diagnostics:
                axs[1].plot(
                    diagnostics.layer,
                    diagnostics[field],
                    ".--",
                    color=c,
                    label=label,
                    ms=3,
                )
        axs[0].set(ylabel="Bipartite cosine distance", ylim=(0, None))
        axs[1].set(ylabel="Symmetric diagonal Gaussian KL", yscale="symlog")
        for ax, title in zip(
            axs, ["(a) State-center persistence", "(b) State separation"]
        ):
            ax.set_xlabel("Layer")
            _panel(ax, title)
            _grid(ax)
            ax.legend(fontsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(9, integer=True))
        return fig


def prediction_controls(metrics, intervals=None):
    frame = metrics.copy()
    method_col = "cluster_method" if "cluster_method" in frame else "method"
    if "average_precision" not in frame:
        # Generic callers may have only the paper's two metrics.
        metrics_list = [("auroc", "AUROC", 0.5), ("accuracy", "Accuracy", None)]
    else:
        metrics_list = [
            ("auroc", "AUROC", 0.5),
            (
                "average_precision",
                "Average precision",
                float(frame.positive_prevalence.iloc[0]),
            ),
        ]
    frame["display_name"] = [
        f"HSS-NB / {METHOD_NAMES.get(m, m)}" if p == "HSS-NB" else p
        for m, p in zip(frame[method_col], frame.predictor)
    ]
    frame = frame.sort_values(["predictor", method_col]).reset_index(drop=True)
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(10.5, max(3.7, 0.39 * len(frame) + 1)),
            layout="constrained",
            sharey=True,
        )
        for ax, (metric, label, baseline) in zip(axs, metrics_list):
            for j, r in frame.iterrows():
                c = (
                    METHOD_COLORS.get(r[method_col], "#777777")
                    if r.predictor == "HSS-NB"
                    else "#8a9299"
                )
                ax.scatter(r[metric], j, c=c, s=35, zorder=3)
                if intervals is not None and len(intervals):
                    ci = intervals[
                        (intervals.trial_id == r.trial_id)
                        & (intervals.predictor == r.predictor)
                        & (intervals.metric == metric)
                    ]
                    if len(ci):
                        ax.hlines(
                            j, ci.lower_95.iloc[0], ci.upper_95.iloc[0], color=c, lw=1.5
                        )
                ax.annotate(
                    f"{r[metric]:.3f}",
                    (r[metric], j),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
            if baseline is not None:
                ax.axvline(
                    baseline,
                    color="#777777",
                    ls="--",
                    lw=0.9,
                    label=f"Reference {baseline:.3f}",
                )
            ax.set(xlabel=label, xlim=(0, 1))
            _grid(ax)
            if baseline is not None:
                ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), fontsize=7)
        axs[0].set_yticks(range(len(frame)), frame.display_name)
        axs[0].invert_yaxis()
        fig.suptitle("Before-generation prediction on held-out responses", fontsize=12)
        fig._hss_encoding = {
            "intervals": "95% stratified test-response bootstrap; fixed fitted model"
            if intervals is not None
            else "none",
            "positive_class": "correct response",
            "selection": "no test-set model selection",
        }
        return fig


def monitoring_controls(metrics):
    part = metrics[metrics.predictor.eq("HSS-NB")].copy()
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
        labels = [METHOD_NAMES.get(m, m) for m in part.method]
        for ax, field, title in zip(
            axs,
            ["test_far", "early_detection_rate", "saved_token_fraction_failed"],
            [
                "(a) False alarm rate",
                "(b) Early failure detection",
                "(c) Saved tokens in failed responses",
            ],
        ):
            ax.bar(
                range(len(part)),
                part[field],
                color=[METHOD_COLORS[m] for m in part.method],
                width=0.6,
            )
            ax.set_xticks(range(len(part)), labels, rotation=25, ha="right")
            ax.set_ylim(0, max(0.15, float(part[field].max()) * 1.35))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            _grid(ax)
            _panel(ax, title)
            for j, v in enumerate(part[field]):
                ax.text(j, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
        axs[0].axhline(
            0.1, color="#555555", ls="--", lw=1, label="Validation target 10%"
        )
        axs[0].legend(fontsize=7)
        fig.suptitle(
            "Sentence-prefix monitoring: held-out responses, validation-only threshold",
            fontsize=12,
        )
        fig._hss_encoding = {
            "target_far": 0.1,
            "far_scope": "all boundaries",
            "early_detection": "before final boundary",
            "missing_methods": "not shown; no placeholder estimates",
        }
        return fig
