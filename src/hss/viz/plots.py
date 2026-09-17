"""Scientific figure functions consuming tables/arrays; never fit or load data."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

STYLE = {
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 180,
    "svg.fonttype": "none",
    "axes.titlesize": 11,
    "legend.fontsize": 8,
}


def dataset_overview(cases):
    """Descriptive response quality; these outcomes are not predictor features."""
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
        category = cases.groupby("category").is_correct.agg(["count", "mean"])
        axs[0].barh(
            category.index.str.replace("_", " "), category["mean"], color="#426f9f"
        )
        axs[0].set(
            xlabel="Correct response fraction",
            title="Correctness by problem type",
            xlim=(0, 1),
        )
        for j, row in enumerate(category.itertuples()):
            axs[0].text(row.mean + 0.01, j, f"n={row.count}", va="center", fontsize=8)
        difficulty = cases.groupby("level").is_correct.mean()
        axs[1].plot(difficulty.index, difficulty.values, marker="o", color="#426f9f")
        axs[1].set(
            xlabel="MATH difficulty level",
            ylabel="Correct response fraction",
            ylim=(0, 1),
            title="Correctness by difficulty",
        )
        if "n_response_tokens" in cases:
            for label, color in [(False, "#be655b"), (True, "#40846f")]:
                values = cases.loc[cases.is_correct == label, "n_response_tokens"]
                axs[2].hist(
                    values,
                    bins=np.linspace(0, max(cases.n_response_tokens), 33),
                    alpha=0.55,
                    label="Correct" if label else "Incorrect",
                    color=color,
                )
            axs[2].legend()
        axs[2].set(
            xlabel="Generated tokens",
            ylabel="Responses",
            title="Observed response lengths",
        )
        fig.suptitle("Llama-3.2-1B-Instruct / MATH — captured response checks")
        return fig


def method_quality(table):
    """Shared descriptive geometry measures, displayed layer by layer."""
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
        fields = [
            ("silhouette", "Silhouette (higher is better)"),
            ("calinski_harabasz", "Calinski–Harabasz (higher is better)"),
            ("davies_bouldin", "Davies–Bouldin (lower is better)"),
            ("active_k", "Occupied clusters"),
            ("max_cluster_fraction", "Largest cluster fraction"),
            ("occupancy_entropy_bits", "Occupancy entropy (bits)"),
        ]
        for ax, (field, title) in zip(axs.flat, fields):
            for name, part in table.groupby("job", sort=True):
                part = part.sort_values("layer")
                ax.plot(part.layer, part[field], ".-", label=name, markersize=4)
            ax.set(xlabel="Layer", title=title)
        axs[1, 1].set_ylim(0, 1)
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2)
        fig.suptitle("Matched-K response-mean methods — descriptive geometry")
        return fig


def em_convergence(table, tolerance):
    """Observed likelihood trace; absence of a plateau must stay visible."""
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for layer, part in table.groupby("layer", sort=True):
            part = part.sort_values("iteration")
            axs[0].plot(
                part.iteration,
                part.log_likelihood - part.log_likelihood.iloc[0],
                label=f"L{layer}",
                linewidth=1,
            )
            delta = part.log_likelihood.diff().abs()
            axs[1].plot(part.iteration, delta.where(delta > 0), linewidth=1)
        axs[0].set(xlabel="EM iteration", ylabel="Mean log-likelihood gain")
        axs[1].set(
            xlabel="EM iteration", ylabel="Absolute change per EM step", yscale="log"
        )
        axs[1].axhline(
            tolerance, color="black", linestyle="--", label=f"Tolerance {tolerance:g}"
        )
        axs[1].legend()
        axs[0].legend(ncol=4, fontsize=7)
        fig.suptitle(
            "MFA saved initialization — inspect convergence before interpretation"
        )
        return fig


def state_graph(nodes, edges, color="entropy", title="State transitions"):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 5.5), layout="constrained")
        positions = {
            (int(r.layer), int(r.state)): (r.position, r.y) for r in nodes.itertuples()
        }
        segments = [
            [
                positions[(r.from_layer, r.from_state)],
                positions[(r.to_layer, r.to_state)],
            ]
            for r in edges.itertuples()
        ]
        if segments:
            rgba = np.tile([0.35, 0.39, 0.43, 1.0], (len(edges), 1))
            rgba[:, 3] = 0.03 + 0.65 * np.sqrt(
                edges.frequency.to_numpy() / max(edges.frequency.max(), 1e-10)
            )
            ax.add_collection(
                LineCollection(
                    segments, colors=rgba, linewidths=0.3 + 2 * edges.frequency
                )
            )
        label = {
            "entropy": "Outgoing transition entropy (bits)",
            "accuracy_delta": "Correctness deviation from global mean",
            "none": "State",
        }[color]
        values = nodes[color].to_numpy() if color != "none" else np.zeros(len(nodes))
        valid = np.isfinite(values)
        cmap = "coolwarm" if color == "accuracy_delta" else "viridis"
        limits = (
            {"vmin": -1, "vmax": 1}
            if color == "accuracy_delta"
            else {
                "vmin": 0,
                "vmax": max(1, float(np.nanmax(values))) if valid.any() else 1,
            }
        )
        points = ax.scatter(
            nodes.loc[valid, "position"],
            nodes.loc[valid, "y"],
            s=12 + 450 * nodes.loc[valid, "frequency"],
            c=values[valid],
            cmap=cmap,
            **limits,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )
        if (~valid).any():
            ax.scatter(
                nodes.loc[~valid, "position"],
                nodes.loc[~valid, "y"],
                s=12 + 450 * nodes.loc[~valid, "frequency"],
                c="#b3b9c0",
                zorder=3,
            )
        if len(nodes) <= 300:
            for r in nodes.itertuples():
                ax.annotate(
                    str(r.state),
                    (r.position, r.y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5,
                )
        ticks = nodes[["layer", "position"]].drop_duplicates()
        ax.set_xticks(ticks.position, ticks.layer)
        padding = max(1, (nodes.y.max() - nodes.y.min()) * 0.12)
        ax.set(
            xlabel="Layer",
            ylabel="States within each layer",
            title=title,
            ylim=(nodes.y.min() - padding, nodes.y.max() + padding),
            xlim=(-0.5, ticks.position.max() + 0.5),
        )
        ax.set_yticks([])
        if color != "none":
            fig.colorbar(points, ax=ax, label=label, shrink=0.75)
        return fig


def geometry(marginal, profile, similarity):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
        ax = axs[0]
        ax.bar(marginal["rank"], marginal.frequency, color="#426f9f", width=1)
        twin = ax.twinx()
        twin.plot(marginal["rank"], marginal.cumulative_mass, color="#b74542")
        twin.set(ylim=(0, 1.03), ylabel="Cumulative mass")
        ax.set(
            xlabel="Global state frequency rank",
            ylabel="Marginal frequency",
            title="(a) State occupancy",
        )
        im = axs[1].imshow(
            similarity,
            vmin=0,
            vmax=1,
            cmap="magma",
            interpolation="nearest",
            rasterized=True,
        )
        axs[1].set(
            xlabel="Reordered trajectory",
            ylabel="Reordered trajectory",
            title="(b) Hamming similarity",
        )
        fig.colorbar(im, ax=axs[1], shrink=0.7)
        axs[2].bar(
            profile.layer, profile.active_k, color="#a9bfd4", label="Active states"
        )
        twin = axs[2].twinx()
        twin.plot(
            profile.layer,
            profile.self_transition,
            color="#b74542",
            marker=".",
            label="Self-transition",
        )
        twin.plot(
            profile.layer,
            profile.uniform_self_transition,
            ":",
            color="black",
            label="Uniform 1/K",
        )
        twin.set(ylabel="Transition probability", ylim=(0, 1.03))
        twin.legend(loc="upper left")
        axs[2].set(xlabel="Layer", ylabel="Active states K", title="(c) Layer dynamics")
        return fig


def profile_plot(profile):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
        for ax, field, label in zip(
            axs,
            ["k", "self_transition"],
            ["Selected components K", "Self-transition probability"],
        ):
            ax.plot(profile.relative_depth, profile[field], "o-", markersize=3)
            ax.set(xlabel="Relative layer depth", ylabel=label)
        axs[1].set_ylim(0, 1)
        return fig


def icl_surface(scan):
    with plt.rc_context(STYLE):
        surface = scan.pivot(index="layer", columns="k", values="relative_criterion")
        fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
        im = ax.imshow(
            surface,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        xs = {k: i for i, k in enumerate(surface.columns)}
        ys = {layer: i for i, layer in enumerate(surface.index)}
        selected = scan[scan.selected]
        near = scan[scan.near_optimal]
        ax.scatter(
            near.k.map(xs),
            near.layer.map(ys),
            s=5,
            c="white",
            alpha=0.6,
            label="Within tolerance",
        )
        ax.scatter(
            selected.k.map(xs),
            selected.layer.map(ys),
            s=22,
            facecolors="none",
            edgecolors="#e76b6b",
            label="Selected K",
        )
        stride = max(1, len(surface.columns) // 16)
        ax.set_xticks(range(0, len(surface.columns), stride), surface.columns[::stride])
        stride = max(1, len(surface.index) // 20)
        ax.set_yticks(range(0, len(surface.index), stride), surface.index[::stride])
        ax.set(
            xlabel="Candidate K",
            ylabel="Layer",
            title="Relative model-selection criterion",
        )
        fig.colorbar(im, ax=ax, label="(criterion - minimum) / max(|minimum|, 1)")
        ax.legend(loc="upper right")
        return fig


def associations(frame):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
        for axis, part in frame.groupby("axis", sort=True):
            part = part.dropna(subset=["cramers_v"])
            if len(part):
                ax.plot(part.layer, part.cramers_v, marker=".", label=axis)
        ax.set(
            xlabel="Layer",
            ylabel="Cramér's V",
            ylim=(0, 1),
            title="State associations with observed labels",
        )
        if ax.lines:
            ax.legend()
        return fig


def correctness_bands(tags):
    with plt.rc_context(STYLE):
        frame = tags.pivot(index="layer", columns="state", values="accuracy_delta")
        fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
        cmap = plt.get_cmap("coolwarm").with_extremes(bad="#edf0f3")
        im = ax.imshow(
            frame,
            vmin=-1,
            vmax=1,
            cmap=cmap,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
        )
        x_stride = max(1, len(frame.columns) // 24)
        ax.set_xticks(
            range(0, len(frame.columns), x_stride),
            frame.columns[::x_stride],
            rotation=45,
        )
        stride = max(1, len(frame) // 20)
        ax.set_yticks(range(0, len(frame), stride), frame.index[::stride])
        for x, state in enumerate(frame.columns):
            active = np.flatnonzero(frame[state].notna().to_numpy())
            if len(active):
                ax.scatter(
                    [x],
                    [active[0]],
                    marker="v",
                    s=15,
                    c="#287647",
                    edgecolors="white",
                    linewidths=0.3,
                )
                ax.scatter(
                    [x],
                    [active[-1]],
                    marker="^",
                    s=15,
                    c="#9d353f",
                    edgecolors="white",
                    linewidths=0.3,
                )
        ax.set(
            xlabel="Global state ID",
            ylabel="Layer",
            title="Correctness deviation (▼ first appearance, ▲ last appearance)",
        )
        fig.colorbar(im, ax=ax, label="State accuracy - global accuracy")
        return fig


def reliability(profiles, comparisons, diagnostics, cross_profiles):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        ax = axs[0, 0]
        for name, subset in [
            ("Seeds", profiles[profiles.fit_fraction.eq(1) & profiles.k_max.eq(80)]),
            ("Subsamples", profiles[profiles.fit_fraction.lt(1)]),
            ("K range", profiles[profiles.k_max.ne(80)]),
        ]:
            if len(subset):
                grouped = subset.groupby("layer").k.agg(["mean", "std"]).fillna(0)
                ax.plot(grouped.index, grouped["mean"], label=name)
                ax.fill_between(
                    grouped.index,
                    grouped["mean"] - 2 * grouped["std"],
                    grouped["mean"] + 2 * grouped["std"],
                    alpha=0.15,
                )
        ax.set(
            xlabel="Layer", ylabel="Selected K", title="(a) K profiles (mean ± 2 SD)"
        )
        if ax.lines:
            ax.legend()
        ax = axs[0, 1]
        for name, frame, field in [
            ("Matched refits", comparisons, "centroid_cosine_distance"),
            (
                "Uniform random assignment",
                diagnostics,
                "uniform_assignment_centroid_distance",
            ),
        ]:
            if len(frame) and field in frame:
                stats = frame.groupby("layer")[field].agg(["mean", "std"]).fillna(0)
                ax.plot(stats.index, stats["mean"], label=name)
                ax.fill_between(
                    stats.index,
                    stats["mean"] - stats["std"],
                    stats["mean"] + stats["std"],
                    alpha=0.15,
                )
        ax.set(
            xlabel="Layer",
            ylabel="Cosine distance",
            title="(b) Matched centers (mean ± SD)",
        )
        if ax.lines:
            ax.legend()
        ax = axs[0, 2]
        for field, label in [
            ("within_fit_symmetric_kl", "Fitted vs empirical"),
            ("between_cluster_symmetric_kl", "Between components"),
        ]:
            if field in diagnostics:
                part = diagnostics.groupby("layer")[field].mean()
                ax.plot(part.index, part, label=label)
        ax.set(
            xlabel="Layer",
            ylabel="Symmetric diagonal Gaussian KL",
            yscale="symlog",
            title="(c) State separation",
        )
        if ax.lines:
            ax.legend()
        for ax, model in zip(
            axs[1],
            [
                "Qwen/Qwen2-7B-Instruct",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
            ],
        ):
            part = cross_profiles[cross_profiles.model.eq(model)]
            for dataset, frame in part.groupby("dataset"):
                ax.plot(frame.relative_depth, frame.k, marker=".", label=dataset)
            ax.set(
                xlabel="Relative layer depth",
                ylabel="Selected K",
                title=model.split("/")[-1],
            )
            if ax.lines:
                ax.legend()
        return fig


def control_profiles(profiles):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for _, part in profiles.groupby("trial_id", sort=True):
            first = part.iloc[0]
            label = f"{first['method']}; seed={first.seed}; f={first.fit_fraction:g}; PCA={first.pca_components}; std={first.standardize}; eta={first.alignment_threshold:g}"
            for ax, field in zip(axs, ["k", "self_transition", "inherited_fraction"]):
                ax.plot(part.relative_depth, part[field], label=label, alpha=0.8)
        for ax, title in zip(
            axs,
            ["Selected K", "Self-transition probability", "Inherited state fraction"],
        ):
            ax.set(xlabel="Relative layer depth", ylabel=title)
        if profiles.trial_id.nunique() <= 12:
            axs[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
        return fig


def schematic(number):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 3.2), layout="constrained")
        ax.set_axis_off()
        labels = (
            [
                "Hidden vectors",
                "Per-layer regions",
                "Discrete state IDs",
                "State transitions",
            ]
            if number == 1
            else [
                "OpenAct / array data",
                "Fit transforms\nand per-layer clusters",
                "Align centers\nand freeze the map",
                "Evaluate and save\nTables → figures",
            ]
        )
        for i, label in enumerate(labels):
            x = 0.12 + i * 0.25
            ax.text(
                x,
                0.54,
                label,
                ha="center",
                va="center",
                wrap=True,
                fontsize=10,
                bbox={"boxstyle": "round,pad=.9", "fc": "#e8eff6", "ec": "#5a7999"},
                transform=ax.transAxes,
            )
            if i < 3:
                ax.annotate(
                    "",
                    xy=(x + 0.16, 0.54),
                    xytext=(x + 0.10, 0.54),
                    xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": "#455c70"},
                )
        ax.text(
            0.5,
            0.12,
            "Illustrative schematic — no experimental measurements",
            ha="center",
            transform=ax.transAxes,
            color="#536271",
        )
        return fig


def control_diagnostics(diagnostics, comparisons):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for ax, field, label in zip(
            axs.flat[:4],
            [
                "covariance_trace",
                "effective_rank",
                "within_fit_symmetric_kl",
                "between_cluster_symmetric_kl",
            ],
            [
                "Covariance trace",
                "Effective rank",
                "Fitted vs empirical KL",
                "Between-component KL",
            ],
        ):
            if len(diagnostics) and field in diagnostics:
                for trial, part in diagnostics.groupby("trial_id"):
                    ax.plot(part.layer, part[field], label=str(trial)[:8])
                ax.set_yscale("symlog")
            ax.set(xlabel="Layer", ylabel=label)
        for ax, field, label in zip(
            axs.flat[4:],
            ["ari", "centroid_cosine_distance"],
            ["Adjusted Rand index", "Matched-center cosine distance"],
        ):
            if len(comparisons):
                for (_, target), part in comparisons.groupby(["reference", "target"]):
                    ax.plot(part.layer, part[field], label=str(target)[:8])
            ax.set(xlabel="Layer", ylabel=label)
        return fig


def prediction_controls(metrics):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        part = metrics[metrics.predictor.eq("HSS-NB")].copy()
        labels = [
            f"{r.method}; PCA={r.pca_components}; std={r.standardize}; eta={r.alignment_threshold:g}; seed={r.seed}"
            for r in part.itertuples()
        ]
        for ax, metric in zip(axs, ["auroc", "accuracy"]):
            ax.barh(range(len(part)), part[metric], color="#547c9d")
            ax.set_yticks(range(len(part)), labels, fontsize=7)
            ax.set(xlim=(0, 1), xlabel=metric.upper())
        return fig


def global_purity(frame):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for trial, part in frame.groupby("trial_id"):
            ax.plot(part.state, part.layer_purity, "o", label=str(trial)[:8])
        ax.set(
            xlabel="Global component",
            ylabel="Maximum layer fraction",
            ylim=(0, 1),
            title="Stacked-layer clustering: layer purity",
        )
        if ax.lines:
            ax.legend()
        return fig
