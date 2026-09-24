"""Scientific figure functions consuming tables/arrays; never fit or load data."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .style import STYLE
from .publication import (  # re-export the stable plotting API
    state_graph as state_graph,
    geometry as geometry,
    icl_surface as icl_surface,
    associations as associations,
    correctness_bands as correctness_bands,
    prediction_controls as prediction_controls,
    monitoring_controls as monitoring_controls,
    stability_summary as stability_summary,
    separation as separation,
    graph_display as graph_display,
    refit_summary as refit_summary,
)


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
