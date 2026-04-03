# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: HSS
#     language: python
#     name: hss
# ---

# %% [markdown]
# # HSS Analysis: Qwen Math — GMM Matrix Results
#
# This notebook performs a complete post-hoc analysis of the 4-combo GMM
# discretization matrix (2 views × 2 preprocessing variants) that was produced
# by `run_bundle_gmm_matrix.py`.
#
# **What has already been computed (expensive):**
# - `scan` — ICL / silhouette / stability for every (layer, k) pair
# - `select` — optimal k per layer via ICL-parsimonious selection
# - `discretize` — full GMM clustering + cross-layer alignment → `global_labels`
#
# **What this notebook computes (cheap, seconds each):**
# - Preliminary suite (dynamics, information theory, label analysis)
# - Prediction (HSS-NB, HSS-Markov vs. continuous baselines)
# - Cross-combo comparison
# - Custom trajectory pattern analysis
#
# All heavy lifting is already done; everything here runs on a small integer
# matrix `global_labels [N, L]` plus pre-computed cluster centers.

# %% [markdown]
# ## 0. Setup

# %%
import sys
import json
import warnings
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm, Normalize
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# %% [markdown]
# ### Path configuration
#
# Adjust these two paths to match your machine:

# %%
# -- EDIT THESE --
PROJECT_ROOT = Path("/workspace/projects/hidden-states-as-states")
RESULTS_ROOT = Path("/workspace/results/qwen_math_gmm_matrix_k50_reg1e4")

# Add project source to Python path
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Quick sanity check
assert RESULTS_ROOT.exists(), f"Results not found: {RESULTS_ROOT}"
print(f"Project root : {PROJECT_ROOT}")
print(f"Results root : {RESULTS_ROOT}")

# %%
import hss
from hss_workbench import Workspace
from experiments.config import (
    PredictionConfig, DiscretizeConfig, StabilityConfig,
    ScanConfig, SelectConfig, VisualizationConfig,
)
from experiments.prediction import run_prediction
from experiments.preliminary import run_preliminary_suite
from experiments.information import run_information_analysis, run_effective_rank
from experiments.label_analysis import run_label_analysis, find_sink_states
from experiments.statistics import run_dynamics, run_sequence_stats
from experiments.discretize import run_tolerance_sweep, build_rel_icl_surface
from hss.trajectory import (
    count_transitions, self_transition_prob, active_states_per_layer,
)

print(f"hss version: {hss.__version__}")

# %%
# Plotting defaults
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "font.size": 10,
})
COLORS = {
    "blue": "#2171b5", "orange": "#d94801", "green": "#238b45",
    "red": "#cb181d", "purple": "#6a51a3", "gray": "#636363",
    "cyan": "#0097a7",
}

# %% [markdown]
# ---
# ## 1. Matrix Overview — What Do We Have?

# %%
# Load the matrix manifest produced by run_bundle_gmm_matrix.py
manifest_path = RESULTS_ROOT / "matrix_manifest.json"
with open(manifest_path) as f:
    matrix_manifest = json.load(f)

combos = []
for entry in matrix_manifest:
    combos.append({
        "view": entry.get("view_name", ""),
        "preprocess": entry.get("preprocess_name", ""),
        "status": entry.get("status", ""),
        "n_items": entry.get("n_items"),
        "n_layers": entry.get("n_layers"),
        "state_dim": entry.get("state_dim"),
        "n_global_states": entry.get("n_global_states"),
        "k_mean": entry.get("k_summary", {}).get("k_mean"),
        "k_min": entry.get("k_summary", {}).get("k_min"),
        "k_max": entry.get("k_summary", {}).get("k_max"),
        "positive_rate": entry.get("positive_rate"),
        "workspace": entry.get("workspace_root", ""),
    })

combo_df = pd.DataFrame(combos)
display(combo_df.drop(columns=["workspace"]))

# %%
# Enumerate workspace directories
COMBO_DIRS = {}
for entry in matrix_manifest:
    if entry.get("status") == "done":
        key = f"{entry['view_name']}__{entry['preprocess_name']}"
        ws_path = Path(entry.get("workspace_root", ""))
        if not ws_path.exists():
            # Try constructing from RESULTS_ROOT
            ws_path = RESULTS_ROOT / f"{entry['view_name']}__{entry['preprocess_name']}__gmm"
        if ws_path.exists():
            COMBO_DIRS[key] = ws_path

print(f"Found {len(COMBO_DIRS)} workspace(s):")
for name, path in COMBO_DIRS.items():
    print(f"  {name:40s} -> {path}")

# %% [markdown]
# ---
# ## 2. Deep Dive — Pick One Combo
#
# We start by doing a thorough analysis of a single combo, then compare
# across all four in Section 5.  By default we pick `sample_mean__standardize`
# because standardization is the recommended preprocessing for GMM.

# %%
# Pick the primary combo to analyze in depth
PRIMARY_KEY = "sample_mean__standardize"
if PRIMARY_KEY not in COMBO_DIRS:
    PRIMARY_KEY = list(COMBO_DIRS.keys())[0]
    print(f"Fallback to: {PRIMARY_KEY}")

PRIMARY_DIR = COMBO_DIRS[PRIMARY_KEY]
print(f"Primary combo: {PRIMARY_KEY}")
print(f"Workspace dir: {PRIMARY_DIR}")

# %%
ws = Workspace(PRIMARY_DIR, quiet=True, thread_limit=1)
data = ws.open_data()

print(f"N items   : {data.loaded.n_items:,}")
print(f"N layers  : {len(data.layers)}")
print(f"State dim : {data.loaded.state_dim}")
print(f"Layers    : {data.layers}")
print(f"Has labels: {len(data.y) > 0}")
if len(data.y) > 0:
    print(f"Positive rate: {data.y.mean():.4f}  ({data.y.sum():,} / {len(data.y):,})")

# %% [markdown]
# ### 2.1 Scan Metrics — ICL, Silhouette, Stability across k

# %%
scan_art = ws.load_scan("scan")
metrics = scan_art.metrics_df.copy()

print(f"Scan metrics: {len(metrics)} rows")
print(f"Layers scanned: {sorted(metrics['layer'].unique())[:5]} ... ({metrics['layer'].nunique()} total)")
print(f"k range: [{metrics['k'].min()}, {metrics['k'].max()}]")
display(metrics.head(10))

# %%
# ICL curves per layer (lower is better)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel 1: ICL vs k, one line per layer
ax = axes[0]
layers_to_show = sorted(metrics["layer"].unique())
n_layers_total = len(layers_to_show)
cmap = plt.cm.viridis
for i, layer in enumerate(layers_to_show):
    sub = metrics[metrics["layer"] == layer].sort_values("k")
    color = cmap(i / max(1, n_layers_total - 1))
    ax.plot(sub["k"], sub["icl"], color=color, alpha=0.6, lw=0.8)
ax.set_xlabel("k")
ax.set_ylabel("ICL (lower = better)")
ax.set_title("ICL vs k (each line = one layer)")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(
    vmin=min(layers_to_show), vmax=max(layers_to_show)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Layer", shrink=0.8)

# Panel 2: Silhouette vs k
ax = axes[1]
for i, layer in enumerate(layers_to_show):
    sub = metrics[metrics["layer"] == layer].sort_values("k")
    color = cmap(i / max(1, n_layers_total - 1))
    ax.plot(sub["k"], sub["silhouette"], color=color, alpha=0.6, lw=0.8)
ax.set_xlabel("k")
ax.set_ylabel("Silhouette (higher = better)")
ax.set_title("Silhouette vs k")

# Panel 3: Stability vs k
ax = axes[2]
for i, layer in enumerate(layers_to_show):
    sub = metrics[metrics["layer"] == layer].sort_values("k")
    if "stability" in sub.columns:
        color = cmap(i / max(1, n_layers_total - 1))
        ax.plot(sub["k"], sub["stability"], color=color, alpha=0.6, lw=0.8)
ax.set_xlabel("k")
ax.set_ylabel("Stability (ARI, higher = better)")
ax.set_title("Stability vs k")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.2 K Selection — What k Was Chosen Per Layer?

# %%
select_art = ws.load_select("select")
k_map = select_art.k_map
ks = np.array(list(k_map.values()))
layers_sorted = sorted(k_map.keys())

print(f"Strategy: {select_art.strategy}")
print(f"k map: {dict(sorted(k_map.items()))}")
print(f"k stats: mean={ks.mean():.1f}, std={ks.std():.1f}, min={ks.min()}, max={ks.max()}")

# %%
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.bar(range(len(layers_sorted)), [k_map[l] for l in layers_sorted],
       color=COLORS["blue"], alpha=0.8, edgecolor="black", linewidth=0.3)
ax.set_xticks(range(len(layers_sorted)))
ax.set_xticklabels([str(l) for l in layers_sorted], rotation=45, ha="right", fontsize=7)
ax.set_xlabel("Layer")
ax.set_ylabel("Selected k")
ax.set_title(f"Number of clusters per layer (mean={ks.mean():.1f})")
ax.axhline(ks.mean(), color=COLORS["red"], ls="--", lw=1, alpha=0.6, label=f"mean={ks.mean():.1f}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3 Tolerance Sweep — How Sensitive Is k to the Parsimony Threshold?

# %%
tolerance_df = run_tolerance_sweep(
    metrics, tolerances=np.linspace(0.0, 0.10, 51),
    stability_min=0.3,
)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(tolerance_df["tolerance"], tolerance_df["avg_k"],
         color=COLORS["blue"], lw=2, marker="o", ms=3)
ax1.set_xlabel("Parsimony tolerance (relative ICL %)")
ax1.set_ylabel("Average k", color=COLORS["blue"])
ax1.tick_params(axis="y", labelcolor=COLORS["blue"])
ax1.axhline(ks.mean(), color=COLORS["blue"], ls=":", alpha=0.4)

if "std_k" in tolerance_df.columns:
    ax2 = ax1.twinx()
    ax2.plot(tolerance_df["tolerance"], tolerance_df["std_k"],
             color=COLORS["orange"], lw=1.2, marker="s", ms=2, alpha=0.7)
    ax2.set_ylabel("Std(k)", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])

ax1.set_title("Tolerance sweep: how parsimony threshold affects k")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.4 Discretization Summary — Global Labels & Alignment

# %%
disc_art = ws.load_discretize("discretize")
gl = np.asarray(disc_art.global_labels)
hss_result = disc_art.hss_result
layers = list(disc_art.layers)

print(f"Global labels shape : {gl.shape}  (N_items x N_layers)")
print(f"N global states     : {disc_art.n_global_states}")
print(f"Effective k map     : {disc_art.k_map}")
print(f"Unique states used  : {len(np.unique(gl[gl >= 0]))}")

# %%
# Active states per layer + self-transition probability
self_trans = self_transition_prob(gl)
active = active_states_per_layer(gl)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
ax.plot(layers, active, marker="o", ms=4, color=COLORS["blue"], lw=1.5)
ax.set_xlabel("Layer")
ax.set_ylabel("Active states")
ax.set_title("Active states per layer")

ax = axes[1]
ax.plot(layers[:-1], self_trans, marker="o", ms=4, color=COLORS["green"], lw=1.5)
ax.set_xlabel("Layer pair (l → l+1)")
ax.set_ylabel("Self-transition probability")
ax.set_title(f"Self-transition (mean={np.nanmean(self_trans):.3f})")
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.5 Alignment Events — Births and Deaths

# %%
if hss_result.alignment is not None:
    alignment = hss_result.alignment
    births_per_step = [len(step.births) for step in alignment.steps]
    deaths_per_step = [len(step.deaths) for step in alignment.steps]
    step_layers = [step.layer_to for step in alignment.steps]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    width = 0.35
    x = np.arange(len(step_layers))
    ax.bar(x - width/2, births_per_step, width, label="Births", color=COLORS["green"], alpha=0.8)
    ax.bar(x + width/2, deaths_per_step, width, label="Deaths", color=COLORS["red"], alpha=0.8)
    ax.set_xticks(x[::max(1, len(x)//15)])
    ax.set_xticklabels([str(step_layers[i]) for i in range(0, len(step_layers), max(1, len(x)//15))],
                       rotation=45, fontsize=7)
    ax.set_xlabel("Target layer")
    ax.set_ylabel("Count")
    ax.set_title("Cluster births and deaths across layers")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"Total births: {sum(births_per_step)}")
    print(f"Total deaths: {sum(deaths_per_step)}")
else:
    print("No alignment (single layer?)")

# %% [markdown]
# ---
# ## 3. Preliminary Analysis — Dynamics, Information, Labels

# %%
# Run the full preliminary suite (fast — operates on global_labels + y)
if not ws.store.has_artifact("analysis", "prelim"):
    ws.run_preliminary("prelim", discretize="discretize")
    print("Preliminary analysis completed.")
else:
    print("Preliminary analysis already exists, loading.")

prelim = ws.load_preliminary("prelim")

# %% [markdown]
# ### 3.1 Dynamics — State Sizes and Flows

# %%
node_df = prelim.dynamics.node_df
flow_df = prelim.dynamics.flow_df

print(f"Node table : {len(node_df)} rows (layer × cluster)")
print(f"Flow table : {len(flow_df)} rows (transitions)")
display(node_df.head(10))

# %%
# Distribution of cluster sizes across all layers
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
ax.hist(node_df["count"], bins=50, color=COLORS["blue"], alpha=0.7, edgecolor="black", lw=0.3)
ax.set_xlabel("Cluster size (# items)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of cluster sizes across all layers")
ax.axvline(node_df["count"].median(), color=COLORS["red"], ls="--", lw=1,
           label=f"median={node_df['count'].median():.0f}")
ax.legend(fontsize=8)

# Fraction of items in top-k clusters per layer
ax = axes[1]
for topn in [1, 3, 5]:
    fracs = []
    for layer in layers:
        sub = node_df[node_df["layer"] == layer].sort_values("count", ascending=False)
        total = sub["count"].sum()
        top = sub.head(topn)["count"].sum()
        fracs.append(top / total if total > 0 else 0)
    ax.plot(layers, fracs, marker="o", ms=3, lw=1.2, label=f"Top-{topn}")
ax.set_xlabel("Layer")
ax.set_ylabel("Fraction of items")
ax.set_title("Concentration: fraction of items in largest clusters")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 Information Theory — Does Discretization Capture Label Info?

# %%
y = data.y
if len(y) > 0 and prelim.information is not None:
    info_df = prelim.information.df.copy()
    display(info_df)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    c1, c2 = COLORS["blue"], COLORS["red"]
    l1, = ax1.plot(info_df["layer"], info_df["H_C"], marker="o", ms=5,
                   color=c1, lw=1.5, label="$H(C_\\ell)$")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("$H(C_\\ell)$ (bits)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2 = ax1.twinx()
    l2, = ax2.plot(info_df["layer"], info_df["I_CY"], marker="s", ms=5,
                   color=c2, lw=1.5, label="$I(C_\\ell; Y)$")
    ax2.set_ylabel("$I(C_\\ell; Y)$ (bits)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    H_Y = float(info_df["H_Y"].iloc[0])
    ax2.axhline(H_Y, color=c2, ls="--", alpha=0.4, lw=1)
    ax2.text(info_df["layer"].iloc[-1], H_Y * 1.02, f"$H(Y)$={H_Y:.3f}",
             fontsize=8, color=c2, ha="right")

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()],
               loc="upper left", fontsize=9)
    ax1.set_title("Information curve: cluster entropy vs. mutual information with label")
    plt.tight_layout()
    plt.show()

    # Information bottleneck plane
    fig, ax = plt.subplots(figsize=(6, 5))
    x = info_df["H_C"].values
    yv = info_df["I_CY"].values
    lvals = info_df["layer"].values
    norm = Normalize(vmin=lvals.min(), vmax=lvals.max())
    sc = ax.scatter(x, yv, c=lvals, cmap="viridis", norm=norm,
                    s=50, edgecolors="black", linewidths=0.4, zorder=3)
    ax.plot(x, yv, lw=0.8, alpha=0.4, color="gray", zorder=1)
    for i in range(0, len(x), max(1, len(x) // 8)):
        ax.annotate(f"L{lvals[i]}", (x[i], yv[i]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("$H(C_\\ell)$ (bits)")
    ax.set_ylabel("$I(C_\\ell; Y)$ (bits)")
    ax.set_title("Information plane")
    plt.colorbar(sc, ax=ax, label="Layer", shrink=0.8)
    plt.tight_layout()
    plt.show()
else:
    print("No binary labels available — skipping information analysis.")

# %% [markdown]
# ### 3.3 Label Analysis — Which Clusters Are "Correct" vs. "Incorrect"?

# %%
if len(y) > 0 and prelim.label_analysis is not None:
    la = prelim.label_analysis
    print(f"Baseline P(correct) = {la.baseline:.4f}")
    print(f"N global states     = {la.delta_matrix.shape[1]}")
    print(f"Cluster order (by mean delta): {la.cluster_order[:15]} ...")

    # Delta heatmap: P(correct|cluster) - P(correct)
    order = la.cluster_order
    sub = la.delta_matrix[:, order]
    finite = sub[np.isfinite(sub)]
    vmax = max(float(np.nanmax(np.abs(finite))) if finite.size else 0.1, 0.01)

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.25), 5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(sub, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Global Cluster ID (sorted by mean delta)")
    ax.set_ylabel("Layer index")

    n_c = len(order)
    step_x = max(1, n_c // 15)
    ax.set_xticks(range(0, n_c, step_x))
    ax.set_xticklabels([str(order[i]) for i in range(0, n_c, step_x)],
                       rotation=45, ha="right", fontsize=7)
    step_y = max(1, len(layers) // 12)
    ax.set_yticks(range(0, len(layers), step_y))
    ax.set_yticklabels([str(layers[i]) for i in range(0, len(layers), step_y)], fontsize=7)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("$P(correct|cluster) - P(correct)$", fontsize=9)
    ax.set_title("Delta heatmap: red = more correct, blue = more incorrect")
    plt.tight_layout()
    plt.show()

    # Top positive and negative clusters
    mean_delta = np.nanmean(la.delta_matrix, axis=0)
    total_count = la.count_matrix.sum(axis=0)
    active_mask = total_count > 0
    gids = np.arange(len(mean_delta))

    top_pos = gids[active_mask][np.argsort(-mean_delta[active_mask])[:10]]
    top_neg = gids[active_mask][np.argsort(mean_delta[active_mask])[:10]]

    print("\nTop 10 POSITIVE clusters (most associated with correct answers):")
    for gid in top_pos:
        print(f"  GID {gid:3d}: delta = {mean_delta[gid]:+.4f}, "
              f"total count = {total_count[gid]:,}")

    print("\nTop 10 NEGATIVE clusters (most associated with incorrect answers):")
    for gid in top_neg:
        print(f"  GID {gid:3d}: delta = {mean_delta[gid]:+.4f}, "
              f"total count = {total_count[gid]:,}")

    # Sink states
    sinks = prelim.sink_states
    if len(sinks) > 0:
        print(f"\nSink states ({len(sinks)} found):")
        display(sinks)
    else:
        print("\nNo sink states found with default criteria.")
else:
    print("No labels — skipping label analysis.")

# %% [markdown]
# ### 3.4 Sequence Statistics

# %%
seq = prelim.sequence_stats
print(f"Self-transition mean : {seq.self_trans_mean:.4f}")
print(f"Active states mean   : {seq.active_mean:.1f}")
print(f"Marginal freq shape  : {seq.marginal_freq.shape}")

# Marginal frequency distribution
fig, ax = plt.subplots(figsize=(8, 3.5))
freq = seq.marginal_freq
freq_nz = freq[freq > 0]
ax.bar(range(len(freq_nz)), np.sort(freq_nz)[::-1],
       color=COLORS["blue"], alpha=0.7, edgecolor="black", lw=0.2)
ax.set_xlabel("Global state (sorted by frequency)")
ax.set_ylabel("Marginal frequency")
ax.set_title(f"State frequency distribution ({len(freq_nz)} active states)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 4. Prediction — Can Discrete Trajectories Predict Correctness?

# %%
if len(y) > 0:
    pred_cfg = PredictionConfig(
        n_splits=5,
        seed=42,
        alpha=1.0,
        methods=["HSS-NB", "HSS-Markov", "GaussianNB", "Logistic"],
        continuous_feature_mode="last_layer",
        continuous_standardize=True,
    )

    if not ws.store.has_artifact("analysis", "pred"):
        print("Running prediction (5-fold CV)...")
        ws.run_prediction("pred", discretize="discretize", config=pred_cfg)
        print("Done.")
    else:
        print("Prediction already exists, loading.")

    pred = ws.load_prediction("pred")
    print("\n=== Prediction Summary ===")
    display(pred.summary_df)
    print()
    display(pred.fold_df.groupby("method")[["auroc", "accuracy", "f1"]].describe().round(4))
else:
    pred = None
    print("No labels — skipping prediction.")

# %%
if pred is not None and len(pred.summary_df) > 0:
    summary = pred.summary_df.copy()
    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    means = summary["mean_auroc"].values
    stds = summary["std_auroc"].values

    # Color coding: HSS methods vs continuous baselines
    colors = []
    for m in methods:
        if m.startswith("HSS"):
            colors.append(COLORS["blue"])
        else:
            colors.append(COLORS["orange"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("Prediction comparison: HSS discrete (blue) vs continuous baselines (orange)")

    # Annotate values
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    # Reference line at best HSS
    hss_mask = [m.startswith("HSS") for m in methods]
    if any(hss_mask):
        best_hss = max(means[i] for i, h in enumerate(hss_mask) if h)
        ax.axhline(best_hss, color=COLORS["blue"], ls="--", lw=0.8, alpha=0.5)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 4.1 Progressive Prefix Prediction — How Many Layers Do We Need?

# %%
if len(y) > 0 and np.unique(y).shape[0] >= 2:
    # Run prediction with increasing number of layers
    prefix_step = max(1, len(layers) // 10)  # ~10 data points
    prefix_records = []

    prefix_cfg = PredictionConfig(
        n_splits=5, seed=42, alpha=1.0,
        methods=["HSS-NB", "HSS-Markov"],
    )

    prefixes = list(range(max(1, prefix_step), len(layers) + 1, prefix_step))
    if prefixes[-1] != len(layers):
        prefixes.append(len(layers))

    print(f"Running progressive prefix prediction ({len(prefixes)} prefixes)...")
    for plen in prefixes:
        gl_prefix = gl[:, :plen]
        layers_prefix = layers[:plen]
        result = run_prediction(
            data.loaded.provider, gl_prefix, y, layers_prefix,
            prefix_cfg, store=None,
        )
        for _, row in result.summary_df.iterrows():
            prefix_records.append({
                "prefix_len": plen,
                "last_layer": layers_prefix[-1],
                "method": row["method"],
                "mean_auroc": row.get("mean_auroc", np.nan),
            })

    prefix_df = pd.DataFrame(prefix_records)
    print("Done.")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for method in prefix_df["method"].unique():
        sub = prefix_df[prefix_df["method"] == method]
        ax.plot(sub["last_layer"], sub["mean_auroc"],
                marker="o", ms=4, lw=1.5, label=method)
    ax.set_xlabel("Last layer included")
    ax.set_ylabel("AUROC")
    ax.set_title("Progressive prefix prediction: more layers → better?")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping prefix prediction (no valid labels).")

# %% [markdown]
# ---
# ## 5. Cross-Combo Comparison
#
# Now we compare all 4 combos (2 views × 2 preprocessings) on the same
# metrics. This answers:
# - Does `sample_mean` vs `sample_prompt_last` matter?
# - Does standardization help?

# %%
cross_records = []

for combo_key, combo_dir in COMBO_DIRS.items():
    try:
        _ws = Workspace(combo_dir, quiet=True, thread_limit=1)
        _data = _ws.open_data()
        _disc = _ws.load_discretize("discretize")
        _select = _ws.load_select("select")
        _gl = np.asarray(_disc.global_labels)
        _y = _data.y
        _layers = list(_disc.layers)
        _ks = np.array(list(_select.k_map.values()))

        # Basic stats
        rec = {
            "combo": combo_key,
            "n_items": _data.loaded.n_items,
            "n_layers": len(_layers),
            "n_global_states": _disc.n_global_states,
            "k_mean": _ks.mean(),
            "k_std": _ks.std(),
            "self_trans_mean": float(np.nanmean(self_transition_prob(_gl))),
            "active_states_mean": float(active_states_per_layer(_gl).mean()),
        }

        # Information (if labels exist)
        if len(_y) > 0:
            from experiments.information import _entropy_bits, _mutual_information_bits
            # Quick MI at last layer
            last_col = _gl[:, -1]
            valid = last_col >= 0
            mi_last = _mutual_information_bits(last_col[valid], _y[valid])
            rec["MI_last_layer"] = mi_last

            # Quick MI at best layer
            best_mi = 0.0
            best_layer = _layers[0]
            for li in range(len(_layers)):
                col = _gl[:, li]
                v = col >= 0
                mi = _mutual_information_bits(col[v], _y[v])
                if mi > best_mi:
                    best_mi = mi
                    best_layer = _layers[li]
            rec["MI_best"] = best_mi
            rec["MI_best_layer"] = best_layer

        # Prediction (run if not exists)
        pred_cfg_cross = PredictionConfig(
            n_splits=5, seed=42, methods=["HSS-NB", "HSS-Markov"],
        )
        pred_name = "pred_cross"
        if not _ws.store.has_artifact("analysis", pred_name):
            _ws.run_prediction(pred_name, discretize="discretize", config=pred_cfg_cross)
        _pred = _ws.load_prediction(pred_name)

        for _, row in _pred.summary_df.iterrows():
            rec[f"auroc_{row['method']}"] = row.get("mean_auroc", np.nan)

        cross_records.append(rec)
        print(f"  ✓ {combo_key}")
    except Exception as e:
        print(f"  ✗ {combo_key}: {e}")

cross_df = pd.DataFrame(cross_records)
display(cross_df.round(4))

# %%
if len(cross_df) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: k_mean comparison
    ax = axes[0]
    x = range(len(cross_df))
    ax.barh(x, cross_df["k_mean"], xerr=cross_df["k_std"],
            color=COLORS["blue"], alpha=0.8, capsize=3)
    ax.set_yticks(x)
    ax.set_yticklabels(cross_df["combo"], fontsize=8)
    ax.set_xlabel("Mean k")
    ax.set_title("Average clusters per layer")

    # Panel 2: n_global_states comparison
    ax = axes[1]
    ax.barh(x, cross_df["n_global_states"], color=COLORS["green"], alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(cross_df["combo"], fontsize=8)
    ax.set_xlabel("N global states")
    ax.set_title("Total unique states after alignment")

    # Panel 3: AUROC comparison
    ax = axes[2]
    auroc_cols = [c for c in cross_df.columns if c.startswith("auroc_")]
    width = 0.35
    for i, col in enumerate(auroc_cols):
        method = col.replace("auroc_", "")
        offset = (i - len(auroc_cols) / 2 + 0.5) * width
        ax.barh([j + offset for j in x], cross_df[col],
                height=width * 0.9, alpha=0.8,
                label=method)
    ax.set_yticks(x)
    ax.set_yticklabels(cross_df["combo"], fontsize=8)
    ax.set_xlabel("AUROC")
    ax.set_title("Prediction performance")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 5.1 Cross-Combo Information Curves

# %%
if len(y) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for combo_key, combo_dir in COMBO_DIRS.items():
        try:
            _ws = Workspace(combo_dir, quiet=True, thread_limit=1)
            _data = _ws.open_data()
            _disc = _ws.load_discretize("discretize")
            _gl = np.asarray(_disc.global_labels)
            _y = _data.y
            _layers = list(_disc.layers)

            if len(_y) == 0:
                continue

            from experiments.information import _entropy_bits, _mutual_information_bits
            h_c_vals, mi_vals = [], []
            for li in range(len(_layers)):
                col = _gl[:, li]
                v = col >= 0
                c_counts = np.bincount(col[v])
                h_c_vals.append(_entropy_bits(c_counts))
                mi_vals.append(_mutual_information_bits(col[v], _y[v]))

            axes[0].plot(_layers, h_c_vals, marker="o", ms=3, lw=1.2,
                         label=combo_key, alpha=0.8)
            axes[1].plot(_layers, mi_vals, marker="s", ms=3, lw=1.2,
                         label=combo_key, alpha=0.8)
        except Exception:
            pass

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("$H(C_\\ell)$ (bits)")
    axes[0].set_title("Cluster entropy across combos")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("$I(C_\\ell; Y)$ (bits)")
    axes[1].set_title("Mutual information with label across combos")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ## 6. Trajectory Pattern Analysis
#
# Custom analysis: what are the most common state trajectories,
# and how do they relate to correctness?

# %%
# Encode each sample's trajectory as a string
traj_strs = ["_".join(str(x) for x in row) for row in gl]
traj_series = pd.Series(traj_strs)
traj_counts = traj_series.value_counts()

print(f"Total unique trajectories: {len(traj_counts):,}")
print(f"Top trajectory covers {traj_counts.iloc[0]:,} / {len(gl):,} "
      f"= {traj_counts.iloc[0]/len(gl):.1%} of items")
print(f"Top 10 trajectories cover {traj_counts.head(10).sum():,} / {len(gl):,} "
      f"= {traj_counts.head(10).sum()/len(gl):.1%} of items")

# %%
# Trajectory diversity curve
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
cumfrac = np.cumsum(traj_counts.values) / len(gl)
ax.plot(range(1, len(cumfrac) + 1), cumfrac, color=COLORS["blue"], lw=1.5)
ax.set_xlabel("Number of trajectory types (sorted by frequency)")
ax.set_ylabel("Cumulative fraction of items")
ax.set_title("Trajectory diversity: how many patterns cover the data?")
ax.axhline(0.5, color=COLORS["gray"], ls="--", alpha=0.5)
ax.axhline(0.9, color=COLORS["gray"], ls="--", alpha=0.5)
ax.set_xscale("log")
n_50 = int(np.searchsorted(cumfrac, 0.5) + 1)
n_90 = int(np.searchsorted(cumfrac, 0.9) + 1)
ax.annotate(f"50% at {n_50} types", xy=(n_50, 0.5), fontsize=8,
            xytext=(n_50 * 2, 0.4), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate(f"90% at {n_90} types", xy=(n_90, 0.9), fontsize=8,
            xytext=(n_90 * 2, 0.8), arrowprops=dict(arrowstyle="->", lw=0.8))

# Trajectory length vs frequency
ax = axes[1]
ax.hist(traj_counts.values, bins=50, color=COLORS["blue"], alpha=0.7,
        edgecolor="black", lw=0.3, log=True)
ax.set_xlabel("Trajectory frequency")
ax.set_ylabel("Count (log scale)")
ax.set_title("Distribution of trajectory frequencies")

plt.tight_layout()
plt.show()

# %%
# Top trajectories and their correctness rates
if len(y) > 0:
    top_n = 20
    top_trajs = traj_counts.head(top_n)

    records = []
    for traj_str, count in top_trajs.items():
        mask = np.array([t == traj_str for t in traj_strs])
        pos_rate = float(y[mask].mean())
        # Count unique states in this trajectory
        states = [int(x) for x in traj_str.split("_")]
        n_unique = len(set(states))
        n_changes = sum(1 for i in range(len(states)-1) if states[i] != states[i+1])
        records.append({
            "trajectory": traj_str[:60] + ("..." if len(traj_str) > 60 else ""),
            "count": count,
            "pct": count / len(gl) * 100,
            "pos_rate": pos_rate,
            "delta": pos_rate - float(y.mean()),
            "n_unique_states": n_unique,
            "n_transitions": n_changes,
        })

    top_df = pd.DataFrame(records)
    display(top_df.round(4))

    # Scatter: trajectory frequency vs correctness
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline = float(y.mean())
    colors_scatter = [COLORS["green"] if r["delta"] > 0 else COLORS["red"]
                      for r in records]
    sizes = [max(20, r["count"] / 5) for r in records]
    ax.scatter(range(top_n), [r["pos_rate"] for r in records],
               c=colors_scatter, s=sizes, edgecolors="black", linewidths=0.4, zorder=3)
    ax.axhline(baseline, color="black", ls="--", lw=1, alpha=0.6,
               label=f"baseline = {baseline:.3f}")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(range(1, top_n + 1))
    ax.set_xlabel(f"Top-{top_n} most frequent trajectories")
    ax.set_ylabel("P(correct)")
    ax.set_title("Correctness rate of most common trajectory patterns")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 6.1 Trajectory Stability — Do Nearby Layers Agree?

# %%
# Per-item "churn": how many times does the global state change?
n_changes = np.sum(gl[:, 1:] != gl[:, :-1], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax = axes[0]
ax.hist(n_changes, bins=range(0, int(n_changes.max()) + 2),
        color=COLORS["blue"], alpha=0.7, edgecolor="black", lw=0.3)
ax.set_xlabel("Number of state changes across layers")
ax.set_ylabel("Number of items")
ax.set_title(f"Trajectory churn (mean={n_changes.mean():.1f}, max={n_changes.max()})")

if len(y) > 0:
    ax = axes[1]
    # Bin by churn level and compute correctness
    max_churn = min(int(n_changes.max()), 30)
    churn_bins = range(0, max_churn + 1)
    churn_rates = []
    churn_counts = []
    for c in churn_bins:
        mask = n_changes == c
        if mask.sum() > 10:
            churn_rates.append(float(y[mask].mean()))
            churn_counts.append(int(mask.sum()))
        else:
            churn_rates.append(np.nan)
            churn_counts.append(0)
    ax.bar(list(churn_bins), churn_rates, color=COLORS["blue"], alpha=0.7,
           edgecolor="black", lw=0.3)
    ax.axhline(float(y.mean()), color=COLORS["red"], ls="--", lw=1,
               label=f"baseline={y.mean():.3f}")
    ax.set_xlabel("Number of state changes")
    ax.set_ylabel("P(correct)")
    ax.set_title("Correctness vs trajectory churn")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 7. Relative ICL Surface — The Landscape of k Choices

# %%
rel_mat, rel_layers, rel_k = build_rel_icl_surface(metrics)

fig, ax = plt.subplots(figsize=(12, 5))
finite = rel_mat[np.isfinite(rel_mat)]
vmax = max(float(np.quantile(finite, 0.95)) if finite.size else 0.2, 0.05)
im = ax.imshow(rel_mat, aspect="auto", interpolation="nearest",
               cmap="magma_r", vmin=0.0, vmax=vmax, origin="lower")
ax.set_xlabel("k")
ax.set_ylabel("Layer")

K = len(rel_k)
L = len(rel_layers)
k_step = max(1, K // 10)
ax.set_xticks(range(0, K, k_step))
ax.set_xticklabels([str(rel_k[j]) for j in range(0, K, k_step)], fontsize=8)
l_step = max(1, L // 10)
ax.set_yticks(range(0, L, l_step))
ax.set_yticklabels([str(int(rel_layers[i])) for i in range(0, L, l_step)], fontsize=8)

# Overlay selected k path
k_to_j = {int(k): j for j, k in enumerate(rel_k)}
xs, ys = [], []
for i, layer in enumerate(rel_layers):
    ks_sel = k_map.get(int(layer))
    j = k_to_j.get(int(ks_sel)) if ks_sel is not None else None
    if j is not None:
        xs.append(j)
        ys.append(i)
if xs:
    ax.plot(xs, ys, color="white", lw=2.5, alpha=0.9, zorder=5)
    ax.plot(xs, ys, color="cyan", lw=1.0, alpha=0.9, zorder=6, label="Selected k")
    ax.legend(fontsize=8)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Relative ICL", fontsize=9)
ax.set_title("Relative ICL surface — darker = better; cyan line = selected k")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 8. Summary Table

# %%
print("=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)
print(f"Primary combo       : {PRIMARY_KEY}")
print(f"N items             : {data.loaded.n_items:,}")
print(f"N layers            : {len(layers)}")
print(f"Hidden dim          : {data.loaded.state_dim}")
print(f"N global states     : {disc_art.n_global_states}")
print(f"Mean k              : {ks.mean():.1f} ± {ks.std():.1f}")
print(f"Self-transition mean: {np.nanmean(self_trans):.4f}")
print(f"Unique trajectories : {len(traj_counts):,}")
print(f"50% coverage at     : {n_50} trajectory types")

if len(y) > 0:
    print(f"Positive rate       : {y.mean():.4f}")
    if prelim.information is not None:
        best_mi_row = prelim.information.df.loc[prelim.information.df["I_CY"].idxmax()]
        print(f"Best MI layer       : {int(best_mi_row['layer'])} "
              f"(I={best_mi_row['I_CY']:.4f} bits)")
    if pred is not None and len(pred.summary_df) > 0:
        best_pred = pred.summary_df.loc[pred.summary_df["mean_auroc"].idxmax()]
        print(f"Best predictor      : {best_pred['method']} "
              f"(AUROC={best_pred['mean_auroc']:.4f})")

if len(cross_df) > 1:
    print(f"\n--- Cross-combo comparison ---")
    best_combo = cross_df.loc[cross_df.get("auroc_HSS-NB", cross_df.iloc[:, -1]).idxmax()]
    print(f"Best combo          : {best_combo['combo']}")

print("=" * 70)
print("Done. All results are saved in each workspace's analysis directory.")

# %% [markdown]
# ---
# ## Appendix: How to Re-run Specific Analyses
#
# ```python
# # Re-run prediction with different methods
# ws.run_prediction("pred_v2", discretize="discretize",
#     config=PredictionConfig(methods=["HSS-NB", "HSS-Markov", "MLP"]),
#     overwrite=True)
#
# # Run stability analysis (slower — re-fits GMM multiple times)
# ws.run_stability("stab_v1", discretize="discretize",
#     discretize_config=DiscretizeConfig(method="gmm"),
#     stability_config=StabilityConfig(seeds=[42,43,44]),
#     overwrite=True)
#
# # Run ablation (alignment strategy comparison)
# from experiments.config import AblationConfig
# ws.run_ablation("ablation_v1", discretize="discretize",
#     ablation_config=AblationConfig(
#         enabled=["cosine_vs_euclidean", "threshold_sweep", "alignment"]),
#     overwrite=True)
#
# # Custom analysis with any function
# def my_func(global_labels, y, layers, store):
#     # ... your code here ...
#     store.save_json("result/my_output.json", {"key": "value"})
#     return result
#
# ws.run_analysis("custom_v1", func=my_func, discretize="discretize")
# ```