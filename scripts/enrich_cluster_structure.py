"""Add shared display embeddings and interpretable profiles to frozen labels.

Example:
  python scripts/enrich_cluster_structure.py --report results/cluster-report \
    --states /path/to/layer_28.npy --rows /path/to/rows.parquet

Embeddings are display artifacts. No clustering is run or changed here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_samples
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from hss.analysis.cluster_profiles import (
    association, neighborhood_audit, participation_dimension, wilson_interval,
)
from hss.analysis.cluster_structure import empirical_spectrum
from hss.experiments.artifacts import file_digest, save_json


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def prepare(report_dir, states, rows):
    root = Path(report_dir)
    report = json.loads((root / "structure.json").read_text())
    meta = pd.read_parquet(rows)
    members = pd.read_parquet(root / "members.parquet")
    X = np.load(states, mmap_mode="r")
    ids = np.array([s["id"] for s in report["samples"]], dtype=str)
    if (X.shape != (report["n_samples"], report["hidden_dim"])
            or not np.array_equal(ids, meta.sample_id.astype(str))
            or not np.array_equal(ids, members.sample_id.astype(str))):
        raise ValueError("Raw state rows do not match the frozen report")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite raw states")
    for name, labels in report["assignments"].items():
        if not np.array_equal(labels, members[name]):
            raise ValueError(f"Frozen assignments mismatch: {name}")
    before = {k: hashlib.sha256(np.asarray(y, dtype=np.int32).tobytes()).hexdigest()
              for k, y in report["assignments"].items()}
    for sample, row in zip(report["samples"], meta.itertuples()):
        sample.update(level=int(row.level), finish_reason=str(row.finish_reason))
    result = dict(created_at=time.time(), states_sha256=file_digest(Path(states)),
                  rows_sha256=file_digest(Path(rows)), source_sha256=file_digest(root / "structure.json"),
                  driver_sha256=file_digest(Path(__file__)), sklearn_version=sklearn.__version__,
                  assignment_sha256=before, methods={}, embeddings={}, agreement={})
    import hss.analysis.cluster_profiles as profiles_module
    import hss.analysis.cluster_structure as structure_module
    result["analysis_sha256"] = {str(Path(m.__file__).name): file_digest(Path(m.__file__))
                                 for m in [profiles_module, structure_module]}
    result["baseline"] = dict(accuracy=float(meta.label.mean()),
                              median_tokens=float(meta.n_tokens.median()),
                              categories={str(k): int(v) for k, v in meta.category.value_counts().items()},
                              levels={str(k): int(v) for k, v in meta.level.value_counts().items()},
                              finish_reasons={str(k): int(v) for k, v in meta.finish_reason.value_counts().items()})
    # Explicitly descriptive expectation under the batch's category x level mix.
    expected = meta.groupby(["category", "level"]).label.transform("mean").to_numpy()
    rng = np.random.default_rng(42)
    silhouette_rows = np.sort(rng.choice(len(X), min(1500, len(X)), replace=False))
    distances = pairwise_distances(X[silhouette_rows], metric="euclidean", n_jobs=1)
    np.fill_diagonal(distances, 0)
    anchors = np.sort(rng.choice(len(X), min(512, len(X)), replace=False))
    result["silhouette_rows"] = silhouette_rows.tolist()
    result["audit_anchors"] = anchors.tolist()
    for name, fit in report["methods"].items():
        y = np.asarray(report["assignments"][name])
        silhouettes = silhouette_samples(distances, y[silhouette_rows], metric="precomputed")
        entry = dict(silhouette=float(silhouettes.mean()), silhouette_n=len(silhouette_rows),
                     clusters=[], local_coordinates=np.zeros((len(X), 2)),
                     radius=np.zeros(len(X)))
        for c in fit["clusters"]:
            k = c["cluster"]
            idx = np.flatnonzero(y == k)
            if not len(idx):
                entry["clusters"].append(dict(cluster=k, n=0))
                continue
            part = np.asarray(X[idx], dtype=np.float64)
            spec, axes = empirical_spectrum(part, components=32, seed=42+k)
            centered = part - part.mean(0)
            local = centered @ axes[:2].T
            entry["local_coordinates"][idx, :local.shape[1]] = local
            radius = np.linalg.norm(centered, axis=1)
            entry["radius"][idx] = radius
            categories = meta.iloc[idx].category.value_counts()
            levels = meta.iloc[idx].level.value_counts()
            selected = np.flatnonzero(y[silhouette_rows] == k)
            info = dict(cluster=k, n=len(idx), accuracy_interval=wilson_interval(int(meta.iloc[idx].label.sum()), len(idx)),
                        expected_accuracy=float(expected[idx].mean()),
                        composition_accuracy_delta=float(meta.iloc[idx].label.mean() - expected[idx].mean()),
                        effective_dimension=participation_dimension(part),
                        levels={str(key): int(v) for key, v in levels.items()},
                        category_lift={str(key): float(v/len(idx)/(result["baseline"]["categories"][str(key)]/len(X)))
                                       for key, v in categories.items()},
                        token_quantiles=np.quantile(meta.iloc[idx].n_tokens, [.1, .5, .9]).tolist(),
                        finish_reasons={str(key): int(v) for key, v in meta.iloc[idx].finish_reason.value_counts().items()},
                        silhouette=float(silhouettes[selected].mean()) if len(selected) else None,
                        silhouette_n=len(selected),
                        pc_log_length_correlation=[corr(local[:, j], np.log1p(meta.iloc[idx].n_tokens)) for j in range(local.shape[1])],
                        pc_level_correlation=[corr(local[:, j], meta.iloc[idx].level) for j in range(local.shape[1])],
                        representatives=idx[np.argsort(radius)[:3]].tolist(),
                        peripheral=idx[np.argsort(radius)[-3:][::-1]].tolist(),
                        pc1_low=idx[np.argsort(local[:, 0])[:3]].tolist() if local.shape[1] else [],
                        pc1_high=idx[np.argsort(local[:, 0])[-3:][::-1]].tolist() if local.shape[1] else [])
            entry["clusters"].append(info)
        entry["local_coordinates"] = entry["local_coordinates"].tolist()
        entry["radius"] = entry["radius"].tolist()
        result["methods"][name] = entry
        emit("profiles_complete", method=name, silhouette=entry["silhouette"])
    names = list(report["methods"])
    for a in names:
        result["agreement"][a] = {b: association(report["assignments"][a], report["assignments"][b]) for b in names}
    with np.load(root / "display_projection.npz") as projection:
        pca2 = projection["coordinates"]
    result["embeddings"]["pca"] = dict(coordinates=pca2.tolist(), label="全局 PCA · 2D", audit=neighborhood_audit(X, pca2, anchors))
    centered = np.asarray(X) - np.mean(X, axis=0)
    u, singular, axes = randomized_svd(centered, n_components=50, n_iter=5, random_state=42)
    pca50 = u * singular
    retained = float(np.square(singular).sum() / np.square(centered, dtype=np.float64).sum())
    result["preprocessing"] = dict(display_only=True, components=50, variance_fraction=retained,
                                   method="randomized_svd", n_iter=5, random_state=42,
                                   audit=neighborhood_audit(X, pca50, anchors), normalization=False)
    np.savez_compressed(root / "display_pca50.npz", coordinates=pca50, axes=axes, sample_ids=ids)
    emit("pca_display_preprocessing", variance_fraction=retained)
    for perplexity, seed in [(30, 42), (80, 42), (30, 43)]:
        start = time.monotonic()
        options = dict(n_components=2, perplexity=perplexity, init="random", random_state=seed,
                       learning_rate="auto", method="barnes_hut", angle=.5, n_jobs=2)
        # sklearn renamed n_iter in 1.5; support the package's >=1.3 dependency.
        import inspect
        options["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 1000
        model = TSNE(**options)
        xy = model.fit_transform(pca50)
        key = f"tsne_p{perplexity}_s{seed}"
        result["embeddings"][key] = dict(coordinates=xy.tolist(),
            label=f"t-SNE · perplexity {perplexity} · seed {seed}", parameters=options,
            kl_divergence=float(model.kl_divergence_), seconds=time.monotonic()-start,
            audit=neighborhood_audit(X, xy, anchors))
        emit("embedding_complete", key=key, seconds=time.monotonic()-start,
             audit=result["embeddings"][key]["audit"])
    for key, labels in report["assignments"].items():
        assert before[key] == hashlib.sha256(np.asarray(labels, dtype=np.int32).tobytes()).hexdigest()
    report["exploration"] = result
    save_json(root / "exploration.json", report)
    render(root)


def render(root):
    root = Path(root)
    report = json.loads((root / "exploration.json").read_text())
    template = Path(__file__).parents[1] / "src/hss/reporting/templates/cluster_explorer.html"
    save_json(root / "render_manifest.json", dict(
        exploration_sha256=file_digest(root / "exploration.json"),
        template_sha256=file_digest(template), renderer_sha256=file_digest(Path(__file__))))
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (root / "index.html").write_text(template.read_text().replace("__CLUSTER_DATA__", payload))
    export_figure(report, root)
    emit("render_complete", path=str(root / "index.html"))


def export_figure(report, root):
    """Portable scientific figure with shared coordinates and equal aspect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    palette = ['#24678d', '#cd7038', '#25876f', '#b44967', '#7367aa', '#b59628',
               '#459aa6', '#825849', '#747b87', '#dc929d', '#4f7844', '#a555ad',
               '#5591d0', '#93791b', '#b2563a', '#43a996', '#687fc0', '#cc982b',
               '#8c6981', '#83a652', '#506770']
    cmap = ListedColormap(palette)
    e = report['exploration']
    xy = np.asarray(e['embeddings']['tsne_p30_s42']['coordinates'])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for ax, name in zip(axes[0], ['kmeans_k21', 'gmm_k21', 'mfa_r8_k21']):
        ax.scatter(*xy.T, c=report['assignments'][name], cmap=cmap,
                   vmin=-.5, vmax=20.5, s=5, alpha=.65, linewidths=0, rasterized=True)
        ax.set_title(name.replace('_', ' '), loc='left', fontsize=13)
    correct = np.array([s['correct'] for s in report['samples']])
    for value, color, label in [(0, '#c8754d', 'Incorrect'), (1, '#228876', 'Correct')]:
        subset = xy[correct == value]
        axes[1, 0].scatter(*subset.T, s=5, color=color, alpha=.65, linewidths=0,
                           rasterized=True, label=label)
    axes[1, 0].legend(frameon=False, markerscale=3, loc='upper right')
    axes[1, 0].set_title('Correctness (descriptive)', loc='left', fontsize=13)
    for ax, values, title, label in [
        (axes[1, 1], [s['level'] for s in report['samples']], 'Question difficulty', 'MATH level'),
        (axes[1, 2], np.log10([s['tokens'] for s in report['samples']]), 'Response length', 'log10 tokens'),
    ]:
        points = ax.scatter(*xy.T, c=values, cmap='viridis', s=5, alpha=.7, linewidths=0, rasterized=True)
        fig.colorbar(points, ax=ax, shrink=.65, label=label)
        ax.set_title(title, loc='left', fontsize=13)
    for ax in axes.flat:
        ax.set_aspect('equal', adjustable='box')
        ax.set(xticks=[], yticks=[], xlabel='t-SNE 1', ylabel='t-SNE 2')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Qwen2-7B-Instruct / MATH 5,000 / raw final post-RMS response means\n'
                 'Shared display: PCA50 + t-SNE (perplexity 30, seed 42). Cluster IDs are method-specific.', fontsize=14)
    fig.savefig(root / 'cluster_atlas.png', dpi=200)
    fig.savefig(root / 'cluster_atlas.pdf', dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--states")
    parser.add_argument("--rows")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render(args.report)
    else:
        if not args.states or not args.rows:
            parser.error("--states and --rows are required to compute diagnostics")
        with threadpool_limits(2):
            prepare(args.report, args.states, args.rows)
