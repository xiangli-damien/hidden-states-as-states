"""Read-only ablation tables/figures; no fit or data-dependent model selection by labels."""

from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from ..experiments.fitting import select_candidate


def render_ablation(root, records, cfg, recipe):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    grouped = {}
    for item in records.values():
        if item["status"] == "complete":
            key = (
                item["view"],
                item["layer"],
                item["method"],
                item["rank"],
                item["seed"],
            )
            grouped.setdefault(key, []).append(item["record"])
    selected = []
    for (view, layer, method, rank, seed), candidates in sorted(grouped.items()):
        if seed != cfg.seed:
            continue
        for criterion in recipe["criteria"] if method != "kmeans" else ["icl"]:
            for tolerance in recipe["icl_tolerances"] if method != "kmeans" else [0.0]:
                try:
                    best = select_candidate(
                        candidates,
                        replace(
                            cfg.cluster,
                            method=method,
                            selection_criterion=criterion,
                            parsimony_tolerance=tolerance,
                        ),
                    )
                except ValueError:
                    continue
                selected.append(
                    dict(
                        view=view,
                        layer=layer,
                        method=method,
                        rank=rank,
                        criterion=(
                            criterion if method != "kmeans" else "negative_silhouette"
                        ),
                        tolerance=tolerance,
                        k=best["k"],
                        score=best["criterion"],
                        candidates=len(candidates),
                        converged=sum(c.get("converged") is True for c in candidates),
                        fit_path=best["fit_path"],
                    )
                )
    selection = pd.DataFrame(selected)
    selection.to_csv(root / "selection_ablation.csv", index=False)
    matched = []
    for (view, layer, method, rank, seed), candidates in sorted(grouped.items()):
        if method != "gmm" or seed != cfg.seed:
            continue
        try:
            baseline = select_candidate(
                candidates,
                replace(
                    cfg.cluster,
                    method="gmm",
                    selection_criterion="icl",
                    parsimony_tolerance=recipe["reference_tolerance"],
                ),
            )
        except ValueError:
            continue
        for r in recipe["ranks"]:
            for candidate in grouped.get((view, layer, "mfa", r, seed), []):
                if candidate["k"] == baseline["k"]:
                    matched.append(
                        dict(
                            view=view,
                            layer=layer,
                            rank=r,
                            k=baseline["k"],
                            icl=candidate.get("icl", candidate["criterion"]),
                            bic=candidate["bic"],
                            converged=candidate.get("converged"),
                            seconds=candidate["seconds"],
                            fit_path=candidate["fit_path"],
                        )
                    )
    pd.DataFrame(matched).to_csv(root / "rank_ablation_matched_k.csv", index=False)
    stability = []
    for key, candidates in grouped.items():
        view, layer, method, rank, seed = key
        if seed == cfg.seed:
            continue
        originals = {
            c["k"]: c for c in grouped.get((view, layer, method, rank, cfg.seed), [])
        }
        for candidate in candidates:
            baseline = originals.get(candidate["k"])
            if not baseline:
                continue
            with (
                np.load(Path(baseline["fit_path"]) / "assignments.npz") as a,
                np.load(Path(candidate["fit_path"]) / "assignments.npz") as b,
            ):
                for assignment in ("posterior", "nearest"):
                    stability.append(
                        dict(
                            view=view,
                            layer=layer,
                            method=method,
                            rank=rank,
                            k=candidate["k"],
                            seed=seed,
                            assignment=assignment,
                            both_converged=bool(
                                candidate.get("converged") and baseline.get("converged")
                            ),
                            ari=adjusted_rand_score(a[assignment], b[assignment]),
                        )
                    )
    pd.DataFrame(stability).to_csv(root / "seed_stability.csv", index=False)
    fragment = "<h2>Ablation results</h2><p><a href='selection_ablation.csv'>K / ICL / BIC / tolerance</a> · <a href='rank_ablation_matched_k.csv'>Rank at fixed GMM K</a> · <a href='seed_stability.csv'>Seed stability (ARI)</a></p>"
    if len(selection):
        for method in ("gmm", "mfa"):
            part = selection[
                (selection.view == "post")
                & (selection.method == method)
                & (selection.criterion == "icl")
            ]
            if part.empty:
                continue
            matrix = part.pivot(
                index=["rank", "tolerance"], columns="layer", values="k"
            )
            fig, ax = plt.subplots(figsize=(12, max(3, len(matrix) * 0.25)))
            im = ax.imshow(
                matrix, aspect="auto", interpolation="nearest", cmap="viridis"
            )
            ax.set_xticks(range(len(matrix.columns)), matrix.columns)
            ax.set_yticks(
                range(len(matrix)), [f"r={r}, tol={t:.1%}" for r, t in matrix.index]
            )
            ax.set(
                xlabel="Captured layer (0 = embedding)",
                title=f"{method.upper()}: selected K; converged candidates only",
            )
            fig.colorbar(im, ax=ax, label="K")
            fig.tight_layout()
            path = f"{method}_icl_tolerance.png"
            fig.savefig(root / path, dpi=160)
            plt.close(fig)
            fragment += f"<img style='max-width:100%' src='{path}'>"
    frame = pd.DataFrame(matched)
    if len(frame):
        part = frame[(frame.view == "post") & frame.converged.eq(True)]
        if len(part):
            matrix = part.pivot(index="rank", columns="layer", values="icl")
            # ICL is only compared within the same layer and fixed K.
            n_samples = len(
                pd.read_parquet(
                    root / "snapshots/post_rows.parquet", columns=["sample_id"]
                )
            )
            matrix = matrix.subtract(matrix.min(axis=0), axis=1) / n_samples
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(
                matrix, aspect="auto", interpolation="nearest", cmap="magma_r"
            )
            ax.set_xticks(range(len(matrix.columns)), matrix.columns)
            ax.set_yticks(range(len(matrix.index)), matrix.index)
            ax.set(
                xlabel="Captured layer",
                ylabel="MFA rank",
                title="Fixed-K rank ablation: excess ICL per sample (lower is better)",
            )
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(root / "mfa_rank_icl.png", dpi=160)
            plt.close(fig)
            fragment += "<img style='max-width:100%' src='mfa_rank_icl.png'>"
    return fragment
