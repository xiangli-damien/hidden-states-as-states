"""Exercise full-dimensional real data and every rank before a long CPU sweep.

Short smoke fits are explicitly NOT scientific/converged study results.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from hss.data import prepare
from hss.experiments.artifacts import save_json, save_npz
from hss.experiments.cluster_ablation import load_recipe, method_config
from hss.experiments.fitting import common_parameters, selection_metrics
from hss.cluster.methods import fit_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen_math_ablation.toml")
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    recipe, base = load_recipe(args.config)
    # Last post-RMS layer alone is enough to exercise D and N; the production
    # study separately freezes and verifies every requested layer.
    spec = replace(base.data, layers=[-1])
    data = prepare(spec, base.execution.cache_root)
    X = np.asarray(data.array(data.layers()[-1]), dtype=np.float64)
    assert X.shape == (base.data.expected_samples, data.state_dim())
    assert np.isfinite(X).all()
    root = Path(args.directory)
    records = []
    with threadpool_limits(limits=base.execution.threads_per_worker):
        for method, ranks in [("kmeans", [0]), ("gmm", [0]), ("mfa", recipe["ranks"])]:
            for rank in ranks:
                cfg = method_config(base, recipe, method, rank)
                cfg.cluster = replace(cfg.cluster, max_iter=8, n_init=1)
                start = time.time()
                model = fit_method(
                    method, X, 2, base.seed, common_parameters(cfg.cluster)
                )
                labels = model.predict(X)
                assert (
                    labels.shape == (len(X),) and labels.min() >= 0 and labels.max() < 2
                )
                assert model.centers().shape == (2, X.shape[1])
                scores = selection_metrics(
                    model, X, method, cfg.cluster.chunk_size, base.seed
                )
                assert np.isfinite(scores["criterion"])
                if method == "mfa":
                    assert np.min(np.diff(model.history_)) > -1e-5
                    assert np.isclose(model.score(X), model.history_[-1], atol=1e-7)
                folder = root / f"{method}_r{rank}"
                save_npz(folder / "model.npz", **model.state_arrays())
                save_npz(folder / "assignments.npz", labels=labels)
                item = dict(
                    method=method,
                    rank=rank,
                    seconds=time.time() - start,
                    scores=scores,
                    model=model.config(),
                )
                save_json(folder / "smoke.json", item)
                records.append(item)
                print(
                    json.dumps({k: item[k] for k in ("method", "rank", "seconds")}),
                    flush=True,
                )
    save_json(
        root / "_SUCCESS.json",
        dict(
            status="smoke_passed",
            n=len(X),
            d=X.shape[1],
            model=data.info["model"],
            fits=len(records),
            note="Eight-iteration smoke fits; not production results.",
        ),
    )


if __name__ == "__main__":
    main()
