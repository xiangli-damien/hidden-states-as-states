"""Central fitting dispatch; every method returns the same portable ClusterModel."""

METHODS = {
    "gmm": {
        "criterion": "ICL = BIC + 2 entropy",
        "backends": ["cpu", "sklearn", "gpu"],
    },
    "mfa": {"criterion": "ICL = BIC + 2 entropy", "backends": ["cpu", "gpu"]},
    "kmeans": {
        "criterion": "negative sampled silhouette",
        "backends": ["cpu", "sklearn"],
    },
    "minibatch_kmeans": {
        "criterion": "negative sampled silhouette",
        "backends": ["cpu", "sklearn"],
    },
}


def fit_method(method, X, k, seed, parameters):
    if method == "gmm":
        from .gmm import _fit_gmm

        return _fit_gmm(X, k, seed, **parameters)
    if method == "mfa":
        from .mfa import fit_mfa

        return fit_mfa(X, k, seed, **parameters)
    if method in ("kmeans", "minibatch_kmeans"):
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from .kmeans import KMeansModel

        estimator = KMeans if method == "kmeans" else MiniBatchKMeans
        fitted = estimator(k, random_state=seed, **parameters).fit(X)
        return KMeansModel(
            fitted.cluster_centers_,
            n_iter_=int(fitted.n_iter_),
            converged_=bool(fitted.n_iter_ < parameters.get("max_iter", 300)),
            inertia_=float(fitted.inertia_),
        )
    raise ValueError(f"Unknown clustering method: {method}")
