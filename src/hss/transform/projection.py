"""Portable train-fitted scaling and PCA, independent of experiment execution."""

import numpy as np


class Projection:
    def __init__(self, mean, scale, pca_mean, components, whitening):
        self.mean, self.scale = mean, scale
        self.pca_mean, self.components, self.whitening = pca_mean, components, whitening

    @classmethod
    def fit(cls, X, config, seed):
        from sklearn.decomposition import PCA

        mean = (
            X.mean(0, dtype=np.float64) if config.standardize else np.zeros(X.shape[1])
        )
        scale = (
            np.sqrt(X.var(0, dtype=np.float64))
            if config.standardize
            else np.ones(X.shape[1])
        )
        scale[scale < 1e-12] = 1
        pca_mean, components, whitening = (
            np.zeros(X.shape[1]),
            np.empty((0, X.shape[1])),
            np.empty(0),
        )
        if config.pca_components is not None:
            if config.pca_components > min(X.shape):
                raise ValueError(
                    "PCA components exceed training samples/features; no silent dimension change"
                )
            pca = PCA(
                n_components=config.pca_components,
                svd_solver="randomized",
                random_state=seed,
            )
            pca.fit((X - mean) / scale)
            pca_mean, components = pca.mean_, pca.components_
            whitening = (
                np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
                if config.whiten
                else np.ones(len(components))
            )
        return cls(mean, scale, pca_mean, components, whitening)

    def transform(self, X):
        Z = (X - self.mean) / self.scale
        return (
            (Z - self.pca_mean) @ self.components.T / self.whitening
            if len(self.components)
            else Z
        )

    def inverse(self, Z):
        X = (
            (Z * self.whitening) @ self.components + self.pca_mean
            if len(self.components)
            else Z
        )
        return X * self.scale + self.mean

    def arrays(self):
        return {
            k: getattr(self, k)
            for k in ("mean", "scale", "pca_mean", "components", "whitening")
        }
