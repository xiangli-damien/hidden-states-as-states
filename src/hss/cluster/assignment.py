"""Shared inference policy for portable models; independent of experiment fitting."""

import numpy as np


def assign(model, X, assignment):
    if assignment not in ("nearest", "posterior"):
        raise ValueError("assignment must be nearest or posterior")
    if assignment == "posterior":
        return model.predict(X)
    centers = np.asarray(model.centers(), dtype=np.float64)
    distance = (
        (X * X).sum(1)[:, None]
        - 2 * X @ centers.T
        + (centers * centers).sum(1)[None, :]
    )
    return distance.argmin(1).astype(np.int32)
