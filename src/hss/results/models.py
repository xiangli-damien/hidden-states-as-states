"""Load portable fitted arrays without starting data preparation or fitting."""

import json
from pathlib import Path
import numpy as np
from ..cluster.registry import rebuild_model
from ..transform.projection import Projection
from ..experiments.artifacts import save_json, save_npz


def save_layer(root, layer, model, projection, scan):
    path = root / "models" / f"layer_{layer}"
    save_npz(path / "model.npz", **model.state_arrays())
    save_npz(path / "projection.npz", **projection.arrays())
    save_json(path / "model.json", model.config())
    save_json(path / "selection.json", scan)


def load_layer(root, layer):
    path = Path(root) / "models" / f"layer_{layer}"
    config = json.loads((path / "model.json").read_text())
    with np.load(path / "model.npz", allow_pickle=False) as f:
        model = rebuild_model(config, {k: f[k] for k in f.files})
    with np.load(path / "projection.npz", allow_pickle=False) as f:
        projection = Projection(**{k: f[k] for k in f.files})
    return model, projection, json.loads((path / "selection.json").read_text())
