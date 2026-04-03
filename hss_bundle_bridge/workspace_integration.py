from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from .adapter import BundleAdapter, OpenedView
from .protocol import SelectionSpec


def open_bundle_view(
    bundle_dir: str | Path,
    view_name: str,
    *,
    selection: Optional[SelectionSpec] = None,
    layers: Optional[Sequence[int]] = None,
) -> OpenedView:
    return BundleAdapter(bundle_dir).open_view(view_name, selection=selection, layers=layers)


def load_workspace_from_bundle(
    workspace: Any,
    *,
    bundle_dir: str | Path,
    view_name: str,
    selection: Optional[SelectionSpec] = None,
    layers: Optional[Sequence[int]] = None,
    persist: bool = True,
) -> OpenedView:
    view = open_bundle_view(bundle_dir, view_name, selection=selection, layers=layers)
    workspace.load_data(view.to_experiments_loaded_inputs(), persist=persist)
    return view


def run_experiment_from_bundle(
    cfg: Any,
    *,
    bundle_dir: str | Path,
    view_name: str,
    selection: Optional[SelectionSpec] = None,
    layers: Optional[Sequence[int]] = None,
    return_view: bool = False,
) -> Any:
    from experiments.runner import run_experiment

    view = open_bundle_view(bundle_dir, view_name, selection=selection, layers=layers)
    outputs = run_experiment(cfg, loaded=view.to_experiments_loaded_inputs())
    if return_view:
        return outputs, view
    return outputs
