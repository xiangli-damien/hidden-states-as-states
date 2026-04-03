from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .adapter import BundleAdapter, OpenedView
from .protocol import SelectionSpec


def load_bundle_view(
    bundle_dir: str | Path,
    view_name: str,
    *,
    selection: Optional[SelectionSpec] = None,
    layers: Optional[Sequence[int]] = None,
) -> OpenedView:
    return BundleAdapter(bundle_dir).open_view(view_name, selection=selection, layers=layers)
