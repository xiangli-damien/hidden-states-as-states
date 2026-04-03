from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from openact_core import DataLoader, LoadedData, Run
from hss import NumpyProvider


def provider_from_run(run_path: Union[str, Path], *, reduction: str = 'mean', layers: Optional[List[int]] = None, labels: Optional[List[str]] = None, only_valid: bool = True, start_pct: Optional[float] = None, end_pct: Optional[float] = None, n_points: Optional[int] = None, dtype: str = 'float32') -> Tuple[NumpyProvider, LoadedData]:
    loader = DataLoader(Run(run_path), only_valid=only_valid)
    if n_points is not None:
        data = loader.load_trajectories(n_points=n_points, layers=layers, labels=labels, dtype=dtype)
    elif start_pct is not None or end_pct is not None:
        data = loader.load_range(start_pct=start_pct, end_pct=end_pct, reduction=reduction, layers=layers, labels=labels, dtype=dtype)
    else:
        data = loader.load(reduction=reduction, layers=layers, labels=labels, dtype=dtype)
    layer_ids = layers or list(range(data.hidden_states.shape[-2]))
    return NumpyProvider(data.hidden_states, layer_ids=layer_ids), data


def discover_tasks(base_dir: Union[str, Path]) -> Dict[str, Path]:
    base = Path(base_dir)
    return {d.name: d for d in sorted(base.iterdir()) if d.is_dir() and (d / 'manifest.json').exists()}
