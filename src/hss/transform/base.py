from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from ..types import BatchFactory


class Transform(ABC):
    @abstractmethod
    def fit(self, factory: BatchFactory) -> None: ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def inverse(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def config(self) -> Dict[str, Any]: ...

    def state_arrays(self) -> Dict[str, np.ndarray]:
        return {}

    def load_state_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        pass
