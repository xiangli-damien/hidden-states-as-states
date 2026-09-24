"""Compatibility imports. New adapters and callers use hss.data."""

from ..data import DataSpec, CachedStates, prepare
from ..data.openact import discover, sentence_boundaries

__all__ = ["DataSpec", "CachedStates", "prepare", "discover", "sentence_boundaries"]
