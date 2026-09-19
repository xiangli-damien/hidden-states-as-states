"""One preparation API; numerical code consumes CachedStates, not source formats."""

from .spec import DataSpec
from .cache import CachedStates


def prepare(spec, cache_root):
    spec.validate()
    if spec.source_format == "openact":
        from .openact import prepare as backend
    else:
        from .arrays import prepare as backend
    return backend(spec, cache_root)


__all__ = ["DataSpec", "CachedStates", "prepare"]
