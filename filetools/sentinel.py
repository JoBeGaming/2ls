# Not exposed to any external APIs

# (c) Joshua Bettin, 2025

from typing import Any

__all__ = [
    "_is_sentinel",
    "_sentinel",
    "no_arg_given"
]


_sentinel: Any = object()

def _is_sentinel(obj: object) -> bool:
    return obj is _sentinel

no_arg_given = _is_sentinel