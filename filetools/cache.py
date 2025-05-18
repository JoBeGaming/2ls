# Not exposed to any external APIs

# (c) Joshua Bettin, 2025

from collections.abc import Callable
from typing import Any


__all__: list[str] = [
    "cache",
    "CacheKey",
    "CacheType"
]


CacheKey = tuple[Any, ...] | tuple[Any, list[Any]]
CacheType = dict[CacheKey, Any]


def _copy(_from: object, _to: object) -> None:
    attributes = {
        "__name__", 
        "__doc__", 
        "__annotations__", 
        "__module__",
        "__defaults__",
        "__kwdefaults__"
    }
    for attr in attributes:
        try:
            setattr(_to, attr, getattr(_from, attr, None))
        except NameError:
            pass


def cache(obj: Callable) -> Callable:
    _cache: CacheType = {}

    def make_key(args: tuple[Any], kwargs: dict[Any, Any]) -> CacheKey:
        key = args
        if kwargs:
            key += tuple(sorted(kwargs.items()))
        return key

    def wrapper(*args, **kwargs):
        key = make_key(args, kwargs)
        if key in _cache:
            return _cache[key]
        result = obj(*args, **kwargs)
        _cache[key] = result
        return result

    _copy(obj, wrapper)
    return wrapper