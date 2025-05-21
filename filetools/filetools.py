from pathlib import Path

from sentinel import _sentinel, _is_sentinel
from cache import cache


__all__ = [
    "convert",
    "get_conf_of",
    "get_root",
    "get_root_conf",
    "to_str",
]


def convert(path: str | Path, /) -> Path:
    """
    Internal Helper to make all files be formatted the same way.  
    We can also be use this to convert strings to Paths.  
    The `path` argument will usually be equal to the `__file__` of the file.
    """

    from pathlib import PosixPath
    Path.__new__(cls=PosixPath)

    return Path(str(path).replace("\\", "/"))


def to_str(path: str) -> str:
    """
    Internal Helper to get the string representation of a path, formatted 
    like when using convert, this however returns a string, and not a Path.
    """

    return str(path).replace("\\", "/")


@cache
def get_root(file: str = _sentinel, /, *, depth: int=2) -> Path:
    """
    Internal Helper to get the main directory.
    If `file` is not given, it is assumed to be `__file__`.
    """

    if _is_sentinel(file):
        file = __file__
    path =  Path(convert(file))
    for _ in range(depth):
        path = path.parent
    return convert(path)


@cache
def get_root_conf() -> dict[str, str]:
    """
    Get all Data stored in `config/root_conf.yaml`
    """

    return get_conf_of("root_conf")


@cache
def get_conf_of(name: str, /) -> dict[str, str]:
    """
    Get all Data stored in the given file
    """

    # PIP knows it as 'pyyaml'
    from yaml import safe_load

    file = convert(f"{get_root(__file__)}/config/{name}.yaml")
    with file.open("r", encoding="utf-8") as _file:
        attributes = safe_load(_file)
    return attributes
