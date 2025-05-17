import ast
import sys
from typing import Any, Callable
import importlib.util
import inspect
import os


class DocTestNotPassedError(Exception):
    """Error that indicates test not passing"""


def test(line: str, func: str, index: int) -> None:
    ...
    raise DocTestNotPassedError(f"line {index} of docstring of {func}: {line} did not pass tests")


def test_doc(doc: str, func: str) -> None:
    index = 0
    for ln in doc.split("\n"):
        if ">>>" in ln:
            test(ln.split(">>>")[1], func, index)
        index += 1


def insert_future_annotations(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines.insert(0, "from __future__ import annotations\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def getdocs(file: str) -> list[Callable]:
    insert_future_annotations(file)
    mod_name = os.path.splitext(os.path.basename(file))[0]
    spec = importlib.util.spec_from_file_location(mod_name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callables: list[Callable[..., Any]] = []
    for _, obj in vars(module).items():
        print(_, obj)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            callables.append(obj)
    return callables


def doctest(file: str):
    for f in getdocs(file):
        if f.__doc__:
            print(f)
            test_doc(f.__doc__, f.__name__)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SyntaxError("Usage: python doctest.py <python_file.py>")
    else:
        try:
            doctest(sys.argv[1])
        except Exception as E:
            has_import = False
            with open(sys.argv[1], "r") as file:
                if file.readlines()[0] == "from __future__ import annotations\n":
                    has_import = True
            if has_import:
                with open(sys.argv[1], "w") as file:
                    file.writelines(
                        open(sys.argv[1], "r").readlines()
                    )
            raise E