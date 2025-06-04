from typing import Any

def dumpdict(file: str, *args: tuple[dict[Any, Any], ...]) -> None:
    for arg in args:
        assert isinstance(arg, dict)
        try:
            with open(file, "a", errors="strict") as f:
                for key in arg:
                    if not key in ("__builtins__"):
                        f.write(f"{key}: {arg[key]}\n")
        except (FileNotFoundError, FileExistsError):
            with open(file, "x"):
                pass
            dumpdict(file, *args)

def dump(file: str, *args: tuple[Any]) -> None:
    for arg in args:
        assert isinstance(arg, dict)
        try:
            with open(file, "a", errors="strict") as f:
                f.write(f"{arg}\n")
        except (FileNotFoundError, FileExistsError):
            with open(file, "x"):
                pass
            dump(file, *args)