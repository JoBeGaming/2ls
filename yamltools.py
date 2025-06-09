__all__: list[str] = [
    "ConfigGetter"
]

class ConfigKeyNotDefinedError(Exception):
    """The specified key was not found in the config file"""


class ConfigGetter:
    """Gets the config data of a yaml at `config.yaml` unless another path is specified"""
    
    path: str = "config.yaml"
    _dict: dict = {}
    
    def __init__(self, new_path: str = "") -> None:
        if new_path:
            ConfigGetter.path = new_path

        with open(ConfigGetter.path, "r") as file:
            lines = file.readlines()

        for ln in lines:
            line = ln.split(":")
            key = line[0]
            value = line[1]
            ConfigGetter._dict[key] = value

    def __getitem__(self, key: str) -> str:
        try:
            return ConfigGetter._dict[key]
        except KeyError:
            raise ConfigKeyNotDefinedError(f"{key} not defined in {ConfigGetter.path}")