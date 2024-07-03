import importlib

from . import constants


def __getattr__(name):
    if name in constants.CATALOGS.keys():
        module = importlib.import_module(
            f".catalogs.{constants.CATALOGS[name]}", __name__
        )
        return getattr(module, name)
    elif name in constants.DATASETS:
        module = importlib.import_module(".datasets", __name__)
        return getattr(module, name)

    module = importlib.import_module(f".{name}", __name__)
    return module
